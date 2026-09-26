"""Building blocks available to evolved Solver programs (see seed_solver.py).

The LLM composes these (plus raw torch.nn) into a denoise->classify network whose STRUCTURE is
evolved by the structure search. Every block below is shape-safe (documented I/O), numerically
stable, and gradient-trainable, so the LLM can COMBINE domain mechanisms it "knows about" but is unlikely to
implement correctly from scratch (Hilbert envelope, spectral-kurtosis band selection, soft-
threshold shrinkage, ...). This turns the LLM's bearing-diagnosis knowledge from "can name it"
into "can use it".

=================  MODULE MENU (which block for which noise / fault problem)  =================
Bearing-diagnosis background: a local defect strikes once per revolution of the contact,
producing a PERIODIC TRAIN OF IMPULSES that AMPLITUDE-MODULATES a high-frequency structural
RESONANCE. Diagnosis = recover those impulses / their repetition rate under noise. The classic
tool-chain is (blind deconvolution -> band selection at the resonance -> envelope demodulation ->
envelope spectrum), and the noise that breaks it is impulsive (heavy-tailed) or multiplicative.

  Target problem                     -> recommended block(s)
  -----------------------------------------------------------------------------------------
  heavy-tailed / impulsive noise     -> SoftThresholdShrinkage, ImpulseWinsorize, RobustStatPool
  multiplicative / speckle noise     -> LogEnvelopeBranch  (log turns x*(1+m) into additive)
  strong broadband low-SNR           -> ParametricFreqFilter (full rank == official 4.2M filter)
  picking the resonance band         -> KurtosisBandGate, AdaptiveBandpass
  extracting the fault signature     -> EnvelopeSpectrum (squared-envelope spectrum: BPFO/BPFI/BSF)
  swept / speed-varying modulation   -> SSMLiteBlock (long-range state-space, no CUDA kernel)
  channel recalibration (cheap)      -> SEBlock, ECABlock
  multi-timescale impulses           -> MultiScaleConv
  ClassBD blind-deconvolution core   -> ConvQuadraticOperation (the SOTA quadratic-conv front-end)

I/O convention: unless noted, blocks map (B, C, L) -> (B, C, L) and preserve length L, so they can
be inserted anywhere in the pipeline. Channel-changing blocks state their out-channels.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvQuadraticOperation(nn.Module):
    """ClassBD quadratic convolution: conv_r(x) * conv_g(x) + conv_b(x^2)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int | str = "same"):
        super().__init__()
        self.weight_r = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.weight_g = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.weight_b = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias_r = nn.Parameter(torch.empty(out_channels))
        self.bias_g = nn.Parameter(torch.ones(out_channels))
        self.bias_b = nn.Parameter(torch.zeros(out_channels))
        self.padding = padding
        nn.init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        nn.init.zeros_(self.weight_g)
        nn.init.zeros_(self.weight_b)
        fan_in = in_channels * kernel_size
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_r, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr = F.conv1d(x, self.weight_r, self.bias_r, padding=self.padding)
        xg = F.conv1d(x, self.weight_g, self.bias_g, padding=self.padding)
        xb = F.conv1d(x.pow(2), self.weight_b, self.bias_b, padding=self.padding)
        return xr * xg + xb


# --------------------------------------------------------------------------------------------
# shared helper: analytic signal / Hilbert envelope (FFT-based, batched, differentiable)
# --------------------------------------------------------------------------------------------
def analytic_envelope(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Hilbert-transform envelope |x + j*H{x}| along the last axis. Input/return (B, C, L).

    Bearing physics: the fault impulses ride as an AMPLITUDE MODULATION on a high-frequency
    resonance carrier. The envelope demodulates that carrier away and exposes the slow impulse
    train — the single most important preprocessing step in vibration bearing diagnosis.
    """
    n = x.shape[-1]
    Xf = torch.fft.fft(x, dim=-1)
    h = torch.zeros(n, device=x.device, dtype=x.dtype)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    analytic = torch.fft.ifft(Xf * h.view(1, 1, n), dim=-1)
    return (analytic.real ** 2 + analytic.imag ** 2 + eps).sqrt()


# --------------------------------------------------------------------------------------------
# channel attention (general-purpose recalibration)
# --------------------------------------------------------------------------------------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention. (B, C, L) -> (B, C, L).

    Recalibrates channels by global context: pools each channel over time, learns a per-channel
    gain in (0,1). Cheap, standard, robustness-friendly (can suppress noise-dominated channels).
    Not bearing-specific, but a reliable general amplifier of the domain blocks below.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        h = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, h), nn.ReLU(inplace=True),
            nn.Linear(h, channels), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x.mean(dim=-1)).unsqueeze(-1)
        return x * w


class ECABlock(nn.Module):
    """Efficient Channel Attention: SE without dimensionality reduction. (B, C, L) -> (B, C, L).

    Uses a 1-D conv over the channel descriptor instead of two FC layers — fewer params, no
    bottleneck. A cheaper drop-in alternative to SEBlock.
    """
    def __init__(self, channels: int, k_size: int = 5):
        super().__init__()
        k_size = k_size if k_size % 2 == 1 else k_size + 1
        self.conv = nn.Conv1d(1, 1, k_size, padding=k_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=-1, keepdim=True).transpose(1, 2)   # (B,1,C)
        s = torch.sigmoid(self.conv(s)).transpose(1, 2)     # (B,C,1)
        return x * s


# --------------------------------------------------------------------------------------------
# impulsive / heavy-tailed noise (heavy_burst)
# --------------------------------------------------------------------------------------------
class SoftThresholdShrinkage(nn.Module):
    """Deep Residual Shrinkage (DRSN) soft-threshold denoising. (B, C, L) -> (B, C, L).

    Domain principle: wavelet/shrinkage denoising sets sub-threshold coefficients (noise) to zero
    and shrinks the rest, which is exactly right for IMPULSIVE / HEAVY-TAILED noise where the
    signal of interest (fault impulses) is sparse and high-amplitude while the noise floor is
    broadband. The threshold is DATA-DRIVEN: a channel-attention subnet predicts a fraction in
    (0,1) of each channel's mean magnitude, so it adapts per-sample to the noise level. DRSN was
    introduced specifically for bearing fault diagnosis under strong noise.

    Targets: heavy_burst (heavy-tailed impulsive). Soft-threshold: sign(x)*relu(|x|-tau).
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        h = max(channels // reduction, 1)
        self.att = nn.Sequential(
            nn.Linear(channels, h), nn.ReLU(inplace=True),
            nn.Linear(h, channels), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gap = x.abs().mean(dim=-1)                 # (B,C) per-channel scale
        tau = (self.att(gap) * gap).unsqueeze(-1)  # (B,C,1) learnable threshold in (0, mean|x|)
        return torch.sign(x) * torch.clamp(x.abs() - tau, min=0.0)


class ImpulseWinsorize(nn.Module):
    """Robust amplitude clipping (winsorization) against outliers. (B, C, L) -> (B, C, L).

    Domain principle: heavy-tailed impulsive interference (e.g. Student-t bursts, electrical
    spikes) injects extreme outliers that dominate mean/variance statistics and destabilize
    training. Winsorizing caps samples to +/- k robust-scales around the per-channel median, where
    the scale is the MAD (median-absolute-deviation, MAD*1.4826 ~ sigma for Gaussian). k is
    learnable per channel, so the network chooses how aggressively to clip.

    Targets: heavy_burst. Complements SoftThresholdShrinkage (clip extremes vs. shrink floor).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.log_k = nn.Parameter(torch.zeros(channels))  # clip level = softplus(k) robust-scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        med = x.median(dim=-1, keepdim=True).values
        mad = (x - med).abs().median(dim=-1, keepdim=True).values + 1e-6
        lim = (F.softplus(self.log_k).view(1, -1, 1) + 0.5) * mad * 1.4826
        return med + torch.clamp(x - med, -lim, lim)


class RobustStatPool(nn.Module):
    """Robust global pooling: concat(mean, median, trimmed-max). (B, C, L) -> (B, 3C).

    Domain principle: under impulsive noise, plain global-average-pooling is corrupted by a few
    huge samples. Median is outlier-robust; a soft trimmed-max still keeps genuine fault-impulse
    energy. Feeding all three to the classifier head gives it robust and peak-sensitive summaries.

    Use as the pooling stage before the classifier head (changes L -> 3 features per channel).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1)
        median = x.median(dim=-1).values
        # soft trimmed max: mean of top-10% magnitudes (keeps impulse energy, drops single spikes)
        k = max(x.shape[-1] // 10, 1)
        topk = x.topk(k, dim=-1).values.mean(dim=-1)
        return torch.cat([mean, median, topk], dim=1)


# --------------------------------------------------------------------------------------------
# multiplicative / speckle noise (mult_speckle)
# --------------------------------------------------------------------------------------------
class LogEnvelopeBranch(nn.Module):
    """Log-envelope demodulation for MULTIPLICATIVE noise. (B, C, L) -> (B, C, L).

    Domain principle (the key trick): multiplicative/speckle corruption x*(1+m) is NOT removable
    by additive/linear filters, but taking the LOG turns a product into a SUM —
    log(signal*(1+m)) = log(signal) + log(1+m) — converting multiplicative noise into ADDITIVE
    noise that downstream conv/denoising layers CAN handle. We first take the Hilbert envelope
    (non-negative, so log is well-defined and demodulates the resonance carrier), then log-
    compress, then BatchNorm. This directly attacks the failure mode where compact/low-rank
    front-ends collapse.

    Targets: mult_speckle. Insert as a parallel branch to the raw/denoised path, then fuse.
    """
    def __init__(self, channels: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        env = analytic_envelope(x)
        return self.bn(torch.log(env + self.eps))


# --------------------------------------------------------------------------------------------
# spectral filtering / band selection
# --------------------------------------------------------------------------------------------
class ParametricFreqFilter(nn.Module):
    """Learnable frequency-domain filter with CONFIGURABLE capacity. (B, C, L) -> (B, C, L).

    Domain principle: the official ClassBD's robustness to multiplicative/heavy-tailed noise comes
    from a FULL-RANK learned spectral filter Linear(L, L) (~4.2M params) that can suppress
    arbitrary noise bands. This block reproduces that at rank=length, and lets the search trade
    capacity vs. size by lowering the rank (low-rank factorization U@V). IMPORTANT: do NOT default
    to a tiny rank — the spectral capacity is what buys low-SNR robustness. Applies a bounded
    spectral gain (sigmoid) to the complex spectrum, preserving phase.

    Targets: broadband low-SNR, heavy_burst, mult_speckle. rank=None -> full official-capacity
    (Linear(L, L) ~ 4.2M params, matching the published ClassBD filter).
    """
    def __init__(self, length: int, rank: int | None = None):
        super().__init__()
        self.length = length
        if rank is None or rank >= length:
            self.full = nn.Linear(length, length)
            self.low_rank = False
        else:
            self.u = nn.Parameter(torch.randn(length, rank) * 0.02)
            self.v = nn.Parameter(torch.randn(rank, length) * 0.02)
            self.low_rank = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Xf = torch.fft.fft(x, dim=-1)               # (B,C,L) full complex spectrum
        mag = Xf.abs()
        if self.low_rank:
            g = torch.matmul(torch.matmul(mag, self.u), self.v)
        else:
            g = self.full(mag)
        gain = torch.sigmoid(g)                      # bounded spectral gain, phase preserved
        return torch.fft.ifft(Xf * gain, dim=-1).real


class KurtosisBandGate(nn.Module):
    """Kurtogram-inspired learnable resonance-band selection. (B, C, L) -> (B, C, L).

    Domain principle: bearing fault energy is not spread evenly across frequency — it concentrates
    in the structural RESONANCE band that the impulses excite. Classical diagnosis uses the
    KURTOGRAM to find the band with the most impulsive (high-kurtosis) content, then band-pass
    filters there before envelope analysis. Here the spectrum is split into n_bands, each gets a
    learnable gate in (0,1), and the gate is BIASED by the measured per-band spectral kurtosis so
    the network is nudged toward impulsive bands while remaining trainable.

    Targets: low-SNR band selection; sharpens the input to EnvelopeSpectrum / classifier.
    """
    def __init__(self, length: int, n_bands: int = 16):
        super().__init__()
        self.n_bands = n_bands
        self.length = length
        self.gate = nn.Parameter(torch.zeros(n_bands))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Xf = torch.fft.fft(x, dim=-1)                # (B,C,L) full complex spectrum
        band = max(self.length // self.n_bands, 1)
        gains = torch.sigmoid(self.gate)             # (n_bands,)
        mask = gains.repeat_interleave(band)
        if mask.shape[0] < self.length:
            mask = torch.cat([mask, gains[-1:].expand(self.length - mask.shape[0])])
        else:
            mask = mask[:self.length]
        return torch.fft.ifft(Xf * mask.view(1, 1, -1), dim=-1).real


class AdaptiveBandpass(nn.Module):
    """Learnable Gaussian band-pass filter bank in the frequency domain. (B,C,L) -> (B, n_filt, L).

    Domain principle: envelope analysis needs a band-pass around the excited resonance FIRST.
    Instead of a fixed filter, this learns n_filt Gaussian pass-bands (center + width per filter),
    a differentiable Gabor/sinc-style filter bank. Each output channel is the input band-passed at
    one learned resonance guess — the search can specialize filters to different fault frequencies.

    Note: expands to n_filt channels (per input channel is summed). Follow with EnvelopeSpectrum.
    """
    def __init__(self, length: int, n_filt: int = 8):
        super().__init__()
        self.length = length
        self.centers = nn.Parameter(torch.linspace(0.05, 0.9, n_filt))
        self.log_width = nn.Parameter(torch.zeros(n_filt) - 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Xf = torch.fft.fft(x, dim=-1).mean(dim=1, keepdim=True)      # (B,1,L) fuse channels
        # symmetric normalized frequency axis 0..1..0 so the Gaussian bank is real-signal valid
        half = self.length // 2 + 1
        f = torch.linspace(0, 1, half, device=x.device)
        f = torch.cat([f, f[1:self.length - half + 1].flip(0)])[:self.length].view(1, -1)
        c = self.centers.view(-1, 1)
        w = (F.softplus(self.log_width) + 1e-3).view(-1, 1)
        banks = torch.exp(-0.5 * ((f - c) / w) ** 2)                 # (n_filt, L)
        out = torch.fft.ifft(Xf * banks.unsqueeze(0), dim=-1).real   # (B,n_filt,L)
        return out


# --------------------------------------------------------------------------------------------
# fault-signature feature
# --------------------------------------------------------------------------------------------
class EnvelopeSpectrum(nn.Module):
    """Squared-envelope spectrum: THE canonical bearing fault signature. (B,C,L) -> (B,C,L).

    Domain principle: a localized bearing fault produces impulses repeating at a CHARACTERISTIC
    FREQUENCY (BPFO/BPFI/BSF/FTF, set by geometry and speed). Those frequencies do NOT appear in
    the raw spectrum (they modulate a high-freq resonance), but they show up as sharp lines in the
    spectrum of the (squared) envelope. Envelope-spectrum analysis is the reference method for
    identifying WHICH bearing element is faulted. Here: Hilbert envelope -> remove DC -> magnitude
    spectrum, zero-padded back to length L so it can feed a conv stack.

    Use as a physics-informed feature branch feeding the classifier.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        env = analytic_envelope(x)
        env = env - env.mean(dim=-1, keepdim=True)
        es = torch.fft.rfft(env, dim=-1).abs()
        return F.pad(es, (0, x.shape[-1] - es.shape[-1]))


# --------------------------------------------------------------------------------------------
# long-range / modulated structure (swept_chirp, speed variation)
# --------------------------------------------------------------------------------------------
class SSMLiteBlock(nn.Module):
    """State-space / Mamba-lite block for long-range structure. (B, C, L) -> (B, C, L).

    Domain principle: under varying speed the fault frequency drifts (swept-chirp-like), and the
    modulation spans long time scales that short conv kernels miss. A state-space model captures
    long-range dependence cheaply. This is a PARALLEL, CUDA-kernel-free approximation: depthwise
    conv (local mixing) -> gated linear unit (content gating) -> learnable exponential-smoothing
    residual (approximates the SSM decay/memory) -> pointwise projection. No external mamba dep.

    Targets: swept_chirp / speed-varying modulation and long-range periodicity.
    """
    def __init__(self, channels: int, kernel: int = 7):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels)
        self.proj = nn.Conv1d(channels, channels * 2, 1)
        self.decay = nn.Parameter(torch.zeros(channels))
        self.out = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x)
        a, b = self.proj(h).chunk(2, dim=1)
        h = a * torch.sigmoid(b)                       # gated linear unit
        alpha = torch.sigmoid(self.decay).view(1, -1, 1)
        h = alpha * h + (1.0 - alpha) * x              # learnable EMA residual (SSM-lite memory)
        return self.out(h)


# --------------------------------------------------------------------------------------------
# multi-scale temporal features
# --------------------------------------------------------------------------------------------
class MultiScaleConv(nn.Module):
    """Parallel multi-kernel conv for multi-timescale impulses. (B, C_in, L) -> (B, C_out, L).

    Domain principle: fault impulses and their ringing occupy different time scales; a single
    kernel size sees only one. Parallel branches with small/medium/large kernels capture sharp
    transients and broader modulations together, then concatenate. Length-preserving (padded).
    """
    def __init__(self, in_ch: int, out_ch: int, kernels=(3, 7, 15)):
        super().__init__()
        n = len(kernels)
        chs = [out_ch // n] * (n - 1) + [out_ch - (out_ch // n) * (n - 1)]
        self.branches = nn.ModuleList(
            [nn.Conv1d(in_ch, c, k, padding=k // 2) for c, k in zip(chs, kernels)]
        )
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(torch.cat([b(x) for b in self.branches], dim=1))


__all__ = [
    "torch", "nn", "F",
    "ConvQuadraticOperation", "analytic_envelope",
    "SEBlock", "ECABlock",
    "SoftThresholdShrinkage", "ImpulseWinsorize", "RobustStatPool",
    "LogEnvelopeBranch",
    "ParametricFreqFilter", "KurtosisBandGate", "AdaptiveBandpass",
    "EnvelopeSpectrum",
    "SSMLiteBlock", "MultiScaleConv",
]
