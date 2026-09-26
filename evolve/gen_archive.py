"""MAP-Elites archive for Challenger programs, keyed on waveform descriptors.

A cell is defined only by features of the generated noise (spectral type, dominant band, crest
factor, burst proxy, ...), so the archive works for free-form programs.  Each cell keeps the
highest-reward program.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from coevolve_bearing import noise as _noise


def features_from_waveform(noise_2d: np.ndarray, fs: float, win_len: int) -> dict:
    """Behavioral features from a batch of noise waveforms (shape [B, win_len]).

    Only waveform-derived quantities are used, so the descriptor is defined for any program.
    """
    n = np.asarray(noise_2d, dtype=np.float64)
    if n.ndim == 1:
        n = n[None, :]
    energy = n * n
    flat_energy = np.sort(energy, axis=-1)
    top_k = max(1, int(round(0.10 * win_len)))
    top_energy_frac = float(flat_energy[:, -top_k:].sum(axis=-1).mean() / (energy.sum(axis=-1).mean() + 1e-12))
    rms = np.sqrt(np.mean(energy, axis=-1, keepdims=True)) + 1e-12
    duty = float(np.mean(np.abs(n) > 1.5 * rms))
    env = np.abs(n)
    env_centered = env - env.mean(axis=-1, keepdims=True)
    env_var = np.mean(env_centered * env_centered, axis=-1) + 1e-12
    env_kurt = float(np.mean(np.mean(env_centered ** 4, axis=-1) / (env_var ** 2)))
    psd = np.abs(np.fft.rfft(n, axis=-1)) ** 2
    mean_psd = psd.mean(axis=0) + 1e-12
    freqs = np.fft.rfftfreq(win_len, d=1.0 / float(fs))
    prob = mean_psd / mean_psd.sum()
    centroid = float((prob * freqs).sum())
    entropy = float(-(prob * np.log(prob + 1e-12)).sum() / np.log(len(prob)))
    low = float(mean_psd[freqs <= 1000].sum() / mean_psd.sum())
    mid = float(mean_psd[(freqs > 1000) & (freqs <= 3000)].sum() / mean_psd.sum())
    high = float(mean_psd[freqs > 3000].sum() / mean_psd.sum())
    crest = float(np.max(np.abs(n)) / (np.sqrt(np.mean(n * n)) + 1e-12))
    peak_count = _noise._spectral_peak_count(mean_psd, freqs)
    spectral_type = _noise._spectral_type(low, mid, high, peak_count, env_kurt)
    dominant_band = _noise._dominant_band(low, mid, high)
    # Waveform-only heavy-burst proxy.
    low_duty = float(np.clip((0.24 - duty) / 0.21, 0.0, 1.0))
    high_top_energy = float(np.clip((top_energy_frac - 0.34) / 0.34, 0.0, 1.0))
    high_kurtosis = float(np.clip((env_kurt - 3.2) / 8.0, 0.0, 1.0))
    heavy_burst_proxy = float(np.clip(0.34 * high_top_energy + 0.33 * low_duty + 0.33 * high_kurtosis, 0.0, 1.0))
    # Waveform-only mechanism-family estimate: a single primitive counts once, a genuine mix
    # (colored + impulse + tonal ...) counts higher.
    max_band = max(low, mid, high)
    fam_broadband = entropy > 0.60
    fam_bandlimited = (not fam_broadband) and (max_band > 0.55)
    fam_impulsive = (heavy_burst_proxy > 0.40) or (env_kurt > 5.0)
    fam_tonal = int(peak_count) >= 3
    families = [name for name, on in (
        ("broadband", fam_broadband), ("bandlimited", fam_bandlimited),
        ("impulsive", fam_impulsive), ("tonal", fam_tonal)) if on]
    n_families = max(1, len(families))
    return {
        "mechanism_families": families,
        "n_mechanism_families": int(n_families),
        "centroid_hz": centroid,
        "spectral_entropy": entropy,
        "frac_0_1k": low,
        "frac_1k_3k": mid,
        "frac_3k_plus": high,
        "dominant_band": dominant_band,
        "spectral_peak_count": int(peak_count),
        "spectral_type": spectral_type,
        "crest_factor": crest,
        "temporal_top10_energy": top_energy_frac,
        "duty_cycle_est": duty,
        "envelope_kurtosis": env_kurt,
        "heavy_burst_proxy": heavy_burst_proxy,
    }


def gen_descriptor_cell(feats: dict) -> tuple:
    """Behavioral cell from waveform features only."""
    c = float(feats.get("centroid_hz", 0.0))
    e = float(feats.get("spectral_entropy", 0.0))
    hb = float(feats.get("heavy_burst_proxy", 0.0))
    c_bin = 0 if c < 1000 else (1 if c <= 3000 else 2)
    e_bin = 0 if e < 0.45 else (1 if e < 0.75 else 2)
    hb_bin = 0 if hb < 0.35 else (1 if hb < 0.70 else 2)
    stype = str(feats.get("spectral_type", "fullband"))
    band = str(feats.get("dominant_band", "unknown"))
    return (stype, band, c_bin, e_bin, hb_bin)


@dataclass
class GenMapElites:
    """Per-cell reward-elitism archive over generator programs (source strings)."""
    cells: dict = field(default_factory=dict)  # cell -> {"program","reward","feats","cell"}

    def insert(self, program, reward: float, feats: dict) -> bool:
        cell = gen_descriptor_cell(feats)
        cur = self.cells.get(cell)
        if cur is None or float(reward) > float(cur["reward"]):
            self.cells[cell] = {"program": program, "reward": float(reward), "feats": feats, "cell": cell}
            return True
        return False

    def coverage(self) -> int:
        return len(self.cells)
