"""T2: held-out noise families outside the Challenger DSL, used only for the final test.

None of these can be produced by the DSL primitives in `noise.py`, so T2 measures generalization
to noise structures the Challenger never generated.

Families:
  - swept_chirp        : linear swept-frequency tone (DSL has fixed-freq sinusoid only)
  - mult_speckle       : multiplicative speckle  x*(1+m)  (DSL is purely additive)
  - heavy_burst        : heavy-tailed (Student-t) impulsive bursts with non-stationary
                         inter-arrival (DSL `burst` is band-pass Gaussian on a regular grid)
  - ood_mix            : composition of the three above
"""
from __future__ import annotations

import numpy as np
import torch.nn as nn

from .metrics import balanced_accuracy
from .train import predict

T2_FAMILIES = ["swept_chirp", "mult_speckle", "heavy_burst", "ood_mix"]


def _inject_additive(X: np.ndarray, delta: np.ndarray, snr_db: float) -> np.ndarray:
    """Add `delta` to `X`, scaling delta per-sample to hit the target SNR."""
    X = np.asarray(X, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    out = np.empty_like(X, dtype=np.float32)
    for i in range(len(X)):
        d = delta[i] - delta[i].mean()
        dn = d / (np.sqrt(np.mean(d * d)) + 1e-12)
        sp = float(np.mean(X[i] ** 2))
        out[i] = (X[i] + dn * np.sqrt(sp / (10.0 ** (snr_db / 10.0)))).astype(np.float32)
    return out


def _swept_chirp(X: np.ndarray, fs: int, rng: np.random.Generator) -> np.ndarray:
    n, L = X.shape
    t = np.arange(L) / fs
    out = np.zeros((n, L), dtype=np.float64)
    for i in range(n):
        f0 = rng.uniform(50.0, 1500.0)
        f1 = rng.uniform(2000.0, 5500.0)
        k = (f1 - f0) / (t[-1] + 1e-12)
        phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t) + rng.uniform(0, 2 * np.pi)
        out[i] = np.sin(phase)
    return out


def _mult_speckle(X: np.ndarray, fs: int, rng: np.random.Generator) -> np.ndarray:
    # multiplicative field; returned as the additive-equivalent delta = x * m
    n, L = X.shape
    m = rng.standard_normal((n, L))
    # mild smoothing so it is correlated speckle, not white
    kernel = np.ones(5) / 5.0
    m = np.stack([np.convolve(m[i], kernel, mode="same") for i in range(n)])
    return np.asarray(X, dtype=np.float64) * m


def _heavy_burst(X: np.ndarray, fs: int, rng: np.random.Generator) -> np.ndarray:
    n, L = X.shape
    out = np.zeros((n, L), dtype=np.float64)
    for i in range(n):
        pos = 0.0
        # non-stationary: inter-arrival drifts across the window
        while pos < L:
            p = int(pos)
            width = int(rng.integers(8, 40))
            # Student-t heavy tail (df=2) amplitude
            amp = rng.standard_t(2)
            end = min(L, p + width)
            win = np.hanning(end - p) if end - p > 1 else np.ones(1)
            out[i, p:end] += amp * win
            gap_scale = 40.0 + 200.0 * (pos / L)  # arrivals get sparser over time
            pos += p + rng.exponential(gap_scale) + width
    return out


def _ood_mix(X: np.ndarray, fs: int, rng: np.random.Generator) -> np.ndarray:
    a = _swept_chirp(X, fs, rng)
    b = _mult_speckle(X, fs, rng)
    c = _heavy_burst(X, fs, rng)
    norm = lambda z: z / (np.sqrt(np.mean(z * z, axis=-1, keepdims=True)) + 1e-12)
    return 0.4 * norm(a) + 0.3 * norm(b) + 0.3 * norm(c)


_GEN = {
    "swept_chirp": _swept_chirp,
    "mult_speckle": _mult_speckle,
    "heavy_burst": _heavy_burst,
    "ood_mix": _ood_mix,
}


def apply_t2_family(X: np.ndarray, family: str, snr_db: float, fs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    delta = _GEN[family](np.asarray(X, dtype=np.float64), fs, rng)
    return _inject_additive(X, delta, float(snr_db))


def evaluate_t2(model: nn.Module, splits, cfg: dict, split: str = "test", seed: int = 0) -> dict:
    """Balanced accuracy on every (family, SNR) cell of `eval.t2_snrs_db`."""
    fs = int(cfg["data"]["sample_rate_hz"])
    snrs = list(cfg["eval"]["t2_snrs_db"])
    X = getattr(splits, f"{split}_X")
    y = getattr(splits, f"{split}_y")
    cells = []
    cid = 0
    for fam in T2_FAMILIES:
        for snr in snrs:
            noisy = apply_t2_family(X, fam, float(snr), fs, seed + 20000 + cid)
            pred = predict(model, noisy, cfg)
            cells.append({"family": fam, "snr_db": float(snr),
                          "balanced_accuracy": balanced_accuracy(y, pred)})
            cid += 1
    values = [row["balanced_accuracy"] for row in cells]
    return {"split": split, "mean_balanced_accuracy": float(np.mean(values)),
            "worst_cell_balanced_accuracy": float(min(values)), "cells": cells}
