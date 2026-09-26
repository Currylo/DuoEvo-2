"""Noise primitives callable from evolved Challenger programs.

Each function is a thin wrapper over the DSL implementation in `coevolve_bearing.noise` and returns
a 1D float64 array of length `n`.  A Challenger program `generate_noise(rng, fs, n)` may compose
them freely, add control flow and invent new structure on top of them.

Contract reminder (enforced by the evaluator): the program never sees the clean signal or the
label, and its output is SNR-locked, so amplitude has no effect on the reward.
"""
from __future__ import annotations

import numpy as np

from coevolve_bearing import noise as _noise

# The 14 typed primitives available to evolved generator programs.
PRIMS = tuple(_noise.RANDOM_PRIMITIVES)


def prim(name: str, rng: np.random.Generator, fs: float, n: int, **params) -> np.ndarray:
    """Generic primitive call: dispatch to coevolve_bearing.noise and return 1D length-n.

    `name` must be one of PRIMS and `params` its typed DSL parameters.  Returns shape (n,).
    """
    comp = {"prim": str(name), "params": {k: v for k, v in params.items() if v is not None}}
    out = _noise._component_noise(comp, (1, int(n)), float(fs), rng)
    return np.asarray(out, dtype=np.float64)[0]


# ----------------------------- named convenience wrappers -----------------------------
# Defaults mirror coevolve_bearing.noise._random_component.

def prim_awgn(rng, fs, n):
    return prim("awgn", rng, fs, n)


def prim_colored(rng, fs, n, beta=1.0):
    return prim("colored", rng, fs, n, beta=beta)


def prim_arma(rng, fs, n, a1=0.5, a2=-0.3):
    return prim("arma", rng, fs, n, a1=a1, a2=a2)


def prim_lowpass(rng, fs, n, cutoff_hz=2000.0):
    return prim("lowpass", rng, fs, n, cutoff_hz=cutoff_hz)


def prim_bandpass(rng, fs, n, band_lo_hz=1000.0, band_hi_hz=3000.0):
    return prim("bandpass", rng, fs, n, band_lo_hz=band_lo_hz, band_hi_hz=band_hi_hz)


def prim_sinusoid(rng, fs, n, freq_hz=50.0, n_harm=1):
    return prim("sinusoid", rng, fs, n, freq_hz=freq_hz, n_harm=int(n_harm))


def prim_line_vfd(rng, fs, n, line_hz=50.0, n_harm=3, amp_drift=0.1):
    return prim("line_vfd", rng, fs, n, line_hz=line_hz, n_harm=int(n_harm), amp_drift=amp_drift)


def prim_impulse_resonance(rng, fs, n, rate_hz=80.0, res_hz=3000.0, decay_s=0.004, jitter=0.3):
    return prim("impulse_resonance", rng, fs, n, rate_hz=rate_hz, res_hz=res_hz,
                decay_s=decay_s, jitter=jitter)


def prim_impulse_train(rng, fs, n, rate_hz=40.0, res_hz=3000.0, decay_s=0.004, jitter=0.4):
    return prim("impulse_train", rng, fs, n, rate_hz=rate_hz, res_hz=res_hz,
                decay_s=decay_s, jitter=jitter)


def prim_resonant_ringdown(rng, fs, n, freq_hz=3000.0, damping=0.04, n_excite=8):
    return prim("resonant_ringdown", rng, fs, n, freq_hz=freq_hz, damping=damping,
                n_excite=int(n_excite))


def prim_order_harmonics(rng, fs, n, f0_hz=120.0, n_harm=5, drift=0.0, sideband=0.0):
    return prim("order_harmonics", rng, fs, n, f0_hz=f0_hz, n_harm=int(n_harm),
                drift=drift, sideband=sideband)


def prim_cyclostationary(rng, fs, n, carrier_hz=3000.0, mod_hz=40.0, depth=0.6):
    return prim("cyclostationary", rng, fs, n, carrier_hz=carrier_hz, mod_hz=mod_hz, depth=depth)


def prim_burst(rng, fs, n, band_lo_hz=1200.0, band_hi_hz=4800.0, duty=0.12, block_len=112, jitter=0.6):
    return prim("burst", rng, fs, n, band_lo_hz=band_lo_hz, band_hi_hz=band_hi_hz,
                duty=duty, block_len=block_len, jitter=jitter)


def prim_amplitude_bursts(rng, fs, n, duty=0.08, n_bursts=6, band_lo_hz=1600.0,
                          band_hi_hz=5200.0, tail_df=2.2):
    return prim("amplitude_bursts", rng, fs, n, duty=duty, n_bursts=int(n_bursts),
                band_lo_hz=band_lo_hz, band_hi_hz=band_hi_hz, tail_df=tail_df)


__all__ = ["PRIMS", "prim"] + [f"prim_{p}" for p in PRIMS]
