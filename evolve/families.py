"""The five canonical noise families behind the promotion gate and the search feedback.

They are built from the Challenger DSL but never selected against a Solver, so "weakest family"
is a fixed ruler across rounds and arms.  T2 uses different, out-of-DSL families
(coevolve_bearing/t2.py).
"""
from __future__ import annotations

import numpy as np

from coevolve_bearing.noise import fixed_program

DIAG_FAMILIES = (
    "colored_bandpass",
    "harmonic_line",
    "impulsive_resonance",
    "bursty",
    "complex_mixed",
)


def diag_program(family: str, snr_db: float, rng: np.random.Generator) -> dict:
    if family == "colored_bandpass":
        return {
            "components": [
                {"prim": "colored", "params": {"beta": 1.0}},
                {"prim": "bandpass", "params": {"band_lo_hz": 900.0, "band_hi_hz": 3600.0}},
                {"prim": "arma", "params": {"a1": 0.55, "a2": -0.22}},
            ],
            "weights": [0.45, 0.40, 0.15],
            "snr_db": float(snr_db),
        }
    if family == "harmonic_line":
        return fixed_program("order_vfd", snr_db, rng)
    if family == "impulsive_resonance":
        return fixed_program("impulse_resonance", snr_db, rng)
    if family == "bursty":
        return fixed_program("burst_band", snr_db, rng)
    if family == "complex_mixed":
        return fixed_program("complex_mixed", snr_db, rng)
    raise ValueError(f"unknown noise family {family}")
