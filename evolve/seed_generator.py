"""Challenger seed: a noise-generator program `generate_noise(rng, fs, n) -> np.ndarray`.

The LLM evolves the body of this program.  It may call the `noise_lib` primitives, write loops
and conditionals, sample parameters and invent new structure on top of them.

Contract (enforced by evolve/evaluator_gen.py; violations score worst and are discarded):
  - The signature is exactly generate_noise(rng, fs, n).  The clean signal and the label are not
    available: the noise is signal-independent and additive.
  - Return a finite 1D numpy array of exactly length n.
  - Be deterministic given `rng`: draw all randomness from `rng`.
  - The evaluator locks the SNR, so amplitude has no effect.  Win by structure, not by volume.

`np` (numpy) and `noise_lib` are in scope.  See noise_lib.PRIMS for the 14 primitives and
noise_lib.prim_<name>(rng, fs, n, **params) for typed wrappers.
"""
from __future__ import annotations

import numpy as np

import noise_lib


# EVOLVE-BLOCK-START
def build_generator():
    """Return a callable generate_noise(rng, fs, n) -> np.ndarray[n].

    Evolve the COMPOSITION/LOGIC of this generator. Seed = a compound mixture of a colored
    broadband background, sparse heavy-tailed amplitude bursts, and a resonant impulse train.
    """
    def generate_noise(rng, fs, n):
        bg = noise_lib.prim_colored(rng, fs, n, beta=0.8)
        bursts = noise_lib.prim_amplitude_bursts(
            rng, fs, n, duty=0.08, n_bursts=6, band_lo_hz=1600.0, band_hi_hz=5200.0
        )
        impulses = noise_lib.prim_impulse_train(
            rng, fs, n, rate_hz=35.0, res_hz=4600.0, decay_s=0.004, jitter=0.55
        )
        return 0.25 * bg + 0.45 * bursts + 0.18 * impulses

    return generate_noise
# EVOLVE-BLOCK-END


def get_generator():
    return build_generator()
