"""Curriculum augmentation: inject Challenger programs or random DSL programs into clean windows."""
from __future__ import annotations

import numpy as np

from .noise import apply_program, inject_noise, random_program


def gen_augment(X, y, gens, fs, win, frac, snr_list, seed):
    """Append `frac * len(X)` noisy copies drawn from the generator callables `gens`.

    Each copy uses a generator and an SNR sampled uniformly.  A generator that fails on a
    particular draw is retried with another one; a sample on which every attempt fails is skipped.
    Returns clean rows followed by the noisy rows.
    """
    rng = np.random.default_rng(seed)
    k = int(round(len(X) * frac))
    if k <= 0 or not gens:
        return X, y
    idx = rng.choice(len(X), k, replace=False)
    rows, labs = [], []
    for j, i in enumerate(idx):
        for attempt in range(3):
            gen = gens[int(rng.integers(0, len(gens)))]
            snr = float(snr_list[int(rng.integers(0, len(snr_list)))])
            try:
                noise = np.asarray(gen(np.random.default_rng(seed + 1000 + j + attempt * 7919),
                                       float(fs), int(win)), dtype=np.float64)
                if noise.shape != (win,) or not np.all(np.isfinite(noise)):
                    raise ValueError("bad noise shape/finiteness")
                rows.append(inject_noise(X[i:i + 1], noise[None, :], snr)[0])
                labs.append(y[i])
                break
            except Exception:
                continue
    if not rows:
        return X, y
    aug = np.asarray(rows, dtype=np.float32)
    ay = np.asarray(labs, dtype=y.dtype)
    return np.concatenate([X, aug], axis=0), np.concatenate([y, ay], axis=0)


def rand_augment(X, y, fs, win, frac, snr_list, seed, pool_size=0):
    """`frac * len(X)` noisy copies under random DSL programs.  Returns only the new rows.

    `pool_size=0` draws a fresh program per sample.  `pool_size=K` draws K programs once and reuses
    them (with a fresh SNR per sample), matching the number of distinct programs an evolved
    curriculum reuses from its archive.
    """
    rng = np.random.default_rng(seed)
    k = int(round(len(X) * frac))
    if k <= 0:
        return X[:0], y[:0]
    idx = rng.choice(len(X), k, replace=False)
    pool = [random_program(rng) for _ in range(int(pool_size))] if int(pool_size) > 0 else None
    rows, labs = [], []
    for j, i in enumerate(idx):
        snr = float(snr_list[int(rng.integers(0, len(snr_list)))])
        try:
            if pool:
                p = dict(pool[int(rng.integers(0, len(pool)))])
                p["snr_db"] = snr
            else:
                p = random_program(rng, snr_db=snr)
            nx, _ = apply_program(p, X[i:i + 1], fs, seed + 4000 + j)
        except Exception:
            continue
        rows.append(nx[0])
        labs.append(y[i])
    if not rows:
        return X[:0], y[:0]
    return np.asarray(rows, dtype=np.float32), np.asarray(labs, dtype=y.dtype)


def oodval_from_curriculum(splits, gens, cfg, fs, win, seed):
    """The structure search's screening set: dev perturbed by the round's generators at the
    hardest curriculum SNR.  Falls back to clean dev when there are no generators."""
    snr = float(min(cfg["curriculum"]["aug_snrs_db"]))
    X, y = gen_augment(splits.dev_X, splits.dev_y, gens, fs, win, 1.0, [snr], seed)
    n0 = len(splits.dev_X)
    if len(X) > n0:
        return X[n0:], y[n0:]
    return splits.dev_X, splits.dev_y
