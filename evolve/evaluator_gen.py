"""OpenEvolve evaluator for Challenger programs.

A program's noise is injected into the frozen V_edge subset at the attack SNR and scored against
the frozen Solver: reward = -(mean classification margin).  The contract keeps free program
synthesis from gaming the reward:

  1. signal-independent additive noise   (generate_noise never sees the signal or the label)
  2. locked SNR                          (each row is renormalized to unit RMS, then scaled to the
                                          attack SNR, so amplitude cannot change the reward)
  3. checked execution                   (shape, finiteness, determinism given rng, runtime cap)
  4. waveform descriptors                (MAP-Elites cell, see gen_archive.py)

The frozen state (Solver, V_edge, config) is read from $COEVO_STATE_DIR, written by
`challenger.freeze_solver`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# OpenEvolve imports this file in fresh worker processes: make the repository importable.
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from coevolve_bearing.metrics import margin_score  # noqa: E402
from coevolve_bearing.noise import inject_noise  # noqa: E402
import noise_lib  # noqa: E402
from gen_archive import features_from_waveform  # noqa: E402
from oe_glm_patch import disable_glm_thinking  # noqa: E402

disable_glm_thinking()

_STATE: dict = {}

# contract knobs
_GEN_RUNTIME_S = 8.0        # per-call wall-clock budget
_WORST_SCORE = -10.0


def _load_state() -> dict:
    if _STATE:
        return _STATE
    sd = Path(os.environ["COEVO_STATE_DIR"])
    with open(sd / "meta.json") as f:
        meta = json.load(f)
    npz = np.load(sd / "vedge.npz")
    obj = torch.load(sd / "solver.pt", weights_only=False, map_location="cpu")
    # Evolved Solver classes live inside build_solver, so the model is rebuilt from its source.
    from evaluator_solver import load_builder_from_source
    model = load_builder_from_source(obj["program_src"])(int(obj["n_classes"]), int(obj["length"]))
    model.load_state_dict(obj["state_dict"])
    model.to("cpu")
    model.eval()
    vedge_X, vedge_y = npz["X"], npz["y"]
    _STATE.update(meta=meta, vedge_X=vedge_X, vedge_y=vedge_y,
                  vedge_n_saved=int(len(vedge_X)), model=model, cfg=meta["cfg"])
    return _STATE


# --------------------------- sandboxed generator loading ---------------------------

def _strip_fence(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return text
    lines = s.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1])
    return text


def load_generator_from_source(src: str):
    """Execute a Challenger program and return its `generate_noise` callable.

    numpy and noise_lib are injected into the namespace.  Crashes and reward hacking are handled
    by the contract checks and the SNR lock, not by import restrictions.
    """
    src = _strip_fence(src)
    g: dict = {"__name__": "evolved_generator", "np": np, "numpy": np, "noise_lib": noise_lib}
    exec(compile(src, "<evolved_generator>", "exec"), g)
    if "build_generator" in g and callable(g["build_generator"]):
        gen = g["build_generator"]()
    elif "generate_noise" in g and callable(g["generate_noise"]):
        gen = g["generate_noise"]
    else:
        raise ValueError("program exposes neither build_generator() nor generate_noise()")
    if not callable(gen):
        raise ValueError("generate_noise is not callable")
    return gen


def _load_generator(program_path: str):
    return load_generator_from_source(Path(program_path).read_text())


def _check_contract(gen, fs: int, win: int) -> None:
    """Raise if the generator violates the hard contract."""
    t0 = time.time()
    out = gen(np.random.default_rng(12345), float(fs), int(win))
    dt = time.time() - t0
    out = np.asarray(out)
    if dt > _GEN_RUNTIME_S:
        raise ValueError(f"runtime {dt:.1f}s > {_GEN_RUNTIME_S}s")
    if out.ndim != 1 or out.shape[0] != win:
        raise ValueError(f"bad shape {out.shape}, expected ({win},)")
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite output")
    # determinism given rng
    again = np.asarray(gen(np.random.default_rng(12345), float(fs), int(win)))
    if not np.allclose(out, again, rtol=1e-5, atol=1e-6):
        raise ValueError("non-deterministic given rng")


def _generate_batch(gen, rng, fs: int, win: int, b: int) -> np.ndarray:
    noise = np.empty((b, win), dtype=np.float64)
    for i in range(b):
        v = np.asarray(gen(rng, float(fs), int(win)), dtype=np.float64)
        if v.ndim != 1 or v.shape[0] != win or not np.all(np.isfinite(v)):
            raise ValueError(f"row {i} violated contract")
        noise[i] = v
    return noise


def _harvest(record: dict) -> None:
    try:
        with open(Path(os.environ["COEVO_STATE_DIR"]) / "harvest.jsonl", "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        pass


def _realized_snr_db(X: np.ndarray, noisy: np.ndarray) -> float:
    n = (np.asarray(noisy, dtype=np.float64) - np.asarray(X, dtype=np.float64))
    sp = float(np.mean(np.asarray(X, dtype=np.float64) ** 2))
    npow = float(np.mean(n ** 2)) + 1e-12
    return float(10.0 * np.log10(sp / npow + 1e-12))


def evaluate(program_path: str) -> dict:
    st = _load_state()
    cfg = st["cfg"]
    fs = int(cfg["data"]["sample_rate_hz"])
    win = int(cfg["data"]["window_len"])
    attack_snr = float(cfg["noise"]["attack_snr_db"])
    src = Path(program_path).read_text()

    # 1) load + contract-check (any failure -> worst score, logged, no crash)
    try:
        gen = _load_generator(program_path)
        _check_contract(gen, fs, win)
    except Exception as e:
        _harvest({"program_src": src, "reward": _WORST_SCORE, "margin": 0.0,
                  "feats": {}, "valid": 0, "error": str(e)[:160]})
        return {"combined_score": _WORST_SCORE, "valid": 0.0, "error": str(e)[:160]}

    # 2) generate -> SNR-locked additive injection -> score vs frozen solver
    try:
        Xe, ye = st["vedge_X"], st["vedge_y"]
        n_saved = int(st["vedge_n_saved"])
        if len(Xe) != n_saved or len(ye) != n_saved:
            raise ValueError("incomplete V_edge")
        _, class_counts = np.unique(ye, return_counts=True)
        if n_saved == 0 or len(class_counts) == 0 or not np.all(class_counts == class_counts[0]):
            raise ValueError("unbalanced V_edge")
        rng = np.random.default_rng(int(st["meta"].get("seed", 11)) + 777)
        noise = _generate_batch(gen, rng, fs, win, n_saved)
        noisy = inject_noise(Xe, noise, attack_snr)
        margin = margin_score(st["model"], noisy, ye, cfg)
        feats = features_from_waveform(noise, fs, win)
        realized = _realized_snr_db(Xe, noisy)
    except Exception as e:
        _harvest({"program_src": src, "reward": _WORST_SCORE, "margin": 0.0,
                  "feats": {}, "valid": 0, "error": str(e)[:160]})
        return {"combined_score": _WORST_SCORE, "valid": 0.0, "error": str(e)[:160]}

    reward = -float(margin)
    _harvest({"program_src": src, "reward": float(reward), "margin": float(margin),
              "feats": feats, "valid": 1, "realized_snr_db": float(realized),
              "n_scored": n_saved})
    return {
        "combined_score": float(reward),
        "margin": float(margin),
        "centroid_hz": float(feats["centroid_hz"]),
        "spectral_entropy": float(feats["spectral_entropy"]),
        "spectral_type": str(feats["spectral_type"]),
        "dominant_band": str(feats["dominant_band"]),
        "crest_factor": float(feats["crest_factor"]),
        "heavy_burst_proxy": float(feats["heavy_burst_proxy"]),
        "realized_snr_db": float(realized),
        "valid": 1.0,
    }
