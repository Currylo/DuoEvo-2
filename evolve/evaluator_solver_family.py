"""Solver screening with per-noise-family feedback for the structure search.

Wraps evaluator_solver.py: after screening training, the candidate is also scored on five fixed
canonical noise families (families.py).  The breakdown is returned as an OpenEvolve artifact, which
the next rewrite prompt shows to the LLM ("raise the weakest family without giving up the
others").  The families are fixed, so "weakest" is comparable across rounds and arms.

Environment (set by duoevo.py):
  COEVO_SOLVER_STATE_DIR   curriculum directory
  COEVO_SOLVER_EVAL_EPOCHS screening epochs
  COEVO_FAMILY_N           probe windows per family            (default 128)
  COEVO_FAMILY_SNRS        comma-separated probe SNRs in dB     (default: the curriculum SNRs)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from openevolve.evaluation_result import EvaluationResult      # noqa: E402
from coevolve_bearing.metrics import balanced_accuracy          # noqa: E402
from coevolve_bearing.noise import apply_program                # noqa: E402
from coevolve_bearing.train import predict                      # noqa: E402
from families import DIAG_FAMILIES, diag_program                # noqa: E402
from evaluator_solver import (                                  # noqa: E402
    WORST_SCORE, _build_and_check, _descriptor, _harvest, _load_state, _train,
    load_builder_from_source, seed_eval,
)


def _probe_snrs(cfg) -> list[float]:
    raw = os.environ.get("COEVO_FAMILY_SNRS")
    if raw:
        return [float(s) for s in raw.split(",") if s.strip()]
    return [float(s) for s in cfg["curriculum"]["aug_snrs_db"]]


def family_scores(model, X, y, cfg, fs, seed) -> dict:
    """Balanced accuracy per canonical noise family (mean over the probe SNRs)."""
    n = int(os.environ.get("COEVO_FAMILY_N", 128))
    if len(X) > n:                      # fixed subsample
        idx = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
        X, y = X[idx], y[idx]
    out = {}
    for i, fam in enumerate(DIAG_FAMILIES):
        vals = []
        for j, snr in enumerate(_probe_snrs(cfg)):
            rng = np.random.default_rng(seed + 7000 + 100 * i + j)
            noisy, _ = apply_program(diag_program(fam, float(snr), rng), X, fs, seed + 9000 + 100 * i + j)
            vals.append(float(balanced_accuracy(y, predict(model, noisy, cfg))))
        out[fam] = float(np.mean(vals)) if vals else 0.0
    return out


def evaluate(program_path: str) -> dict:
    st = _load_state()
    cfg = st["cfg"]
    n_classes = int(st["meta"]["n_classes"])
    length = int(st["meta"]["length"])
    src = Path(program_path).read_text(encoding="utf-8")
    seed = int(st["meta"].get("seed", 0))
    seed_eval(seed, src)

    # 1) build + contract check
    try:
        model = _build_and_check(load_builder_from_source(src), n_classes, length)
    except Exception as e:  # noqa: BLE001
        _harvest({"program_src": src, "reward": WORST_SCORE, "valid": 0, "error": str(e)[:160]})
        return {"combined_score": WORST_SCORE, "valid": 0.0, "error": str(e)[:160]}

    # 2) train on the curriculum -> overall OOD score + per-family breakdown
    try:
        feats = _descriptor(model, src)
        model = _train(model, st)
        ood = float(balanced_accuracy(st["oodv_y"], predict(model, st["oodv_X"], cfg)))
        fams = family_scores(model, st["dev_X"], st["dev_y"], cfg,
                             int(cfg["data"]["sample_rate_hz"]), seed)
    except Exception as e:  # noqa: BLE001
        _harvest({"program_src": src, "reward": WORST_SCORE, "valid": 0,
                  "error": f"train/eval: {str(e)[:140]}"})
        return {"combined_score": WORST_SCORE, "valid": 0.0, "error": str(e)[:160]}

    weakest = min(fams, key=fams.get) if fams else ""
    _harvest({"program_src": src, "reward": ood, "valid": 1, "ood_val": ood,
              "feats": feats, "family_scores": fams, "weakest_family": weakest})

    metrics = {"combined_score": ood, "valid": 1.0, "ood_val": ood,
               "params": feats["params"], "n_conv": feats["n_conv"],
               "weakest_family": weakest}
    metrics.update({f"fam_{k}": round(v, 4) for k, v in fams.items()})
    return EvaluationResult(metrics=metrics, artifacts={"robustness_by_noise_family": _report(fams, ood)})


def _report(fams: dict, ood: float) -> str:
    """The text the LLM actually reads in the next prompt's {artifacts} section."""
    weakest = min(fams, key=fams.get)
    width = max(len(k) for k in fams)
    lines = [f"  {k.ljust(width)}  {v:.4f}" + ("   <<< WEAKEST" if k == weakest else "")
             for k, v in sorted(fams.items(), key=lambda kv: kv[1])]
    spread = max(fams.values()) - min(fams.values())
    return ("Balanced accuracy of THIS program per canonical noise family "
            "(fixed probe, higher is better):\n" + "\n".join(lines) +
            f"\n\noverall OOD {ood:.4f} | spread best-worst {spread:.4f}\n"
            f"The next rewrite should raise '{weakest}' without giving up the others.")
