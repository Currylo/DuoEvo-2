"""Static-baseline table: published backbones on the same curriculum, recipe and evaluation.

  * curriculum  C(0), the random-DSL curriculum every co-evolution arm starts from, byte-identical
                under one (dataset, seed);
  * recipe      baselines/train.py (class-weighted CE, shared batch order, noisy-dev selection);
  * budget      the co-evolution deployment budget, boot_epochs + rounds * continue_epochs;
  * evaluation  the promotion-gate families on dev and T2 on test, as for the co-evolution arms.

Results are appended to <out>/results.json; finished (model, dataset, seed) entries are skipped.

    python -m baselines.run_table --datasets pu,cwru,jnu --seeds 11,23,37
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import duoevo as D
from baselines import adapters as A
from baselines.train import train_unified
from coevolve_bearing.data import load_splits
from coevolve_bearing.metrics import balanced_accuracy
from coevolve_bearing.train import predict
from evaluator_solver_family import family_scores
from gen_archive import GenMapElites

ROOT = Path(__file__).resolve().parents[1]


def log(msg, fh=None):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def deployment_epochs(cfg) -> int:
    exp = cfg["exp"]
    return int(exp["boot_epochs"]) + int(exp["rounds"]) * int(exp["continue_epochs"])


def load_everything(dataset, seed, fh=None, cur_root="runs/baselines/curricula"):
    """(cfg, splits, C(0), n_classes) for one dataset and seed."""
    cfg = D.load_config(ROOT / "configs" / f"{dataset}.yaml", seed)
    splits = load_splits(cfg)
    cfg["data"]["n_classes"] = int(splits.meta["n_classes"])
    cur_dir = Path(cur_root) / f"{dataset}_s{seed}"
    cur_dir.mkdir(parents=True, exist_ok=True)
    cur0, sd0, meta0 = D.freeze_curriculum(0, cur_dir, splits, GenMapElites(), cfg, seed,
                                           mix_random_frac=1.0,
                                           archive_cap=int(cfg["exp"]["archive_materialize_cap"]))
    D.set_search_env(sd0, cfg)
    log(f"{dataset} s{seed}: {len(cur0['train_X'])} training rows, "
        f"curriculum sha {meta0['data_sha256'][:16]}", fh)
    return cfg, splits, cur0, int(splits.meta["n_classes"])


def evaluate(model, splits, cfg, seed) -> dict:
    """Gate families on dev (as in the promotion gate) and T2 on test."""
    fs = int(cfg["data"]["sample_rate_hz"])
    prev_snrs, prev_n = os.environ.get("COEVO_FAMILY_SNRS"), os.environ.get("COEVO_FAMILY_N")
    os.environ["COEVO_FAMILY_SNRS"] = ",".join(str(s) for s in D.GATE_SNRS)
    os.environ["COEVO_FAMILY_N"] = str(int(cfg["exp"]["gate_probe_n"]))
    try:
        fams = family_scores(model, splits.dev_X, splits.dev_y, cfg, fs,
                             int(seed) + D.GATE_SEED_OFFSET)
    finally:
        for k, v in (("COEVO_FAMILY_SNRS", prev_snrs), ("COEVO_FAMILY_N", prev_n)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    ranked = sorted(fams.values())
    t2_mean, t2_worst, t2_cells = D.t2_score(model, splits, cfg, seed)
    return {
        "clean": float(balanced_accuracy(splits.dev_y, predict(model, splits.dev_X, cfg))),
        "gate_mean": float(np.mean(list(fams.values()))),
        "gate_worst": float(np.mean(ranked[:D.GATE_WORST_K])),
        "gate_families": {k: round(v, 4) for k, v in fams.items()},
        "t2_mean": float(t2_mean), "t2_worst_cell": float(t2_worst),
        "t2_cells": {k: round(float(v), 4) for k, v in t2_cells.items()},
    }


def main():
    ap = argparse.ArgumentParser(description="static-baseline table")
    ap.add_argument("--out", default="runs/baselines")
    ap.add_argument("--datasets", default="pu,cwru,jnu")
    ap.add_argument("--seeds", default="11,23,37")
    ap.add_argument("--models", default=None,
                    help=f"comma-separated subset of {sorted(A.REGISTRY)}")
    ap.add_argument("--epochs", type=int, default=None,
                    help="training epochs (default: the co-evolution deployment budget)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / "table.log", "a", encoding="utf-8")
    results_path = out / "results.json"
    rows = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    done = {(r["model"], r["dataset"], r["seed"]) for r in rows if "error" not in r}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [m.strip() for m in args.models.split(",")] if args.models else list(A.REGISTRY)
    unknown = [m for m in models if m not in A.REGISTRY]
    if unknown:
        raise SystemExit(f"unknown baseline {unknown}; available: {sorted(A.REGISTRY)}")

    for dataset in args.datasets.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            todo = [m for m in models if (m, dataset, seed) not in done]
            if not todo:
                continue
            cfg, splits, cur0, n_classes = load_everything(dataset, seed, fh, out / "curricula")
            epochs = args.epochs or deployment_epochs(cfg)
            for model_name in todo:
                t0 = time.time()
                try:
                    state, best_sel, _ = train_unified(
                        lambda n, length, _m=model_name: A.build(_m, n, length),
                        epochs=epochs, seed=seed, cur=cur0, cfg=cfg, n_classes=n_classes,
                        dataset=dataset, device=device)
                    net = A.build(model_name, n_classes, int(cfg["data"]["window_len"])).to(device)
                    net.load_state_dict(state)
                    net.eval()
                    row = {"model": model_name, "dataset": dataset, "seed": seed,
                           "n_classes": n_classes, "epochs": epochs,
                           "params": int(sum(p.numel() for p in net.parameters())),
                           "best_selection_ba": round(float(best_sel), 4),
                           "train_s": round(time.time() - t0, 1),
                           **evaluate(net, splits, cfg, seed)}
                    del net
                    log(f"  {model_name:16s} {dataset:5s} s{seed}  T2={row['t2_mean']:.4f} "
                        f"(worst {row['t2_worst_cell']:.4f})  gate={row['gate_mean']:.4f}", fh)
                except Exception as e:                                    # noqa: BLE001
                    log(f"  {model_name} {dataset} s{seed} failed: {type(e).__name__}: {e}", fh)
                    log(traceback.format_exc(), fh)
                    row = {"model": model_name, "dataset": dataset, "seed": seed,
                           "error": f"{type(e).__name__}: {e}"}
                rows.append(row)
                results_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
                if device == "cuda":
                    torch.cuda.empty_cache()
    fh.close()


if __name__ == "__main__":
    main()
