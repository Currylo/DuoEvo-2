"""ClassBD trained with MART-LOAT adversarial training on the static C(0) curriculum.

Same curriculum, backbone adapter, trainer and budget as the static-baseline table; only the
objective changes.  The attack SNR and MART weight come from each dataset config (`mart_loat`),
selected on seed-11 validation with the same rule for every dataset.  T2 is read only with
--evaluate, after training.

    python -m baselines.run_mart_loat --datasets pu,cwru,jnu --seeds 11,23,37 --evaluate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from baselines.adapters import build_classbd
from baselines.mart_loat import MartLoatObjective, loat_phase
from baselines.run_table import deployment_epochs, evaluate, load_everything
from baselines.train import train_unified
from coevolve_bearing.metrics import balanced_accuracy
from coevolve_bearing.train import predict

MODEL_NAME = "ClassBD+MART-LOAT"


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def curriculum_hash(cur):
    digest = hashlib.sha256()
    for key in ("train_X", "train_y", "sel_X", "sel_y"):
        value = np.ascontiguousarray(cur[key])
        digest.update(f"{key}:{value.dtype}:{value.shape}".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def run_one(args, dataset, seed):
    out = Path(args.out)
    run_dir = out / f"{dataset}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path, checkpoint = run_dir / "result.json", run_dir / "best_state.pt"
    if result_path.exists() and not args.evaluate:
        print(f"{dataset} s{seed}: completed, skipping", flush=True)
        return json.loads(result_path.read_text())

    with (run_dir / "train.log").open("a", buffering=1) as log_file:
        def log(message):
            line = f"[{time.strftime('%F %T')}] {message}"
            print(line, flush=True)
            log_file.write(line + "\n")

        random.seed(seed)
        np.random.seed(seed)
        cfg, splits, cur, n_classes = load_everything(dataset, seed, log_file, out / "curricula")
        epochs = args.epochs or deployment_epochs(cfg)
        hp = cfg["mart_loat"]
        objective = MartLoatObjective(epochs, steps=int(hp["steps"]),
                                      snr_db=float(hp["attack_snr_db"]), beta=float(hp["beta"]),
                                      theta=float(hp["theta"]), gamma=float(hp["gamma"]), seed=seed)
        digest = curriculum_hash(cur)

        if not result_path.exists():
            start = time.monotonic()

            def report(epoch, state, best, history):
                torch.save(state, checkpoint)
                write_json(run_dir / "progress.json", {
                    "epoch": epoch + 1, "epochs": epochs, "phase": loat_phase(epoch, epochs),
                    "best_selection_ba": best, "elapsed_s": time.monotonic() - start})

            log(f"{MODEL_NAME} {dataset} s{seed}: {epochs} epochs, L2 PGD-{objective.steps} at "
                f"{objective.snr_db} dB, beta={objective.beta}")
            state, best, history = train_unified(
                build_classbd, epochs, seed, cur, cfg, n_classes, dataset=dataset,
                device=args.device, log=log, loss_fn=objective, epoch_callback=report)
            result = {"model": MODEL_NAME, "dataset": dataset, "seed": seed, "n_classes": n_classes,
                      "epochs": epochs, "objective": asdict(objective),
                      "best_selection_ba": best, "best_epoch": int(np.argmax(history)) + 1,
                      "history": history, "train_s": time.monotonic() - start,
                      "curriculum_sha256": digest, "evaluated": False}
        else:
            result = json.loads(result_path.read_text())
            if result["curriculum_sha256"] != digest:
                raise ValueError("recreated curriculum differs from the completed run")

        net = build_classbd(n_classes, int(cfg["data"]["window_len"])).to(args.device)
        net.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
        net.eval()
        reloaded = float(balanced_accuracy(cur["sel_y"], predict(net, cur["sel_X"], cfg)))
        if reloaded != result["best_selection_ba"]:
            raise RuntimeError("checkpoint reload does not reproduce the best selection score")
        result["params"] = sum(p.numel() for p in net.parameters())
        if args.evaluate:
            result.update(evaluate(net, splits, cfg, seed))
            result["evaluated"] = True
        write_json(result_path, result)
        log(f"saved {result_path}; T2 evaluated={result['evaluated']}")
        return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="ClassBD + MART-LOAT baseline")
    ap.add_argument("--out", default="runs/mart_loat")
    ap.add_argument("--datasets", default="pu,cwru,jnu")
    ap.add_argument("--seeds", default="11,23,37")
    ap.add_argument("--epochs", type=int, default=None,
                    help="training epochs (default: the co-evolution deployment budget)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--evaluate", action="store_true", help="score the selected checkpoint on T2")
    args = ap.parse_args(argv)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rows = [run_one(args, d, int(s)) for d in args.datasets.split(",") for s in args.seeds.split(",")]
    write_json(Path(args.out) / "results.json", rows)


if __name__ == "__main__":
    main()
