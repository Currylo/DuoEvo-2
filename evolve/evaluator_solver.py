"""Screening evaluator for Solver programs.

A candidate `build_solver(n_classes, length) -> nn.Module` is built, checked, trained for
$COEVO_SOLVER_EVAL_EPOCHS epochs on the round's curriculum and scored by balanced accuracy on the
screening set.  The contract keeps free network synthesis safe:

  1. instantiable with correct I/O   (B, 1, L) -> (B, n_classes) finite class probabilities
  2. parameter cap                   no win by capacity
  3. checked execution               any exception scores worst and is logged
  4. descriptors                     #params, #conv layers, for the search archive

The round's curriculum is read from $COEVO_SOLVER_STATE_DIR (written by `duoevo.freeze_curriculum`).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from coevolve_bearing.metrics import balanced_accuracy  # noqa: E402
from coevolve_bearing.train import predict  # noqa: E402
import solver_lib  # noqa: E402

_STATE: dict = {}

WORST_SCORE = -1.0
PARAM_CAP = 8_000_000
BUILD_RUNTIME_S = 30.0


def _state_dir() -> Path:
    return Path(os.environ["COEVO_SOLVER_STATE_DIR"])


def _load_state() -> dict:
    if _STATE:
        return _STATE
    sd = _state_dir()
    with open(sd / "meta.json") as f:
        meta = json.load(f)
    data = np.load(sd / "data.npz")
    _STATE.update(meta=meta, cfg=meta["cfg"],
                  train_X=data["train_X"], train_y=data["train_y"],
                  dev_X=data["dev_X"], dev_y=data["dev_y"],
                  oodv_X=data["oodv_X"], oodv_y=data["oodv_y"])
    return _STATE


def seed_eval(meta_seed: int, src: str) -> int:
    """Seed weight init and shuffles from (seed XOR sha256(program source)).

    The same program is scored identically wherever it is trained, while different programs get
    distinct fixed seeds, so a screening win cannot come from a lucky initialization.
    """
    h = int(hashlib.sha256((src or "").encode("utf-8")).hexdigest()[:8], 16)
    seed = (int(meta_seed) ^ h) & 0x7FFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN convolution is nondeterministic by default.  (use_deterministic_algorithms(True) would
    # raise on AdaptiveAvgPool1d backward; cudnn.deterministic covers the convolutions.)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def _strip_fence(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return text
    lines = s.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1])
    return text


def load_builder_from_source(src: str):
    """Execute a Solver program and return its `build_solver` callable."""
    src = _strip_fence(src)
    g: dict = {"__name__": "evolved_solver", "torch": torch, "nn": nn,
               "F": torch.nn.functional, "solver_lib": solver_lib}
    exec(compile(src, "<evolved_solver>", "exec"), g)
    if "build_solver" in g and callable(g["build_solver"]):
        return g["build_solver"]
    if "get_solver" in g and callable(g["get_solver"]):
        return g["get_solver"]
    raise ValueError("program exposes neither build_solver(n_classes,length) nor get_solver(...)")


def _build_and_check(builder, n_classes: int, length: int) -> nn.Module:
    t0 = time.time()
    model = builder(int(n_classes), int(length))
    if not isinstance(model, nn.Module):
        raise ValueError("build_solver did not return an nn.Module")
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if n_params > PARAM_CAP:
        raise ValueError(f"param count {n_params} > cap {PARAM_CAP}")
    if n_params == 0:
        raise ValueError("model has no trainable parameters")
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(4, 1, int(length))
        out = model(dummy)
    if out.shape != (4, int(n_classes)):
        raise ValueError(f"forward output shape {tuple(out.shape)} != (4,{n_classes})")
    if not torch.isfinite(out).all():
        raise ValueError("non-finite forward output")
    if out.min().item() < -1e-5:
        raise ValueError("forward output must be class probabilities (negative value found)")
    row_sum_error = (out.sum(dim=1) - 1.0).abs().max().item()
    if row_sum_error > 1e-3:
        raise ValueError(
            "forward output must be class probabilities "
            f"(max row-sum error {row_sum_error:.3g})"
        )
    if time.time() - t0 > BUILD_RUNTIME_S:
        raise ValueError("build/forward exceeded runtime budget")
    return model


def _train(model: nn.Module, st: dict) -> nn.Module:
    """Screening training under the published ClassBD recipe (see `duoevo.train_official`).

    Keeps the epoch with the best clean-dev balanced accuracy.  A structure whose epoch exceeds
    $COEVO_EPOCH_TIME_CAP_S seconds is rejected: some convolution stacks have no fast
    deterministic kernel and would stall every later training.
    """
    cfg = st["cfg"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = int(os.environ["COEVO_SOLVER_EVAL_EPOCHS"])
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=1e-8)
    crit = nn.CrossEntropyLoss()
    xb = torch.from_numpy(np.asarray(st["train_X"], dtype=np.float32)).unsqueeze(1)
    yb = torch.from_numpy(np.asarray(st["train_y"], dtype=np.int64))
    bs = int(cfg["train"]["batch_size"])
    n = len(xb)
    best_state, best = None, -1.0
    cap_s = os.environ.get("COEVO_EPOCH_TIME_CAP_S")
    cap_s = float(cap_s) if cap_s else None
    for _ in range(epochs):
        t_ep = time.time()
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            bx, by = xb[idx].to(device), yb[idx].to(device)
            opt.zero_grad(set_to_none=True)
            out = model(bx)
            loss = crit(out, by)
            k, g = getattr(model, "aux_k", None), getattr(model, "aux_g", None)
            if k is not None and g is not None:
                ls = torch.tensor([-0.5, -0.5, -0.5], device=out.device)
                loss = (torch.stack([loss, k.to(out.device), g.to(out.device)])
                        / (3 * ls.exp()) + ls / 3).sum()
            loss.backward()
            opt.step()
        if cap_s is not None:
            dt = time.time() - t_ep
            if dt > cap_s:
                raise ValueError(f"epoch wall-clock {dt:.1f}s > cap {cap_s:.0f}s")
        sched.step()
        b = balanced_accuracy(st["dev_y"], predict(model, st["dev_X"], cfg))
        if b > best:
            best = b
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _descriptor(model: nn.Module, src: str) -> dict:
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv1d))
    n_quad = sum(1 for m in model.modules() if isinstance(m, solver_lib.ConvQuadraticOperation))
    return {"params": n_params, "n_conv": int(n_conv), "n_quad": int(n_quad),
            "uses_fft": int("fft" in src)}


def _harvest(record: dict) -> None:
    try:
        with open(_state_dir() / "harvest.jsonl", "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        pass


def evaluate(program_path: str) -> dict:
    st = _load_state()
    n_classes = int(st["meta"]["n_classes"])
    length = int(st["meta"]["length"])
    src = Path(program_path).read_text()
    seed_eval(int(st["meta"].get("seed", 0)), src)

    # 1) build + contract-check (any failure -> worst score, logged, never crash)
    try:
        builder = load_builder_from_source(src)
        model = _build_and_check(builder, n_classes, length)
    except Exception as e:  # noqa: BLE001
        _harvest({"program_src": src, "reward": WORST_SCORE, "valid": 0, "error": str(e)[:160]})
        return {"combined_score": WORST_SCORE, "valid": 0.0, "error": str(e)[:160]}

    # 2) train on the current curriculum -> score on the OOD-val set
    try:
        feats = _descriptor(model, src)
        model = _train(model, st)
        ood = float(balanced_accuracy(st["oodv_y"], predict(model, st["oodv_X"], st["cfg"])))
    except Exception as e:  # noqa: BLE001
        _harvest({"program_src": src, "reward": WORST_SCORE, "valid": 0,
                  "error": f"train/eval: {str(e)[:140]}"})
        return {"combined_score": WORST_SCORE, "valid": 0.0, "error": str(e)[:160]}

    _harvest({"program_src": src, "reward": ood, "valid": 1, "ood_val": ood, "feats": feats})
    return {"combined_score": ood, "valid": 1.0, "ood_val": ood,
            "params": feats["params"], "n_conv": feats["n_conv"]}
