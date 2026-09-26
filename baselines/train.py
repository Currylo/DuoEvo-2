"""Shared trainer for the static baselines.

The published optimizer schedule (SGD 0.01, momentum 0.9, cosine annealing, no gradient clipping,
batch 64) with two changes that make architectures comparable:

  * the loss is class-weighted cross-entropy for every architecture, with no auxiliary terms;
  * the batch order is a function of (dataset, seed, epoch) only, drawn from its own generator,
    so every architecture sees the same samples in the same order under one seed.

The checkpoint is selected on the curriculum's selection fixture, as in the co-evolution arms.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn as nn

from coevolve_bearing.metrics import balanced_accuracy
from coevolve_bearing.train import predict


def class_weights(y, n_classes: int, device=None) -> torch.Tensor:
    """Inverse-frequency weights from the training labels, normalized to mean 1."""
    y = np.asarray(y).ravel()
    counts = np.bincount(y, minlength=int(n_classes)).astype(np.float64)
    present = counts > 0
    w = np.ones(int(n_classes), dtype=np.float64)
    w[present] = counts[present].sum() / (present.sum() * counts[present])
    w[present] /= w[present].mean()
    t = torch.as_tensor(w, dtype=torch.float32)
    return t.to(device) if device is not None else t


def epoch_permutation(n: int, dataset: str, seed: int, epoch: int) -> torch.Tensor:
    """Batch order for one epoch, a pure function of (dataset, seed, epoch)."""
    key = int(hashlib.sha256(str(dataset).encode("utf-8")).hexdigest()[:8], 16)
    g = torch.Generator()
    g.manual_seed(key * 1_000_003 + int(seed) * 10_007 + int(epoch))
    return torch.randperm(int(n), generator=g)


def train_unified(build_fn, epochs, seed, cur, cfg, n_classes, *, dataset=None,
                  init_seed=None, device=None, log=None, loss_fn=None, epoch_callback=None):
    """Train one architecture -> (best_state, best selection BA, per-epoch history).

    `build_fn(n_classes, length) -> nn.Module` returns raw logits.  `loss_fn(model, x, y, epoch,
    criterion)` replaces the loss for training-method baselines; `epoch_callback(epoch,
    best_state, best, history)` observes the selected checkpoint after every epoch.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = dataset or cfg["data"]["dataset"]
    win = int(cfg["data"]["window_len"])

    torch.manual_seed(int(seed if init_seed is None else init_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed if init_seed is None else init_seed))
    model = build_fn(int(n_classes), win).to(device)

    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(epochs)), eta_min=1e-8)
    xb = torch.from_numpy(np.asarray(cur["train_X"], dtype=np.float32)).unsqueeze(1)
    yb = torch.from_numpy(np.asarray(cur["train_y"], dtype=np.int64))
    crit = nn.CrossEntropyLoss(weight=class_weights(cur["train_y"], n_classes, device))
    bs = int(cfg["train"]["batch_size"])
    n = len(xb)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best, history = -1.0, []
    for ep in range(int(epochs)):
        model.train()
        perm = epoch_permutation(n, dataset, seed, ep)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            if loss_fn is None:
                out = model(xb[idx].to(device))
                if out.dim() != 2 or out.shape[1] != int(n_classes):
                    raise RuntimeError(f"adapter must return (B, {n_classes}) logits, "
                                       f"got {tuple(out.shape)}")
                loss = crit(out, yb[idx].to(device))
            else:
                loss = loss_fn(model, xb[idx].to(device), yb[idx].to(device), ep, crit)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {ep}")
            loss.backward()
            opt.step()
        sched.step()
        b = float(balanced_accuracy(cur["sel_y"], predict(model, cur["sel_X"], cfg)))
        history.append(b)
        if b > best:
            best = b
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if log:
            log(f"  epoch {ep + 1}/{epochs} selection BA={b:.4f}")
        if epoch_callback:
            epoch_callback(ep, best_state, best, history)
    model.load_state_dict(best_state)
    return best_state, best, history
