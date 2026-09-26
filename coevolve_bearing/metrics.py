from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                      n_classes: int | None = None) -> float:
    """Mean per-class recall."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    classes = list(range(n_classes)) if n_classes is not None else sorted(set(y_true.tolist()))
    vals = []
    for cls in classes:
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        vals.append(float((y_pred[mask] == cls).mean()))
    return float(np.mean(vals)) if vals else 0.0


def as_probs(out: torch.Tensor, tol: float = 1e-4) -> torch.Tensor:
    """Map a network output to probabilities, whatever its convention.

    Evolved Solvers may return probabilities (the ClassBD anchor ends in a softmax),
    log-probabilities or logits.  Applying softmax to probabilities would compress the margin,
    so the convention is detected from the rows.
    """
    if out.min() >= -tol and (out.sum(dim=1) - 1.0).abs().max() < 1e-2:
        return out
    if out.max() <= tol and (out.exp().sum(dim=1) - 1.0).abs().max() < 1e-2:
        return out.exp()
    return torch.softmax(out, dim=1)


@torch.no_grad()
def margin_score(model: nn.Module, X: np.ndarray, y: np.ndarray, cfg: dict) -> float:
    """Mean classification margin E[p(y) - max_{k != y} p(k)] in [-1, 1]; high means easy."""
    device = next(model.parameters()).device
    model.eval()
    bs = int(cfg["train"]["eval_batch_size"])
    y = np.asarray(y).astype(int)
    margins = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(np.asarray(X[i: i + bs], dtype=np.float32), device=device).unsqueeze(1)
        probs = as_probs(model(xb)).cpu().numpy()
        yb = y[i: i + bs]
        ptrue = probs[np.arange(len(yb)), yb]
        tmp = probs.copy()
        tmp[np.arange(len(yb)), yb] = -1.0
        margins.append(ptrue - tmp.max(axis=1))
    return float(np.concatenate(margins).mean()) if margins else 0.0
