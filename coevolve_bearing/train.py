from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def predict(model: nn.Module, X: np.ndarray, cfg: dict) -> np.ndarray:
    """Arg-max class of every window, in batches of `train.eval_batch_size`."""
    device = next(model.parameters()).device
    model.eval()
    x = torch.from_numpy(np.asarray(X, dtype=np.float32)).unsqueeze(1)
    loader = DataLoader(TensorDataset(x), batch_size=int(cfg["train"]["eval_batch_size"]),
                        shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.int64)
