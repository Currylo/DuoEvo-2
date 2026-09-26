"""Shared baseline trainer: class weights, architecture-independent batch order, selection."""
import os
import subprocess
import sys

import numpy as np
import torch

from baselines.train import class_weights, epoch_permutation, train_unified
from conftest import ROOT


def test_class_weights_have_mean_one():
    assert np.allclose(class_weights(np.array([0, 1, 2] * 10), 3).numpy(), 1.0)
    w = class_weights(np.array([0] * 90 + [1] * 10), 2).numpy()
    assert w[1] > w[0] and np.isclose(w.mean(), 1.0)
    w = class_weights(np.array([0] * 20 + [2] * 20), 3).numpy()      # class 1 absent
    assert w[1] == 1.0 and np.isclose(w[[0, 2]].mean(), 1.0)


def test_batch_order_depends_only_on_dataset_seed_epoch():
    a = epoch_permutation(100, "pu", 11, 0)
    assert torch.equal(a, epoch_permutation(100, "pu", 11, 0))
    for other in (("pu", 11, 1), ("pu", 23, 0), ("cwru", 11, 0)):
        assert not torch.equal(a, epoch_permutation(100, *other))
    torch.manual_seed(0)
    torch.randn(10_000)                                   # global RNG use must not matter
    assert torch.equal(a, epoch_permutation(100, "pu", 11, 0))
    assert sorted(a.tolist()) == list(range(100))


def test_batch_order_is_stable_across_processes():
    code = ("import sys; sys.path.insert(0, %r); from baselines.train import epoch_permutation;"
            "print(epoch_permutation(50, 'jnu', 37, 2).tolist())" % str(ROOT))
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": h}).stdout for h in ("1", "2")}
    assert len(outs) == 1


def test_train_unified_selects_on_the_selection_fixture(tiny_splits, tiny_cfg):
    cur = {"train_X": tiny_splits.train_X, "train_y": tiny_splits.train_y,
           "sel_X": tiny_splits.dev_X, "sel_y": tiny_splits.dev_y}

    def build(n_classes, length):
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(length, n_classes))

    state, best, history = train_unified(build, 3, 11, cur, tiny_cfg, 2, dataset="toy", device="cpu")
    assert len(history) == 3 and best == max(history)
    assert set(state) == {"1.weight", "1.bias"}
