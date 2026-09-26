"""Adapters exposing each published backbone as `build(n_classes, length) -> nn.Module`.

Contract: forward (B, 1, length) -> (B, n_classes) raw logits, finite, no internal softmax.
Adapters only (a) set the class count, (b) fix batch-size-1 shape bugs and (c) drop auxiliary
outputs or a trailing softmax; published architectures are not restructured.  Length-locked
models assert their length instead of being adapted.  `SPECS` records what was touched.

The upstream implementations are not redistributed; clone them into third_party/ (see README).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

BASELINES = Path(__file__).resolve().parents[1] / "third_party"

SPECS = {
    "ClassBD-BDWDCNN": "n_classes native; dropped softmax + (k, g) aux outputs; batch-1 eval via duplicated pair (upstream kurtosis squeeze)",
    "DRSN-CW": "n_classes native; no other change",
    "GTFENet": "class_number -> n_classes; squeeze() -> flatten(1) for batch=1",
    "QCNN": "10-way head -> n_classes; dropped trailing softmax; flatten untouched, length==2048 asserted",
    "WDCNN": "10-way head -> n_classes; dropped trailing softmax; flatten untouched, length==2048 asserted",
    "TFN-STTF": "out_channels -> n_classes; AdaptiveMaxPool1d(4) makes it length-flexible",
}


class _PathCtx:
    """Import upstream modules that assume their own directory is on `sys.path`.

    ClassBD and Qttention both ship a top-level package called `Model`; purging the shared names
    on entry and exit keeps the two repositories independent.
    """

    SHARED = ("Model", "Models")

    def __init__(self, *parts):
        self.paths = [str(BASELINES / p) for p in parts]

    def _purge(self):
        for name in list(sys.modules):
            if name in self.SHARED or any(name.startswith(s + ".") for s in self.SHARED):
                del sys.modules[name]

    def __enter__(self):
        self._purge()
        for p in reversed(self.paths):
            if p not in sys.path:
                sys.path.insert(0, p)
        return self

    def __exit__(self, *exc):
        for p in self.paths:
            if p in sys.path:
                sys.path.remove(p)
        self._purge()


# ---------------------------------------------------------------- ClassBD


class ClassBDLogits(nn.Module):
    """`BDWDCNN` without the trailing softmax and the physics auxiliary outputs."""

    def __init__(self, n_classes: int):
        super().__init__()
        with _PathCtx("ClassBD", "ClassBD/Model"):
            from Model.BDCNN import BDWDCNN
        self.net = BDWDCNN(int(n_classes))

    def forward(self, x):
        # Upstream `funcKurtosis` squeezes without a dim, so a batch of 1 breaks.  In eval mode every
        # layer is batch-independent, so evaluating a duplicated pair is exactly equivalent.
        if x.size(0) == 1 and not self.training:
            return self._logits(x.expand(2, *x.shape[1:]))[:1]
        return self._logits(x)

    def _logits(self, x):
        m = self.net
        a2, _k, _g = m.classbd(x)
        out = m.cnn(a2)
        out = m.fc1(out.reshape(x.size(0), -1))
        out = m.fc2(m.dp(m.relu1(out)))
        return out                                     # raw logits


def build_classbd(n_classes: int, length: int) -> nn.Module:
    _require_len(length, 2048, "ClassBD-BDWDCNN")
    return ClassBDLogits(n_classes)


# ---------------------------------------------------------------- DRSN


def build_drsn(n_classes: int, length: int) -> nn.Module:
    with _PathCtx("DRSN"):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "drsn_cw", BASELINES / "DRSN" / "DRSN-CW.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod.RSNet(mod.BasicBlock, [2, 2, 2, 2], num_classes=int(n_classes))


# ---------------------------------------------------------------- GTFENet


class GTFENetLogits(nn.Module):
    """`the_model` with the batch axis restored after its dimensionless `squeeze()` at batch 1."""

    def __init__(self, n_classes: int):
        super().__init__()
        with _PathCtx("GTFENet"):
            from GTFENet import the_model
        self.net = the_model(class_number=int(n_classes))
        self.n_classes = int(n_classes)

    def forward(self, x):
        out = self.net(x)
        if out.dim() == 1:                                  # batch-1 collapse
            out = out.unsqueeze(0)
        return out


def build_gtfenet(n_classes: int, length: int) -> nn.Module:
    return GTFENetLogits(n_classes)


# ---------------------------------------------------------------- QCNN / WDCNN


def _swap_head(net: nn.Module, n_classes: int) -> nn.Module:
    """Replace the hardcoded 10-way `fc2` with an `n_classes` one, keeping everything else."""
    old = net.fc2
    net.fc2 = nn.Linear(old.in_features, int(n_classes))
    return net


class PreSoftmaxLogits(nn.Module):
    """`QCNN` / `WDCNN` without the trailing softmax, so every backbone trains on logits.

    Every weight-bearing layer is the upstream object; QCNN and WDCNN share one forward body.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        m = self.net
        out = m.cnn(x)
        out = m.fc1(out.view(x.size(0), -1))
        out = m.fc2(m.dp(m.relu1(out)))
        return out                                     # raw logits


def build_qcnn(n_classes: int, length: int) -> nn.Module:
    _require_len(length, 2048, "QCNN")
    with _PathCtx("Qttention", "Qttention/Model"):
        from Model.QCNN import QCNN
    return PreSoftmaxLogits(_swap_head(QCNN(), n_classes))


def build_wdcnn(n_classes: int, length: int) -> nn.Module:
    _require_len(length, 2048, "WDCNN")
    with _PathCtx("Qttention", "Qttention/Model"):
        from Model.WDCNN import WDCNN
    return PreSoftmaxLogits(_swap_head(WDCNN(), n_classes))


# ---------------------------------------------------------------- TFN


def build_tfn(n_classes: int, length: int) -> nn.Module:
    with _PathCtx("TFN", "TFN/Models"):
        from Models.TFN import TFN_STTF
    return TFN_STTF(in_channels=1, out_channels=int(n_classes))


# ---------------------------------------------------------------- registry


def _require_len(length: int, need: int, who: str):
    if int(length) != int(need):
        raise ValueError(
            f"{who} hardcodes a flatten sized for length {need}; got {length}")


REGISTRY = {
    "ClassBD-BDWDCNN": build_classbd,
    "DRSN-CW": build_drsn,
    "GTFENet": build_gtfenet,
    "QCNN": build_qcnn,
    "WDCNN": build_wdcnn,
    "TFN-STTF": build_tfn,
}


def build(name: str, n_classes: int, length: int) -> nn.Module:
    if name not in REGISTRY:
        raise KeyError(f"unknown baseline {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](int(n_classes), int(length))


def looks_like_probabilities(y: torch.Tensor) -> bool:
    """True if `y` could be a softmax output: all non-negative and rows summing to 1."""
    s = y.sum(dim=1)
    return bool((y >= 0).all() and torch.allclose(s, torch.ones_like(s), atol=1e-4))


def check(name: str, n_classes: int = 3, length: int = 2048, device="cpu") -> dict:
    """Build + forward at batch 1 and 2, returning the facts the main table needs."""
    net = build(name, n_classes, length).to(device).eval()
    out = {}
    for b in (1, 2):
        with torch.no_grad():
            y = net(torch.randn(b, 1, length, device=device))
        if y.shape != (b, n_classes):
            raise AssertionError(f"{name}: batch={b} gave {tuple(y.shape)}, want {(b, n_classes)}")
        if not torch.isfinite(y).all():
            raise AssertionError(f"{name}: non-finite output at batch={b}")
        out[f"shape_b{b}"] = tuple(y.shape)
    with torch.no_grad():
        probe = net(torch.randn(8, 1, length, device=device))
    if looks_like_probabilities(probe):
        raise AssertionError(
            f"{name}: output looks like a softmax; the contract is raw logits")
    out["params"] = int(sum(p.numel() for p in net.parameters() if p.requires_grad))
    out["adapted"] = SPECS[name]
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="check every baseline adapter")
    ap.add_argument("--n-classes", type=int, default=3)
    ap.add_argument("--length", type=int, default=2048)
    args = ap.parse_args()
    print("%-18s %-12s %-12s %10s  %s" % ("model", "batch=1", "batch=2", "params", "adapted"))
    for name in REGISTRY:
        try:
            r = check(name, args.n_classes, args.length)
            print("%-18s %-12s %-12s %10s  %s" % (
                name, r["shape_b1"], r["shape_b2"], format(r["params"], ","), r["adapted"]))
        except Exception as e:                                            # noqa: BLE001
            print("%-18s FAIL: %s" % (name, str(e)[:90]))
