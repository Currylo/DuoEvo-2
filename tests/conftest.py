import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duoevo  # noqa: E402  (also puts evolve/ on sys.path)
from coevolve_bearing.data import BearingSplits  # noqa: E402


@pytest.fixture
def tiny_splits():
    """Two-class synthetic splits with short windows."""
    rng = np.random.default_rng(0)

    def part(n):
        return rng.standard_normal((n, 256)).astype(np.float32), np.arange(n) % 2

    arrays = {}
    for name, n in (("train", 40), ("dev", 20), ("test", 20), ("cross", 10)):
        arrays[f"{name}_X"], arrays[f"{name}_y"] = part(n)
    return BearingSplits(meta={"n_classes": 2}, **arrays)


@pytest.fixture
def tiny_cfg():
    cfg = duoevo.load_config(ROOT / "configs" / "cwru.yaml", 11)
    cfg["data"].update(window_len=256, n_classes=2)
    return cfg
