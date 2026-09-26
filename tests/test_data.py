"""Data splits: time-ordered splits share no raw samples; subsets are balanced and fixed."""
import numpy as np
import pytest

from coevolve_bearing import data


@pytest.mark.parametrize("stride", [256, 1024])
def test_time_split_purges_overlapping_windows(stride):
    win, n = 2048, 200
    windows = np.stack([np.arange(i * stride, i * stride + win) for i in range(n)])
    train, dev, test = data._time_split(windows, win, stride)
    assert train.max() < dev.min() and dev.max() < test.min()
    assert len(train) == int(0.6 * n)
    assert len(dev) == int(0.8 * n) - int(0.6 * n) - (win // stride - 1)


def test_resampling_to_12khz():
    sig = np.random.default_rng(0).standard_normal(64000)
    assert len(data._resample(sig, 64000, 12000)) == 12000
    assert len(data._resample(sig, 50000, 12000)) == round(64000 * 12 / 50)
    assert data._resample(sig, 12000, 12000) is sig


def test_unknown_dataset():
    with pytest.raises(ValueError, match="unknown dataset"):
        data.load_splits({"data": {"dataset": "mfpt"}})


def test_edge_probe_is_balanced_disjoint_and_fixed(tiny_splits):
    a = data.carve_edge_probe(tiny_splits, n_per_class=4, seed=11)
    b = data.carve_edge_probe(tiny_splits, n_per_class=4, seed=11)
    assert np.array_equal(a.vedge_X, b.vedge_X)
    assert np.bincount(a.vedge_y).tolist() == [4, 4]
    rows = {r.tobytes() for r in a.vedge_X}
    assert not rows & {r.tobytes() for r in a.probe_X}
