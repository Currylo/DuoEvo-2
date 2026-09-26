"""Curriculum materialization, replay and the accumulated-schedule rebuild."""
import json

import numpy as np
import pytest

import duoevo as D
from coevolve_bearing import augment
from gen_archive import GenMapElites

KEYS = ["train_X", "train_y", "dev_X", "dev_y", "oodv_X", "oodv_y", "sel_X", "sel_y"]


def test_backfill_keeps_rows_per_elite_constant():
    assert D.effective_mix_random(0.5, 8, 8) == 0.5
    assert D.effective_mix_random(0.5, 4, 8) == pytest.approx(0.75)
    assert D.effective_mix_random(0.5, 0, 8) == 1.0
    for n in (1, 3, 8):                     # archive rows per elite = (1 - mix) / n * cap
        assert (1 - D.effective_mix_random(0.5, n, 8)) / n == pytest.approx(0.5 / 8)


def test_freeze_is_deterministic_and_complete(tiny_splits, tiny_cfg, tmp_path):
    a = D.freeze_curriculum(0, tmp_path / "a", tiny_splits, GenMapElites(), tiny_cfg, 11, 1.0, 8)
    b = D.freeze_curriculum(0, tmp_path / "b", tiny_splits, GenMapElites(), tiny_cfg, 11, 1.0, 8)
    assert list(np.load(a[1] / "data.npz").files) == KEYS
    assert a[2]["data_sha256"] == b[2]["data_sha256"] == D._sha256(a[1] / "data.npz")
    n_aug = round(len(tiny_splits.train_X) * tiny_cfg["curriculum"]["aug_fraction"])
    assert a[2]["n_aug"] == n_aug and len(a[0]["sel_X"]) == len(tiny_splits.dev_X)


def test_random_pool_reuses_a_fixed_number_of_programs(tiny_splits, tiny_cfg, tmp_path, monkeypatch):
    calls = []
    original = augment.random_program
    monkeypatch.setattr(augment, "random_program", lambda *a, **k: calls.append(1) or original(*a, **k))
    augment.rand_augment(tiny_splits.train_X, tiny_splits.train_y, 12000, 256, 1.0, [0.0], 3, pool_size=4)
    assert len(calls) == 4
    calls.clear()
    augment.rand_augment(tiny_splits.train_X, tiny_splits.train_y, 12000, 256, 1.0, [0.0], 3)
    assert len(calls) == len(tiny_splits.train_X)


def test_replay_is_byte_identical_and_checks_the_donor(tiny_splits, tiny_cfg, tmp_path):
    donor = tmp_path / "donor"
    cur, _, meta = D.freeze_curriculum(1, donor, tiny_splits, GenMapElites(), tiny_cfg, 11, 0.5, 8)
    (donor / "round_log.json").write_text(json.dumps([{"round": 1, "curriculum": meta}]))
    got, _, rmeta = D.replay_curriculum(1, tmp_path / "recipient", donor)
    assert rmeta["data_sha256"] == meta["data_sha256"]
    assert all(np.array_equal(got[k], cur[k]) for k in KEYS)

    (donor / "round_log.json").write_text(json.dumps([{"round": 1, "curriculum": {**meta, "data_sha256": "0" * 64}}]))
    with pytest.raises(RuntimeError, match="donor sha"):
        D.replay_curriculum(1, tmp_path / "recipient2", donor)
    with pytest.raises(FileNotFoundError, match="no round 2"):
        D.replay_curriculum(2, tmp_path / "recipient3", donor)


def _materialize(arm_dir, rounds):
    for k in range(rounds + 1):
        d = arm_dir / f"round{k}" / "curriculum"
        d.mkdir(parents=True)
        np.savez(d / "data.npz", train_X=np.full((4, 8), float(k), np.float32),
                 train_y=np.zeros(4, np.int64))


@pytest.fixture
def train_spy(monkeypatch):
    calls = []

    def fake_train(src, epochs, seed, cur, cfg, init_state=None):
        calls.append({"epochs": epochs, "seed": seed, "round": float(cur["train_X"][0, 0]),
                      "warm": init_state is not None})
        return {"w": len(calls)}, 0.0

    monkeypatch.setattr(D, "train_official", fake_train)
    return calls


def test_rebuild_walks_the_schedule(tmp_path, train_spy):
    _materialize(tmp_path, 6)
    _, spent = D.rebuild_on_accumulated_schedule("src", tmp_path, 6, 8, 32, 61000, {})
    assert spent == 8 + 6 * 32
    assert [c["round"] for c in train_spy] == [0, 1, 2, 3, 4, 5, 6]
    assert [c["epochs"] for c in train_spy] == [8] + [32] * 6
    assert [c["warm"] for c in train_spy] == [False] + [True] * 6
    assert len({c["seed"] for c in train_spy}) == 7


def test_rebuild_refuses_a_missing_round(tmp_path, train_spy):
    _materialize(tmp_path, 4)
    (tmp_path / "round3" / "curriculum" / "data.npz").unlink()
    with pytest.raises(FileNotFoundError, match="round 3"):
        D.rebuild_on_accumulated_schedule("src", tmp_path, 4, 8, 32, 61000, {})
