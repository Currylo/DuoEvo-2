"""Resume checkpoints refuse to continue a different run."""
import random

import pytest

from resume import ResumeError, open_progress, restore_rng, save_progress, save_rng

CFG = {"rounds": 20}


def test_fresh_start_and_identity_checks(tmp_path):
    assert open_progress(tmp_path, "DuoEvo", 11, CFG, resume=False) is None
    with pytest.raises(ResumeError, match="no resume checkpoint"):
        open_progress(tmp_path, "DuoEvo", 11, CFG, resume=True)
    save_progress(tmp_path, {"version": 1, "arm": "DuoEvo", "seed": 11, "exp_cfg": CFG, "round": 3})
    with pytest.raises(ResumeError, match="pass --resume"):
        open_progress(tmp_path, "DuoEvo", 11, CFG, resume=False)
    assert open_progress(tmp_path, "DuoEvo", 11, CFG, resume=True)["round"] == 3
    with pytest.raises(ResumeError, match="identity"):
        open_progress(tmp_path, "DuoEvo", 23, CFG, resume=True)
    with pytest.raises(ResumeError, match="configuration"):
        open_progress(tmp_path, "DuoEvo", 11, {"rounds": 12}, resume=True)


def test_search_rng_round_trip(tmp_path):
    rng = random.Random(5)
    rng.random()
    save_rng(tmp_path / "rng.pt", rng)
    expected = [rng.random() for _ in range(3)]
    fresh = random.Random(0)
    restore_rng(tmp_path / "rng.pt", fresh, completed=1)
    assert [fresh.random() for _ in range(3)] == expected
