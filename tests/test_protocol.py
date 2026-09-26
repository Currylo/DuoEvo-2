"""Every dataset runs the same DuoEvo protocol; only the data block and MART-LOAT settings
differ."""
import copy

import pytest

import duoevo as D
from conftest import ROOT

DATASETS = ("pu", "cwru", "jnu")


def _cfg(name):
    return D.load_config(ROOT / "configs" / f"{name}.yaml", 11)


def test_protocol_is_identical_across_datasets():
    cfgs = [_cfg(d) for d in DATASETS]
    for block in ("exp", "curriculum", "eval", "noise", "train"):
        assert all(c[block] == cfgs[0][block] for c in cfgs), block
    shared = {k: v for k, v in cfgs[0]["data"].items() if k in ("sample_rate_hz", "window_len")}
    assert all({k: c["data"][k] for k in shared} == shared for c in cfgs)


def test_deployment_budget_is_648_epochs():
    exp = _cfg("cwru")["exp"]
    assert exp["boot_epochs"] + exp["rounds"] * exp["continue_epochs"] == 648


def test_configured_arms():
    cfg = _cfg("pu")
    assert D.arm_spec("DuoEvo", cfg) == {"structure": "patience", "curriculum": "evolve",
                                         "patience": 3, "max_search_rounds": 8}
    assert D.arm_spec("Evolve-NoStruct", cfg)["structure"] == "never"
    assert D.arm_spec("Replay-Anchor", cfg)["donor"] == "DuoEvo"
    assert D.arm_spec("Random-Noise", cfg)["curriculum"] == "random"


@pytest.mark.parametrize("spec, match", [
    ({"structure": "sometimes"}, "structure policy"),
    ({"curriculum": "static"}, "curriculum source"),
    ({"curriculum": "replay"}, "donor"),
])
def test_bad_arm_specs_are_rejected(spec, match):
    cfg = copy.deepcopy(_cfg("cwru"))
    cfg["exp"]["arms"]["Bad"] = spec
    with pytest.raises(ValueError, match=match):
        D.arm_spec("Bad", cfg)
    with pytest.raises(KeyError):
        D.arm_spec("Missing", cfg)


def test_patience_router_closes_after_consecutive_failures():
    spec = {"structure": "patience", "patience": 3, "max_search_rounds": 8}
    stale, nxt = 0, 1
    for t in (1, 2):
        assert D.route_structure(t, nxt, spec)
        stale, nxt = D.advance_router(t, False, stale, spec)
        assert nxt == t + 1
    stale, nxt = D.advance_router(3, False, stale, spec)
    assert nxt == D.NEVER_SEARCH and not D.route_structure(4, nxt, spec)
    assert D.apply_search_budget(nxt, 3, spec) == (D.NEVER_SEARCH, "patience")


def test_promotion_resets_the_streak_and_budget_caps_searches():
    spec = {"structure": "patience", "patience": 3, "max_search_rounds": 8}
    assert D.advance_router(5, True, 2, spec) == (0, 6)
    assert D.apply_search_budget(9, 8, spec) == (D.NEVER_SEARCH, "budget")
    assert D.apply_search_budget(9, 7, spec) == (9, None)


def test_never_and_always():
    assert not D.route_structure(1, 1, {"structure": "never"})
    assert D.route_structure(1, D.NEVER_SEARCH, {"structure": "always"})
