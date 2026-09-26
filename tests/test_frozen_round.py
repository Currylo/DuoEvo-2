"""A round with the structural branch closed must cost exactly one weight continuation."""
import pytest

import duoevo as D


def row(**changes):
    r = {"arm": "Evolve-NoStruct", "round": 4, "structure": {"action": "id"}, "search": {},
         "confirm": [], "promotion": {"searched": False, "promoted": False},
         "champion_is_anchor": True,
         "cost": {"round_epochs": 32.0, "search_epochs_this_round": 0.0, "round_llm_calls": 16}}
    r.update(changes)
    return r


def test_valid_frozen_round_passes():
    D.assert_frozen_round_is_contained(row(), 32, 16, anchor_only=True)
    D.assert_frozen_round_is_contained(
        row(cost={"round_epochs": 32.0, "search_epochs_this_round": 0.0, "round_llm_calls": 0}),
        32, 16, no_challenger=True)


@pytest.mark.parametrize("bad", [
    {"search": {"proposals": 40}},
    {"confirm": [{"role": "incumbent"}]},
    {"promotion": {"searched": True, "promoted": False}},
    {"champion_is_anchor": False},
    {"cost": {"round_epochs": 136.0, "search_epochs_this_round": 104.0, "round_llm_calls": 16}},
    {"cost": {"round_epochs": 32.0, "search_epochs_this_round": 0.0, "round_llm_calls": 0}},
])
def test_violations_are_caught(bad):
    with pytest.raises(AssertionError, match="not a frozen round"):
        D.assert_frozen_round_is_contained(row(**bad), 32, 16, anchor_only=True)
