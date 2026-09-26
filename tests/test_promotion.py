"""Promotion gate and switch audit."""
import pytest

import duoevo as D

GATES = {"delta_promote": 0.0283, "epsilon_clean": 0.01, "epsilon_worst": 0.0845,
         "epsilon_family": 0.03}
FAMS = ["colored_bandpass", "harmonic_line", "impulsive_resonance", "bursty", "complex_mixed"]


def rec(role, gate_mean, *, clean=1.0, worst=None, fams=None, params=1000, sha=None):
    return {"role": role, "id": role, "screen": gate_mean, "cell": None, "params": params,
            "source_sha256": sha or role, "gate_mean": gate_mean, "clean": clean,
            "gate_worst": gate_mean if worst is None else worst,
            "gate_families": dict(zip(FAMS, fams or [gate_mean] * 5))}


def test_promotes_when_every_condition_holds():
    d = D.decide_promotion([rec("incumbent", 0.80), rec("top_score", 0.85)], GATES, 104, 648)
    assert d["promoted"] and d["winner"]["role"] == "top_score"
    assert [c["name"] for c in d["conditions"]] == [
        "beats_incumbent_by_delta", "clean_not_degraded", "worst_not_degraded", "no_family_regressed"]


def test_margin_below_delta_is_not_enough():
    d = D.decide_promotion([rec("incumbent", 0.80), rec("top_score", 0.82)], GATES, 104, 648)
    assert not d["promoted"] and "beats_incumbent_by_delta" in d["reason"]
    assert d["primary_margin"] == pytest.approx(0.82 - 0.80 - 0.0283)


@pytest.mark.parametrize("mutation, failed", [
    (rec("top_score", 0.90, clean=0.98), "clean_not_degraded"),
    (rec("top_score", 0.90, worst=0.70), "worst_not_degraded"),
    (rec("top_score", 0.90, fams=[0.99, 0.99, 0.75, 0.99, 0.99]), "no_family_regressed"),
])
def test_each_guardrail_can_block(mutation, failed):
    d = D.decide_promotion([rec("incumbent", 0.80, worst=0.80), mutation], GATES, 104, 648)
    assert not d["promoted"]
    assert [c["name"] for c in d["conditions"] if not c["passed"]] == [failed]


def test_guardrails_compare_against_the_incumbent_source():
    # The incumbent source (retrained at the confirm budget) is the only reference.
    inc = rec("incumbent", 0.80, fams=[0.80] * 5)
    cand = rec("novel_cell", 0.90, fams=[0.90] * 5)
    d = D.decide_promotion([inc, cand], GATES, 104, 648)
    fam = next(c for c in d["conditions"] if c["name"] == "no_family_regressed")
    assert d["promoted"] and fam["drop"] == pytest.approx(0.10)


def test_ties_prefer_the_smaller_network():
    muts = [rec("top_score", 0.9, params=2000, sha="a"), rec("novel_cell", 0.9, params=1000, sha="b")]
    d = D.decide_promotion([rec("incumbent", 0.8)] + muts, GATES, 104, 648)
    assert d["winner"]["role"] == "novel_cell"


def test_no_mutation_means_no_promotion():
    d = D.decide_promotion([rec("incumbent", 0.8)], GATES, 104, 648)
    assert not d["promoted"] and d["reason"] == "no valid mutation in the shortlist"


def test_switch_audit_keeps_the_champion_on_a_tie_and_rolls_back_on_a_loss():
    anchor = rec("a", 0.85)
    assert not D.decide_anchor_fallback(rec("c", 0.85), anchor, GATES)["anchor_preferred"]
    assert D.decide_anchor_fallback(rec("c", 0.84), anchor, GATES)["anchor_preferred"]
