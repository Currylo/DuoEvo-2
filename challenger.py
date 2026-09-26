"""Challenger branch: evolve noise-generator programs against the frozen Solver.

A Challenger program defines `generate_noise(rng, fs, n)` and returns signal-independent additive
noise.  `evolve/evaluator_gen.py` injects it into a fixed class-balanced dev subset at a locked SNR
and scores it by the frozen Solver's classification margin (reward = -margin), so a program can only
win by the structure of its noise, never by its energy.  Scored programs enter a MAP-Elites archive
keyed by waveform descriptors (`evolve/gen_archive.py`).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
EVOLVE = ROOT / "evolve"
CONFIGS = ROOT / "configs"
for _p in (ROOT, EVOLVE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from coevolve_bearing.metrics import margin_score                 # noqa: E402
from coevolve_bearing.utils import ensure_dir, to_jsonable, write_json  # noqa: E402

SEED_SOLVER = EVOLVE / "seed_solver.py"          # ClassBD anchor, about 4.27M parameters
SEED_GEN = EVOLVE / "seed_generator.py"
EVAL_GEN = EVOLVE / "evaluator_gen.py"
CONF_GEN = CONFIGS / "llm" / "challenger.yaml"


def run_oe(seed_program: Path, evaluator: Path, oe_config: Path, iterations: int, out_dir: Path):
    """One OpenEvolve pass; `iterations` overrides `max_iterations` in the YAML."""
    from openevolve import run_evolution
    from oe_glm_patch import disable_glm_thinking
    disable_glm_thinking()          # GLM reasoning tokens would eat the budget and truncate code
    run_evolution(initial_program=str(seed_program), evaluator=str(evaluator),
                  config=str(oe_config), iterations=int(iterations),
                  output_dir=str(ensure_dir(out_dir)), cleanup=False)


def read_harvest(state_dir) -> list:
    """Every candidate the evaluator scored, one JSON record per line."""
    p = Path(state_dir) / "harvest.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def valid_records(records: list) -> list:
    return [r for r in records if int(r.get("valid", 0)) == 1]


def freeze_solver(model, program_src, probe, cfg, state_dir, seed, n_classes, win):
    """Write the current Solver where the Challenger evaluator loads it.

    Evolved Solver classes are defined inside `build_solver`, so the module itself cannot be
    pickled; the evaluator rebuilds it from {state_dict, program_src}.
    """
    state_dir = ensure_dir(state_dir)
    m_clean = float(margin_score(model, probe.vedge_X, probe.vedge_y, cfg))
    model.to("cpu")
    torch.save({"state_dict": model.state_dict(), "program_src": program_src,
                "n_classes": int(n_classes), "length": int(win)}, state_dir / "solver.pt")
    np.savez(state_dir / "vedge.npz", X=probe.vedge_X, y=probe.vedge_y)
    write_json(state_dir / "meta.json",
               {"m_clean": m_clean, "seed": int(seed), "cfg": to_jsonable(cfg)})
    return m_clean


def rescore_archive(archive, state_dir):
    """Re-score every archived program against the currently frozen Solver.

    An elite's reward was earned against an earlier opponent.  Once the opponent changes that
    fitness is stale, and a stale elite could block a program that is harder for the current
    Solver.  Forward passes only.  Returns the number of re-scored elites.
    """
    if not archive.cells:
        return 0
    import evaluator_gen
    evaluator_gen._STATE.clear()          # rebind the scorer to the just-frozen solver
    tmp = Path(state_dir) / "_rescore"
    tmp.mkdir(parents=True, exist_ok=True)
    harvest = Path(state_dir) / "harvest.jsonl"
    saved = harvest.read_text(encoding="utf-8") if harvest.exists() else ""
    n = 0
    for entry in list(archive.cells.values()):
        p = tmp / "cand.py"
        p.write_text(entry["program"], encoding="utf-8")
        try:
            res = evaluator_gen.evaluate(str(p))
        except Exception:                  # a program that no longer builds keeps its old score
            continue
        if float(res.get("valid", 0.0)) == 1.0:
            entry["reward"] = float(res["combined_score"])
            n += 1
    harvest.write_text(saved, encoding="utf-8")     # re-scoring must not enter the harvest
    return n


def challenger_evolve(rnd, out_dir, state_dir, iters):
    """Evolve noise generators against the frozen Solver; return the valid harvested records."""
    (Path(state_dir) / "harvest.jsonl").write_text("", encoding="utf-8")
    os.environ["COEVO_STATE_DIR"] = str(state_dir)
    # The evaluator caches the frozen Solver in `_STATE`, and OpenEvolve scores the initial program
    # in this process: without clearing, round t would be scored against round 1's Solver.
    import evaluator_gen
    evaluator_gen._STATE.clear()
    run_oe(SEED_GEN, EVAL_GEN, CONF_GEN, iters, Path(out_dir) / f"challenger_round{rnd}")
    return valid_records(read_harvest(state_dir))
