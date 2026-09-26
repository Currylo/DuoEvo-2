"""Structure-search lineage: the proposal tree, parent sampling and quality rollback.

Each search window grows a tree of Solver programs from the current anchor.  A parent is sampled
either from the deepest surviving lineages or from a MAP-Elites archive over (log10 params, #conv).
When a sampled chain has not improved its best 8-epoch screening score for `patience` generations,
breeding restarts from that best node instead.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

SEARCH_ADDENDUM = (
    "\n\nSANDBOX EXPLORATION WINDOW: you are inside a protected lineage window. Intermediate "
    "generations do NOT need to beat the anchor's score — you may deliberately pass through "
    "temporarily-worse stepping stones while developing a structural idea across generations. "
    "The whole window is judged only at the end by full-fidelity training of a few endpoints. "
    "Build on the PARENT program you are given (do not restart from the official seed unless "
    "the parent is broken)."
)


class LLMConnectionError(RuntimeError):
    """The endpoint stayed unreachable after every retry.

    Distinguishes an infrastructure outage (retry the same proposal slot without charging the
    budget) from a response that contains no usable code (charged: that is real search signal).
    """


def extract_code(text: str) -> str | None:
    """The longest fenced Python block that defines `build_solver`, or None."""
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    if not m:
        return None
    code = max(m, key=len)
    return code if "def build_solver" in code else None


def module_signature(program_src):
    """(solver_lib modules used, hand-written block types) of a Solver program."""
    mods = sorted(set(re.findall(r"solver_lib\.(\w+)", program_src or "")))
    hand = [kw for kw in ("MultiheadAttention", "Sigmoid", "GLU", "LayerNorm", "GroupNorm")
            if kw in (program_src or "")]
    return mods, hand


def fp_distance(src: str, anchor_src: str) -> dict:
    """Structural distance from the anchor: symmetric difference of module signatures."""
    m1, h1 = module_signature(src)
    m0, h0 = module_signature(anchor_src)
    return {"mods_added": sorted(set(m1) - set(m0)), "mods_removed": sorted(set(m0) - set(m1)),
            "hand_added": sorted(set(h1) - set(h0)), "hand_removed": sorted(set(h0) - set(h1)),
            "d_fp": len(set(m1) ^ set(m0)) + len(set(h1) ^ set(h0))}


class Lineage:
    """Append-only proposal tree persisted to `<out_dir>/lineage.json`."""

    def __init__(self, out_dir: Path):
        self.path = Path(out_dir) / "lineage.json"
        self.nodes = []
        if self.path.exists():
            self.nodes = json.loads(self.path.read_text())["nodes"]

    def add(self, rec: dict) -> None:
        self.nodes.append(rec)
        self.path.write_text(json.dumps({"nodes": self.nodes}, indent=1))

    def valid_nodes(self):
        return [n for n in self.nodes if n["valid"] and n["id"] != "anchor"]

    def node(self, nid):
        return next(n for n in self.nodes if n["id"] == nid)

    def chain(self, nid):
        out = []
        cur = self.node(nid)
        while cur is not None:
            out.append(cur)
            cur = self.node(cur["parent"]) if cur["parent"] else None
        return list(reversed(out))

    def deep_chain_count(self, gmin=8):
        tips = {n["id"] for n in self.valid_nodes()}
        for n in self.valid_nodes():
            tips.discard(n["parent"])
        return sum(1 for t in tips if self.node(t)["g"] >= gmin)


@dataclass(frozen=True)
class ParentDecision:
    sampled_parent_id: str
    parent_id: str
    source: str
    best_parent_id: str
    stale_steps: int
    rolled_back: bool


def _anchor_decision(source="anchor"):
    return ParentDecision("anchor", "anchor", source, "anchor", 0, False)


def pick_parent(lin: Lineage, rng, gmax, archive, patience=0) -> ParentDecision:
    """Depth-biased or archive parent sampling, with quality rollback after `patience` stale
    generations."""
    pool = [n for n in lin.valid_nodes() if n["g"] < gmax]
    if not pool:
        return _anchor_decision("empty_pool")
    if rng.random() < 0.5:          # depth-biased: uniform among tips of the deepest lineages
        gm = max(n["g"] for n in pool)
        deep = [n for n in pool if n["g"] >= max(1, gm - 1)]
        sampled = rng.choice(deep)["id"]
        source = "depth"
    else:
        elite_ids = [nid for nid in archive.values()
                     if nid != "anchor" and lin.node(nid)["valid"] and lin.node(nid)["g"] < gmax]
        if not elite_ids:
            return _anchor_decision("archive_empty")
        sampled = rng.choice(elite_ids)
        source = "archive"

    chain = lin.chain(sampled)
    scored = [(i, n["reward"]) for i, n in enumerate(chain) if n.get("reward") is not None]
    best_i = max(scored, key=lambda item: item[1])[0]
    best_id = chain[best_i]["id"]
    stale_steps = len(chain) - 1 - best_i
    rolled_back = patience > 0 and stale_steps >= patience
    parent_id = best_id if rolled_back else sampled
    return ParentDecision(sampled, parent_id, source, best_id, stale_steps, rolled_back)
