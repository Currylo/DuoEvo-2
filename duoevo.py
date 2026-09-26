#!/usr/bin/env python
"""DuoEvo: Challenger-Solver co-evolution for noise-robust bearing fault diagnosis.

Every macro round follows S(t-1) -> C(t) -> S(t):

  1. Challenger   LLM-guided evolution of noise-generator programs against the frozen S(t-1).
                  Archived programs are re-scored against the current opponent first.
  2. Curriculum   C(t) is materialized once: half fresh random-DSL noise (coverage), half the
                  archive's hardest programs (pressure).  Every training run in the round reads it.
  3. Weights      the deployed checkpoint continues for `continue_epochs` on C(t).
  4. Structure    when the router opens the structural branch, an LLM proposes network rewrites;
                  three sources are retrained from scratch at `confirm_epochs` and the promotion
                  gate decides whether a new structure is deployed.

The arms are defined in configs/base.yaml.  The held-out-noise test (T2) is never read by `run`;
`evaluate` scores the saved checkpoints once every arm has finished.

    python duoevo.py run       --config configs/cwru.yaml --seed 11
    python duoevo.py evaluate  --config configs/cwru.yaml --seed 11
    python duoevo.py calibrate --config configs/cwru.yaml --seed 11
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# `challenger` puts evolve/ on sys.path, so it is imported before the evolve modules.
from challenger import CONFIGS, SEED_SOLVER, challenger_evolve, freeze_solver, rescore_archive
from coevolve_bearing.augment import gen_augment, oodval_from_curriculum, rand_augment
from coevolve_bearing.config import deep_update
from coevolve_bearing.data import carve_edge_probe, load_splits
from coevolve_bearing.t2 import evaluate_t2
from coevolve_bearing.metrics import balanced_accuracy
from coevolve_bearing.train import predict
from coevolve_bearing.utils import ensure_dir, seed_everything, to_jsonable, write_json
from evaluator_gen import load_generator_from_source
from evaluator_solver import load_builder_from_source, seed_eval
from evaluator_solver_family import family_scores
from gen_archive import GenMapElites
from lineage import SEARCH_ADDENDUM, LLMConnectionError, Lineage, extract_code, fp_distance, pick_parent
from resume import ResumeError, open_progress, restore_rng, save_progress, save_rng

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- promotion gate -----------------------------------------------------------------------------
# The gate reads the same dev windows as the structure search but with its own noise seeds and on
# SNRs the curriculum never uses (curriculum and T2: 0 / -3 / -6 dB), so it is not the ruler the
# search already optimizes.
GATE_SEED_OFFSET = 900_000
GATE_SNRS = (-1.5, -4.5, -7.5)
# `gate_worst` is the mean of the two weakest families.  Over ten retrainings of one source the raw
# minimum had sigma 0.039 and the two-worst mean 0.017, so the trimmed statistic is the usable one.
GATE_WORST_K = 2
EPSILON_ANCHOR = 0.0            # switch audit: the champion may not fall below the anchor
NEVER_SEARCH = 10 ** 9          # `next_search_at` sentinel: the structural branch is closed


# ================================ configuration ====================================================

def load_config(path, seed) -> dict:
    """configs/base.yaml overlaid with a dataset config; the seed comes from the command line."""
    cfg = yaml.safe_load((CONFIGS / "base.yaml").read_text(encoding="utf-8"))
    override = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = deep_update(cfg, override)
    cfg["project"]["seed"] = int(seed)
    return cfg


def arm_spec(arm, cfg) -> dict:
    """An arm is a structure policy plus a curriculum source (`exp.arms` in configs/base.yaml).

    structure   never | always | patience   (patience closes the branch after `patience`
                consecutive non-promoting searches or after `max_search_rounds` searches)
    curriculum  evolve (own Challenger) | replay (a donor arm's curricula, byte for byte) |
                random (random DSL programs, no Challenger)
    """
    arms = cfg["exp"]["arms"]
    if arm not in arms:
        raise KeyError(f"unknown arm {arm!r}; configured arms: {sorted(arms)}")
    spec = {"structure": "never", "curriculum": "evolve", **(arms[arm] or {})}
    if spec["structure"] not in ("never", "always", "patience"):
        raise ValueError(f"arm {arm!r}: unknown structure policy {spec['structure']!r}")
    if spec["curriculum"] not in ("evolve", "replay", "random"):
        raise ValueError(f"arm {arm!r}: unknown curriculum source {spec['curriculum']!r}")
    if spec["curriculum"] == "replay" and not spec.get("donor"):
        raise ValueError(f"arm {arm!r}: a replayed curriculum needs a `donor` arm")
    return spec


# ================================ structure router =================================================

def route_structure(t, next_search_at, spec) -> bool:
    """Does round `t` open the structural branch?"""
    mode = spec["structure"]
    if mode == "never":
        return False
    if mode == "always":
        return True
    return int(t) >= int(next_search_at)


def advance_router(t, promoted, stale_searches, spec):
    """Router state after a search round -> (stale_searches, next_search_at).

    A promotion resets the streak.  Under `patience`, `patience` consecutive non-promoting searches
    close the branch for good: the arm then keeps evolving its curriculum and accumulating weights
    on the deployed structure.
    """
    if promoted:
        return 0, int(t) + 1
    stale = int(stale_searches) + 1
    if spec["structure"] == "patience" and stale >= int(spec.get("patience", 2)):
        return stale, NEVER_SEARCH
    return stale, int(t) + 1


def apply_search_budget(next_search_at, searches_done, spec):
    """Close the branch once `max_search_rounds` searches have run -> (next_search_at, lock_reason)."""
    cap = int(spec.get("max_search_rounds", 0))
    if cap > 0 and int(searches_done) >= cap:
        return NEVER_SEARCH, "budget"
    return next_search_at, ("patience" if int(next_search_at) >= NEVER_SEARCH else None)


def search_round_cost(stats, n_confirm, cfg, rebuild_epochs=0) -> float:
    """Anchor-epoch equivalents charged by a search round on top of the weight continuation."""
    return float(int(stats.get("proposals", 0)) * int(cfg["exp"]["solver_eval_epochs"])
                 + int(n_confirm) * int(cfg["exp"]["confirm_epochs"])
                 + int(rebuild_epochs))


# ================================ curriculum =======================================================

def materialize_capped(archive: GenMapElites, cap: int) -> list:
    """The archive's `cap` highest-reward elites as callable generators.

    The archive grows every round; materializing all of it would dilute the share of augmented
    samples that carries any single hard program.
    """
    top = sorted(archive.cells.values(), key=lambda v: -float(v["reward"]))[:int(cap)]
    gens = []
    for rec in top:
        try:
            gens.append(load_generator_from_source(rec["program"]))
        except Exception:                   # a program that no longer builds simply drops out
            continue
    return gens


def effective_mix_random(mix_random_frac, n_gens, archive_cap) -> float:
    """Random-DSL share of the augmented volume after routing an archive shortfall to it.

    With fewer than `archive_cap` elites the archive half would concentrate the same volume on
    fewer programs.  Moving the missing share to the random half keeps rows per elite constant.
    """
    mix = float(mix_random_frac)
    if int(archive_cap) <= 0:
        return mix
    fill = min(int(n_gens), int(archive_cap)) / float(archive_cap)
    return mix + (1.0 - mix) * (1.0 - fill)


def build_curriculum(splits, gens, oodv, fs, win, frac, snrs, seed, mix, rand_pool=0) -> dict:
    """Clean train windows plus `frac * len(train)` noisy copies, split `1 - mix` / `mix` between
    archive generators and random DSL programs."""
    mix = min(max(float(mix), 0.0), 1.0)
    Xa, ya = gen_augment(splits.train_X, splits.train_y, gens, fs, win, frac * (1.0 - mix), snrs, seed)
    if mix > 0.0:
        rX, ry = rand_augment(splits.train_X, splits.train_y, fs, win, frac * mix, snrs, seed + 77,
                              pool_size=rand_pool)
        if len(rX):
            Xa, ya = np.concatenate([Xa, rX]), np.concatenate([ya, ry])
    return {"train_X": Xa, "train_y": ya, "dev_X": splits.dev_X, "dev_y": splits.dev_y,
            "oodv_X": oodv[0], "oodv_y": oodv[1]}


def selection_fixture(splits, gens, cfg, fs, win, seed, mix_random_frac, rand_pool=0):
    """Checkpoint-selection set: dev windows carrying the round's curriculum noise.

    Clean dev saturates at 1.0 within a few epochs, so selecting on it would freeze the checkpoint
    early and discard the rest of the robustness training.  This set uses the full curriculum SNR
    grid and its own seed, so checkpoint selection, search screening (`oodv`), the promotion gate
    and T2 read four separate fixtures.
    """
    snrs = [float(s) for s in cfg["curriculum"]["aug_snrs_db"]]
    mix = min(max(float(mix_random_frac), 0.0), 1.0)
    n0 = len(splits.dev_X)
    parts_X, parts_y = [], []
    if mix < 1.0 and gens:
        X, y = gen_augment(splits.dev_X, splits.dev_y, gens, fs, win, 1.0 - mix, snrs, seed)
        if len(X) > n0:
            parts_X.append(X[n0:])
            parts_y.append(y[n0:])
    if mix > 0.0:
        rX, ry = rand_augment(splits.dev_X, splits.dev_y, fs, win, mix, snrs, seed + 77,
                              pool_size=rand_pool)
        if len(rX):
            parts_X.append(rX)
            parts_y.append(ry)
    if not parts_X:
        return splits.dev_X, splits.dev_y
    return (np.concatenate(parts_X, axis=0).astype(np.float32),
            np.concatenate(parts_y, axis=0))


def freeze_curriculum(rnd, arm_dir, splits, archive, cfg, seed, mix_random_frac, archive_cap,
                      rand_pool=0):
    """Materialize C(t) once and write it where the Solver evaluator reads it.

    The weight continuation, every search candidate and every confirm training of the round then
    train on byte-identical data.  Returns (curriculum, state_dir, meta).
    """
    fs = int(cfg["data"]["sample_rate_hz"])
    win = int(cfg["data"]["window_len"])
    frac = float(cfg["curriculum"]["aug_fraction"])
    snrs = [float(s) for s in cfg["curriculum"]["aug_snrs_db"]]
    gens = materialize_capped(archive, archive_cap)
    eff_mix = effective_mix_random(mix_random_frac, len(gens), archive_cap)

    oodv = oodval_from_curriculum(splits, gens, cfg, fs, win, seed + 100 + rnd)
    cur = build_curriculum(splits, gens, oodv, fs, win, frac, snrs, seed + rnd, eff_mix, rand_pool)
    cur["sel_X"], cur["sel_y"] = selection_fixture(splits, gens, cfg, fs, win, seed + 200 + rnd,
                                                   eff_mix, rand_pool)

    state_dir = ensure_dir(arm_dir / f"round{rnd}" / "curriculum")
    np.savez(state_dir / "data.npz", **cur)
    write_json(state_dir / "meta.json",
               {"cfg": to_jsonable(cfg), "seed": int(seed), "n_classes": int(cfg["data"]["n_classes"]),
                "length": win})
    (state_dir / "harvest.jsonl").write_text("", encoding="utf-8")
    n_clean = len(splits.train_X)
    n_aug = int(len(cur["train_X"]) - n_clean)
    meta = {"archive_cells": archive.coverage(), "gens_materialized": len(gens),
            "archive_cap": int(archive_cap), "mix_random_frac": float(mix_random_frac),
            "mix_random_frac_effective": float(eff_mix), "random_pool": int(rand_pool),
            "aug_fraction": frac, "aug_snrs_db": snrs,
            "n_train_clean": n_clean, "n_train_total": int(len(cur["train_X"])), "n_aug": n_aug,
            "aug_rows_per_generator": (round(n_aug * (1.0 - eff_mix) / len(gens), 1) if gens else None),
            "data_sha256": _sha256(state_dir / "data.npz")}
    return cur, state_dir, meta


def load_curriculum_npz(path) -> dict:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def replay_curriculum(rnd, arm_dir, donor_arm_dir):
    """Copy round `rnd`'s curriculum from a donor arm instead of evolving one.

    Two arms that each evolve their own curriculum differ in structure and curriculum at once.
    Replaying makes the training data byte-identical round by round, so the difference isolates
    the structural branch.  The replayed curriculum was evolved against the donor's structure.
    """
    src = Path(donor_arm_dir) / f"round{rnd}" / "curriculum"
    src_npz = src / "data.npz"
    if not src_npz.exists():
        raise FileNotFoundError(f"replay donor has no round {rnd}: {src_npz} is missing. Run the "
                                f"donor arm to at least as many rounds first.")
    state_dir = ensure_dir(arm_dir / f"round{rnd}" / "curriculum")
    shutil.copyfile(src_npz, state_dir / "data.npz")
    shutil.copyfile(src / "meta.json", state_dir / "meta.json")
    (state_dir / "harvest.jsonl").write_text("", encoding="utf-8")
    cur = load_curriculum_npz(state_dir / "data.npz")

    donor = _donor_round_curriculum_meta(donor_arm_dir, rnd)
    sha = _sha256(state_dir / "data.npz")
    if donor.get("data_sha256") and donor["data_sha256"] != sha:
        raise RuntimeError(f"replay round {rnd}: copied sha {sha[:8]} != donor sha "
                           f"{donor['data_sha256'][:8]}")
    meta = {**donor, "data_sha256": sha, "replayed_from": str(src),
            "n_train_total": int(len(cur["train_X"]))}
    meta.setdefault("gens_materialized", 0)
    meta.setdefault("n_aug", None)
    return cur, state_dir, meta


def _donor_round_curriculum_meta(donor_arm_dir, rnd) -> dict:
    """The donor's own curriculum record for one round, or {}."""
    log = Path(donor_arm_dir) / "round_log.json"
    if rnd > 0 and log.exists():
        for r in json.loads(log.read_text(encoding="utf-8")):
            if int(r.get("round", -1)) == int(rnd):
                return dict(r.get("curriculum", {}))
    boot = Path(donor_arm_dir) / f"round{rnd}" / "champion.json"
    if rnd == 0 and boot.exists():
        return dict(json.loads(boot.read_text(encoding="utf-8")).get("curriculum", {}))
    return {}


def rebuild_on_accumulated_schedule(src, arm_dir, upto_round, ep_boot, ep_cont, seed_base, cfg):
    """Retrain `src` from scratch along the curriculum schedule the deployed checkpoint walked.

    `ep_boot` epochs on C(0), then `ep_cont` on each of C(1..t), each leg warm-started from the
    previous one.  The total equals the accumulated budget by construction, so a promoted or
    rolled-back source is deployed with the same training budget and the same curriculum exposure
    as the checkpoint it replaces.  Returns (state_dict, epochs_spent).
    """
    st, spent = None, 0
    for k in range(0, int(upto_round) + 1):
        npz = Path(arm_dir) / f"round{k}" / "curriculum" / "data.npz"
        if not npz.exists():
            raise FileNotFoundError(f"rebuild needs round {k}'s curriculum but {npz} is missing")
        ep = int(ep_boot if k == 0 else ep_cont)
        # 97 is coprime with the round stride, so no leg of any rebuild collides with another's seed.
        st, _ = train_official(src, ep, int(seed_base) + 97 * k,
                               load_curriculum_npz(npz), cfg, init_state=st)
        spent += ep
    return st, spent


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ================================ training and the gate ============================================

def set_search_env(state_dir, cfg):
    """Point the Solver evaluator at this round's frozen C(t)."""
    os.environ["COEVO_SOLVER_STATE_DIR"] = str(state_dir)
    os.environ["COEVO_SOLVER_EVAL_EPOCHS"] = str(int(cfg["exp"]["solver_eval_epochs"]))
    os.environ["COEVO_EPOCH_TIME_CAP_S"] = str(float(cfg["exp"]["epoch_time_cap_s"]))
    os.environ["COEVO_FAMILY_N"] = str(int(cfg["exp"]["family_probe_n"]))
    os.environ.pop("COEVO_FAMILY_SNRS", None)     # the search-side probe uses the curriculum SNRs
    import evaluator_solver
    evaluator_solver._STATE.clear()               # C(t) changed; the cached state must not survive


def train_official(src, epochs, seed, cur, cfg, init_state=None):
    """The published ClassBD recipe; `init_state=None` trains from scratch, otherwise warm-starts.

    SGD (lr 0.01, momentum 0.9), cosine annealing to 1e-8, batch 64, cross-entropy combined with
    the network's `aux_k` / `aux_g` physics terms by uncertainty weighting when it exposes them, no
    gradient clipping.  The checkpoint with the best balanced accuracy on the curriculum's
    selection fixture is kept.  Returns (state_dict, balanced accuracy on the screening set).
    """
    seed_eval(int(seed), src)
    model = load_builder_from_source(src)(int(cfg["data"]["n_classes"]),
                                          int(cfg["data"]["window_len"])).to(DEVICE)
    if init_state is not None:
        model.load_state_dict(init_state)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(epochs)), eta_min=1e-8)
    crit = nn.CrossEntropyLoss()
    xb = torch.from_numpy(np.asarray(cur["train_X"], dtype=np.float32)).unsqueeze(1)
    yb = torch.from_numpy(np.asarray(cur["train_y"], dtype=np.int64))
    bs, n = int(cfg["train"]["batch_size"]), len(xb)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best = -1.0
    for _ in range(int(epochs)):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            out = model(xb[idx].to(DEVICE))
            loss = crit(out, yb[idx].to(DEVICE))
            k, g = getattr(model, "aux_k", None), getattr(model, "aux_g", None)
            if k is not None and g is not None:
                ls = torch.tensor([-0.5, -0.5, -0.5], device=out.device)
                loss = (torch.stack([loss, k.to(out.device), g.to(out.device)])
                        / (3 * ls.exp()) + ls / 3).sum()
            loss.backward()
            opt.step()
        sched.step()
        b = balanced_accuracy(cur["sel_y"], predict(model, cur["sel_X"], cfg))
        if b > best:
            best = b
            best_state = {k2: v.detach().cpu().clone() for k2, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    ood = float(balanced_accuracy(cur["oodv_y"], predict(model, cur["oodv_X"], cfg)))
    return best_state, ood


_GLOBAL_CFG: dict = {}


class UndeployableCandidate(RuntimeError):
    """A trained checkpoint cannot be reloaded into a module freshly built from its own source
    (e.g. the candidate creates parameters lazily inside `forward`)."""


def build_model(src, state):
    """Rebuild an evolved Solver from (source, state_dict).

    Extra `aux_*` entries are tolerated: they are physics-loss tensors that some candidates
    register on their first forward pass and that never reach the classifier head.  Any other
    mismatch raises.
    """
    model = load_builder_from_source(src)(int(_GLOBAL_CFG["data"]["n_classes"]),
                                          int(_GLOBAL_CFG["data"]["window_len"])).to(DEVICE)
    res = model.load_state_dict(state, strict=False)
    if res.missing_keys:
        raise UndeployableCandidate(f"checkpoint does not cover {len(res.missing_keys)} weights, "
                                    f"e.g. {res.missing_keys[:5]}")
    stray = [k for k in res.unexpected_keys if not k.rsplit(".", 1)[-1].startswith("aux_")]
    if stray:
        raise UndeployableCandidate(f"checkpoint carries {len(stray)} unexpected non-aux weights, "
                                    f"e.g. {stray[:5]}")
    model.eval()
    return model


def count_params(src, state) -> int:
    model = build_model(src, state)
    n = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    del model
    torch.cuda.empty_cache()
    return n


def gate_score(src, state, splits, cfg, seed):
    """The promotion ruler: five canonical noise families on dev, gate-only seeds and SNRs.

    Fixed across rounds and arms, so `gate_mean` is comparable over the whole trajectory.
    """
    model = build_model(src, state)
    prev_snrs = os.environ.get("COEVO_FAMILY_SNRS")
    prev_n = os.environ.get("COEVO_FAMILY_N")
    os.environ["COEVO_FAMILY_SNRS"] = ",".join(str(s) for s in GATE_SNRS)
    os.environ["COEVO_FAMILY_N"] = str(int(cfg["exp"]["gate_probe_n"]))
    try:
        fams = family_scores(model, splits.dev_X, splits.dev_y, cfg,
                             int(cfg["data"]["sample_rate_hz"]), int(seed) + GATE_SEED_OFFSET)
    finally:
        for k, v in (("COEVO_FAMILY_SNRS", prev_snrs), ("COEVO_FAMILY_N", prev_n)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    clean = float(balanced_accuracy(splits.dev_y, predict(model, splits.dev_X, cfg)))
    del model
    torch.cuda.empty_cache()
    ranked = sorted(fams.values())
    return {"gate_mean": float(np.mean(list(fams.values()))),
            "gate_worst": float(np.mean(ranked[:GATE_WORST_K])),
            "gate_worst_min": float(ranked[0]),
            "gate_worst_family": min(fams, key=fams.get),
            "clean": clean,
            "gate_families": {k: round(v, 4) for k, v in fams.items()}}


def t2_score(model, splits, cfg, seed):
    """T2: held-out noise families built outside the Challenger DSL, on the test split.

    The noise instances depend only on `seed`, so every method evaluated at a seed sees the same
    corrupted test windows.  Returns (mean over family x SNR cells, worst cell, per-cell dict).
    """
    res = evaluate_t2(model, splits, cfg, split="test", seed=seed)
    cells = {f"{c['family']}@{c['snr_db']:g}dB": float(c["balanced_accuracy"]) for c in res["cells"]}
    return float(np.mean(list(cells.values()))), float(min(cells.values())), cells


# ================================ structure search =================================================

def make_llm(yaml_path: Path):
    """Chat-completions client for the structure search, with retries.

    The system prompt is the YAML's `system_message` plus `SEARCH_ADDENDUM`, which allows
    intermediate generations to be temporarily worse than the anchor.
    """
    conf = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    llm_conf = conf["llm"]
    api_key = os.path.expandvars(str(llm_conf["api_key"]))
    if "${" in api_key:
        raise RuntimeError(f"{yaml_path}: set the environment variable referenced by llm.api_key")
    from openai import OpenAI
    client = OpenAI(base_url=llm_conf["api_base"], api_key=api_key,
                    timeout=float(llm_conf.get("timeout", 240)))
    model = llm_conf["models"][0]["name"]
    system = conf["prompt"]["system_message"] + SEARCH_ADDENDUM
    # GLM reasoning tokens share the max_tokens budget and would truncate a full-network rewrite.
    extra = {"thinking": {"type": "disabled"}} if model.lower().startswith("glm") else None

    def call(user_msg: str) -> str:
        last = None
        for attempt in range(8):
            try:
                kwargs = {"extra_body": extra} if extra else {}
                r = client.chat.completions.create(
                    model=model, temperature=float(llm_conf.get("temperature", 0.7)),
                    max_tokens=int(llm_conf.get("max_tokens", 16384)),
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user_msg}], **kwargs)
                return r.choices[0].message.content or ""
            except Exception as e:                                  # noqa: BLE001
                last = e
                time.sleep(min(10 * (1.8 ** attempt), 60))
        raise LLMConnectionError(f"LLM failed after retries: {last}")

    return call, {"model": model, "system": system}


def structure_search(rnd, arm_dir, anchor_src, cfg, llm, tag, resume=False):
    """One search window of `solver_proposals_per_round` proposals against the frozen C(t).

    The lineage is rebuilt every round: the anchor is re-screened under the new curriculum and no
    screening score from an older curriculum is carried over.  8-epoch screening scores only steer
    breeding; deployment is decided by the confirm trainings and the gate.
    """
    out_dir = ensure_dir(arm_dir / f"round{rnd}" / "search")
    ensure_dir(out_dir / "candidates")
    gmax = int(cfg["exp"]["solver_gmax"])
    patience = int(cfg["exp"]["solver_patience"])
    n_prop = int(cfg["exp"]["solver_proposals_per_round"])

    import evaluator_solver_family as evsolver
    lin = Lineage(out_dir)
    rng = random.Random(int(cfg["project"]["seed"]) * 9176 + 100 * rnd + 2)

    anchor_path = out_dir / "candidates" / "anchor.py"
    anchor_path.write_text(anchor_src, encoding="utf-8")
    if not lin.nodes:
        res = evsolver.evaluate(str(anchor_path))
        res = res.metrics if hasattr(res, "metrics") else res
        lin.add({"id": "anchor", "parent": None, "g": 0, "d": 0,
                 "reward": float(res.get("combined_score", 0.0)), "valid": bool(res.get("valid")),
                 "params": int(res.get("params", 0) or 0), "n_conv": int(res.get("n_conv", 0) or 0),
                 "cell": None, "file": str(anchor_path)})
        print(f"[{tag}] anchor re-screened on C({rnd}): {lin.node('anchor')['reward']:.4f}", flush=True)
    anchor = lin.node("anchor")

    archive: dict = {}                                  # (log10 params, #conv) cell -> node id
    for n in lin.valid_nodes():
        c = n["cell"]
        if c is not None and (c not in archive or n["reward"] > lin.node(archive[c])["reward"]):
            archive[c] = n["id"]

    done = len([n for n in lin.nodes if n["id"] != "anchor"])
    rng_path = out_dir / "rng.pt"
    if resume:
        restore_rng(rng_path, rng, done)
    elif done:
        raise ResumeError("existing structure-search lineage requires --resume")
    while done < n_prop:
        decision = pick_parent(lin, rng, gmax, archive, patience=patience)
        pid = decision.parent_id
        parent = lin.node(pid)
        nid = f"p{done + 1:03d}"
        fp = out_dir / "candidates" / f"{nid}.py"
        err = None

        parent_src = anchor_src if pid == "anchor" else Path(parent["file"]).read_text(encoding="utf-8")
        hist = sorted(lin.valid_nodes(), key=lambda n: -n["reward"])[:4]
        hist_txt = "\n".join(f"  {h['id']}: screen={h['reward']:.4f} g={h['g']} d_fp={h['d']}"
                             for h in hist) or "  (none yet)"
        chain_txt = ""
        if pid != "anchor":
            ch = lin.chain(pid)
            chain_txt = ("\nLINEAGE CHAIN root->parent (screen scores): "
                         + " -> ".join(f"{c['id']}({c['reward']:.4f})" for c in ch)
                         + f"\nThis child will be generation g={parent['g'] + 1} (window g_max={gmax}).")
        user = (f"ANCHOR screen OOD-val (8ep, reference): {anchor['reward']:.4f}\n"
                f"BEST PREVIOUS PROPOSALS THIS WINDOW:\n{hist_txt}{chain_txt}\n\n"
                f"PARENT program (screen={parent['reward']:.4f}, params={parent['params']:,}):\n"
                f"```python\n{parent_src}\n```\n\n"
                "Produce the NEXT variant as exactly one fenced Python code block "
                "defining build_solver(n_classes, length).")
        code = None
        for outer in range(3):      # an outage retries the same slot without charging the budget
            try:
                raw = llm(user)
                (out_dir / "candidates" / f"{nid}.llm.txt").write_text(raw, encoding="utf-8")
                code = extract_code(raw)
                if code is None:
                    err = "no build_solver code block in LLM response"      # charged
                break
            except LLMConnectionError as e:
                print(f"[{tag}] {nid} endpoint outage (retry {outer + 1}/3, not charged): "
                      f"{str(e)[:120]}", flush=True)
                time.sleep(60)
        fp.write_text(code if code is not None else "", encoding="utf-8")
        if code is None and err is None:
            err = "llm connection exhausted after outer retries"

        if err is None:
            res = evsolver.evaluate(str(fp))
            res = res.metrics if hasattr(res, "metrics") else res
            valid = bool(res.get("valid"))
            reward = float(res.get("combined_score", 0.0))
            params = int(res.get("params", 0) or 0)
            n_conv = int(res.get("n_conv", 0) or 0)
            err = res.get("error")
        else:
            valid, reward, params, n_conv = False, 0.0, 0, 0

        d_val, fpd = 0, None
        if valid:
            fpd = fp_distance(fp.read_text(encoding="utf-8"), anchor_src)
            d_val = fpd["d_fp"]
        cell = (round(float(np.log10(max(params, 1))), 1), n_conv) if valid else None
        rec = {"id": nid, "parent": pid, "g": parent["g"] + 1, "d": d_val,
               "reward": reward, "valid": valid, "params": params, "n_conv": n_conv,
               "cell": str(cell) if cell else None, "file": str(fp), "fp": fpd,
               "error": (str(err)[:160] if err else None),
               "sampled_parent": decision.sampled_parent_id, "parent_source": decision.source,
               "quality_best_parent": decision.best_parent_id,
               "stale_steps": decision.stale_steps, "quality_rollback": decision.rolled_back}
        lin.add(rec)
        save_rng(rng_path, rng)
        if valid and rec["cell"] and (rec["cell"] not in archive
                                      or reward > lin.node(archive[rec["cell"]])["reward"]):
            archive[rec["cell"]] = nid
        done += 1
        print(f"[{tag}] {nid} parent={pid} src={decision.source} stale={decision.stale_steps} "
              f"rollback={decision.rolled_back} g={rec['g']} d={d_val} "
              f"{'ok reward=%.4f' % reward if valid else 'INVALID ' + str(err)[:60]}", flush=True)

    valid = lin.valid_nodes()
    stats = {"proposals": done, "n_valid": len(valid),
             "best_screen": max((n["reward"] for n in valid), default=None),
             "max_g": max((n["g"] for n in valid), default=0),
             "n_cells": len(archive), "deep_chains_g8": lin.deep_chain_count(8),
             "anchor_screen": anchor["reward"],
             "n_rollback": sum(1 for n in lin.nodes if n.get("quality_rollback"))}
    return lin, stats


def shortlist(lin, incumbent_src):
    """The sources retrained at `confirm_epochs`: incumbent, best screen, best in a different cell."""
    out = [{"role": "incumbent", "id": "incumbent", "src": incumbent_src,
            "screen": None, "cell": None}]
    valid = sorted(lin.valid_nodes(), key=lambda n: (-n["reward"], n["id"]))
    if not valid:
        return out
    top = valid[0]
    out.append({"role": "top_score", "id": top["id"],
                "src": Path(top["file"]).read_text(encoding="utf-8"),
                "screen": top["reward"], "cell": top["cell"]})
    for n in valid[1:]:
        if n["cell"] is not None and n["cell"] != top["cell"]:
            out.append({"role": "novel_cell", "id": n["id"],
                        "src": Path(n["file"]).read_text(encoding="utf-8"),
                        "screen": n["reward"], "cell": n["cell"]})
            break
    return out


# ================================ one arm: the macro-round loop ====================================

def run_arm(arm, cfg, splits, probe, out_root, llm, resume=False):
    seed = int(cfg["project"]["seed"])
    n_rounds = int(cfg["exp"]["rounds"])
    arm_dir = ensure_dir(out_root / arm)
    gen_state = ensure_dir(arm_dir / "_gen_state")
    iters_c = int(cfg["exp"]["challenger_oe_iters"])
    ep_boot = int(cfg["exp"]["boot_epochs"])
    ep_cont = int(cfg["exp"]["continue_epochs"])
    ep_conf = int(cfg["exp"]["confirm_epochs"])
    cap = int(cfg["exp"]["archive_materialize_cap"])
    mix = float(cfg["exp"]["curriculum_mix_random_frac"])
    patience_rounds = int(cfg["exp"]["rollback_patience_rounds"])
    gates = cfg["exp"]["promotion"]
    spec = arm_spec(arm, cfg)
    source = spec["curriculum"]
    random_curriculum = source == "random"
    rand_pool = cap if random_curriculum else 0      # as many distinct programs as the archive reuses
    replay_from = None
    if source == "replay":
        replay_from = out_root / spec["donor"]
        if not replay_from.is_dir():
            raise FileNotFoundError(f"arm {arm!r} replays {spec['donor']!r}, but {replay_from} "
                                    f"does not exist; run the donor arm first")
    anchor_src = SEED_SOLVER.read_text(encoding="utf-8")

    def curriculum_for(rnd, archive, mix_random_frac):
        if replay_from is not None:
            return replay_curriculum(rnd, arm_dir, replay_from)
        return freeze_curriculum(rnd, arm_dir, splits, archive, cfg, seed,
                                 mix_random_frac=mix_random_frac, archive_cap=cap,
                                 rand_pool=rand_pool)

    archive = GenMapElites()
    state = open_progress(arm_dir, arm, seed, cfg["exp"], resume)
    if state is None:
        # ---- S0: the anchor, `boot_epochs` on a random-DSL curriculum ----
        print(f"[{arm}] S0: anchor, {ep_boot}ep on the random-DSL base curriculum", flush=True)
        cur0, sd0, meta0 = curriculum_for(0, archive, 1.0)
        set_search_env(sd0, cfg)
        champ_src = anchor_src
        accum, boot_ood = train_official(champ_src, ep_boot, seed + 20000, cur0, cfg)
        g0 = gate_score(champ_src, accum, splits, cfg, seed)
        print(f"[{arm}] S0: gate_mean={g0['gate_mean']:.4f} worst={g0['gate_worst']:.4f} "
              f"clean={g0['clean']:.4f} (screen {boot_ood:.4f})", flush=True)
        _save_champion(arm_dir / "round0", champ_src, accum,
                       {"role": "S0", "epochs": ep_boot, "gate": g0, "curriculum": meta0})
        write_json(arm_dir / "round0" / "archive.json", to_jsonable(archive.cells))
        state = {
            "version": 1, "arm": arm, "seed": seed, "exp_cfg": cfg["exp"],
            "phase": "bootstrap_done", "round": 0, "champ_src": champ_src,
            "accum": accum, "epochs_accum": ep_boot, "archive_cells": {},
            "search_tip": champ_src, "stale_rounds": 0, "rollback_count": 0,
            "next_search_at": 1, "stale_searches": 0, "searches_done": 0, "lock_reason": None,
            "cost_epochs": float(ep_boot), "llm_calls": 0, "round_log": [], "current": {},
        }
        save_progress(arm_dir, state)

    champ_src = state["champ_src"]
    accum = state["accum"]
    epochs_accum = int(state["epochs_accum"])
    archive.cells = state["archive_cells"]
    search_tip = state["search_tip"]
    stale_rounds = int(state["stale_rounds"])
    rollback_count = int(state["rollback_count"])
    next_search_at = int(state["next_search_at"])
    stale_searches = int(state["stale_searches"])
    searches_done = int(state["searches_done"])
    lock_reason = state["lock_reason"]
    cost_epochs = float(state["cost_epochs"])
    llm_calls = int(state["llm_calls"])
    round_log = state["round_log"]
    if state["phase"] == "complete":
        return round_log

    for t in range(int(state["round"]) + 1, n_rounds + 1):
        t_round = time.time()
        current = state.setdefault("current", {})
        if current and int(current.get("round", -1)) != t:
            raise ResumeError(f"resume round mismatch: expected {t}, found {current.get('round')}")
        current.setdefault("round", t)

        # ---- 0) structure action for this round ----
        routed_from = next_search_at
        do_search = route_structure(t, next_search_at, spec)
        if not do_search:
            print(f"[{arm}] round {t}: structural branch closed (policy={spec['structure']})",
                  flush=True)

        # ---- 1) Challenger against the current Solver ----
        if source != "evolve" and "challenger" not in current:
            current["challenger"] = {"source": source, "n_proposals": 0, "new_cells": 0,
                                     "archive_rescored": 0}
            state["phase"] = "challenger_done"
            save_progress(arm_dir, state)
        if "challenger" not in current:
            freeze_solver(build_model(champ_src, accum), champ_src, probe, cfg, gen_state,
                          seed, int(cfg["data"]["n_classes"]), int(cfg["data"]["window_len"]))
            n_rescored = rescore_archive(archive, gen_state)
            recs = challenger_evolve(t, arm_dir, gen_state, iters_c)
            n_new = sum(int(archive.insert(r["program_src"], float(r["reward"]), r["feats"]))
                        for r in recs)
            margins = [float(r["margin"]) for r in recs if "margin" in r]
            current["challenger"] = {
                "source": source, "n_proposals": len(recs), "new_cells": n_new,
                "archive_rescored": n_rescored,
                "margin_min": min(margins) if margins else None,
                "margin_mean": float(np.mean(margins)) if margins else None,
                "margin_max": max(margins) if margins else None,
            }
            state["archive_cells"] = archive.cells
            state["phase"] = "challenger_done"
            save_progress(arm_dir, state)
        chall = current["challenger"]

        # ---- 2) freeze C(t) ----
        if "curriculum_dir" not in current:
            cur, state_dir, cmeta = curriculum_for(t, archive, 1.0 if random_curriculum else mix)
            current["curriculum_dir"] = str(state_dir)
            current["curriculum_meta"] = cmeta
            state["phase"] = "curriculum_done"
            save_progress(arm_dir, state)
        else:
            state_dir = Path(current["curriculum_dir"])
            cur = load_curriculum_npz(state_dir / "data.npz")
            cmeta = current["curriculum_meta"]
        set_search_env(state_dir, cfg)
        print(f"[{arm}] round {t}: C({t}) frozen ({source}) -- {cmeta['n_train_total']} train rows, "
              f"{cmeta['gens_materialized']} archive generators, sha {cmeta['data_sha256'][:12]}",
              flush=True)

        # ---- 3) continue the deployed checkpoint on C(t): this round's incumbent ----
        if "incumbent_path" not in current:
            inc_state, inc_ood = train_official(champ_src, ep_cont, seed + 30000 + t, cur, cfg,
                                                init_state=accum)
            epochs_accum += ep_cont
            inc_gate = gate_score(champ_src, inc_state, splits, cfg, seed)
            incumbent_path = arm_dir / f"round{t}" / "resume_incumbent.pt"
            _save_state(incumbent_path, champ_src, inc_state)
            current.update({"incumbent_path": str(incumbent_path),
                            "incumbent_ood": inc_ood, "incumbent_gate": inc_gate})
            state["epochs_accum"] = epochs_accum
            state["phase"] = "incumbent_done"
            save_progress(arm_dir, state)
        else:
            inc_state = torch.load(current["incumbent_path"], map_location="cpu")["state_dict"]
            inc_ood = current["incumbent_ood"]
            inc_gate = current["incumbent_gate"]
            epochs_accum = int(state["epochs_accum"])
        print(f"[{arm}] round {t}: incumbent +{ep_cont}ep (total {epochs_accum}) -> "
              f"gate_mean={inc_gate['gate_mean']:.4f} worst={inc_gate['gate_worst']:.4f} "
              f"clean={inc_gate['clean']:.4f}", flush=True)

        promo = {"searched": False, "promoted": False}
        search_stats, confirm, audit = {}, [], None
        if do_search:
            # ---- 4) structure search, anchored on the search tip ----
            if "search" not in current:
                lin, search_stats = structure_search(t, arm_dir, search_tip, cfg, llm,
                                                     tag=f"{arm} r{t}", resume=resume)
                search_stats["anchor_is_champion"] = bool(search_tip.strip() == champ_src.strip())
                search_stats["stale_rounds_at_entry"] = stale_rounds
                assert _sha256(state_dir / "data.npz") == cmeta["data_sha256"], \
                    "curriculum drifted during the structure search"
                current["search"] = search_stats
                current["candidates"] = shortlist(lin, champ_src)
                current["confirm"] = {}
                state["phase"] = "search_done"
                save_progress(arm_dir, state)
            search_stats = current["search"]
            cand = current["candidates"]

            # ---- 5) confirm: shortlisted sources from scratch, same C(t), same seed ----
            # A candidate that screens well can still be undeployable (lazy parameters); it is
            # recorded and dropped.  The incumbent is never dropped.
            confirm_seed = seed + 40000 + t
            confirmed = current.setdefault("confirm", {})
            rejected = current.setdefault("confirm_rejected", {})
            for c in cand:
                if c["role"] in confirmed or c["role"] in rejected:
                    continue
                try:
                    st, ood = train_official(c["src"], ep_conf, confirm_seed, cur, cfg)
                    g = gate_score(c["src"], st, splits, cfg, seed)
                    rec = {**{k: v for k, v in c.items() if k != "src"}, "confirm_ood": ood, **g,
                           "params": count_params(c["src"], st),
                           "source_sha256": hashlib.sha256(c["src"].encode("utf-8")).hexdigest()}
                    _save_state(arm_dir / f"round{t}" / "confirm" / f"{c['role']}.pt", c["src"], st)
                except UndeployableCandidate as e:
                    if c["role"] == "incumbent":
                        raise
                    rejected[c["role"]] = {
                        **{k: v for k, v in c.items() if k != "src"},
                        "error": f"{type(e).__name__}: {str(e)[:200]}",
                        "source_sha256": hashlib.sha256(c["src"].encode("utf-8")).hexdigest()}
                    state["phase"] = "confirming"
                    save_progress(arm_dir, state)
                    print(f"[{arm}] round {t}: confirm {c['role']:<11} ({c['id']}) rejected -- "
                          f"{str(e)[:120]}", flush=True)
                    torch.cuda.empty_cache()
                    continue
                confirmed[c["role"]] = rec
                state["phase"] = "confirming"
                save_progress(arm_dir, state)
                print(f"[{arm}] round {t}: confirm {c['role']:<11} ({c['id']}) {ep_conf}ep -> "
                      f"gate_mean={g['gate_mean']:.4f} worst={g['gate_worst']:.4f} "
                      f"clean={g['clean']:.4f} params={rec['params']:,}", flush=True)
                del st
                torch.cuda.empty_cache()
            confirm = [confirmed[c["role"]] for c in cand if c["role"] in confirmed]

            promo = decide_promotion(confirm, gates, ep_conf, epochs_accum)
            promo["searched"] = True
            searches_done += 1

            # ---- 6) rebuild the selected source at the accumulated budget, then confirm deployment ----
            rebuilt = None
            if promo["promoted"]:
                win = promo["winner"]
                src_new = next(c["src"] for c in cand if c["role"] == win["role"])
                if epochs_accum == ep_conf:
                    st = torch.load(arm_dir / f"round{t}" / "confirm" / f"{win['role']}.pt",
                                    map_location="cpu")["state_dict"]
                else:
                    st, spent = rebuild_on_accumulated_schedule(
                        src_new, arm_dir, t, ep_boot, ep_cont, seed + 60000 + t, cfg)
                    if spent != epochs_accum:
                        raise RuntimeError(f"rebuild spent {spent}ep but the accumulated budget "
                                           f"is {epochs_accum}ep")
                    _save_state(arm_dir / f"round{t}" / "deploy_rebuilt.pt", src_new, st)
                    print(f"[{arm}] round {t}: selected source rebuilt over C(0..{t}) "
                          f"({spent}ep)", flush=True)
                # The rebuilt source must match or beat the checkpoint it would replace, both at
                # `epochs_accum` epochs.
                deploy_audit = None
                if epochs_accum != ep_conf:
                    g_new = gate_score(src_new, st, splits, cfg, seed)
                    deployed = g_new["gate_mean"] >= inc_gate["gate_mean"]
                    deploy_audit = {"deployed": deployed, "rebuilt_gate_mean": g_new["gate_mean"],
                                    "accumulated_gate_mean": inc_gate["gate_mean"]}
                    if not deployed:
                        promo["promoted"] = False
                        promo["reason"] = ("selected but not deployed: the rebuilt source lost to "
                                           "the accumulated checkpoint at matched budget")
                        print(f"[{arm}] round {t}: selected {win['role']} but not deployed -- "
                              f"rebuilt {g_new['gate_mean']:.4f} < accumulated "
                              f"{inc_gate['gate_mean']:.4f}", flush=True)
                promo["deploy_confirmation"] = deploy_audit
                rebuilt = (src_new, st)

            if promo["promoted"]:
                champ_src, accum = rebuilt
                search_tip, stale_rounds = champ_src, 0
                print(f"[{arm}] round {t}: PROMOTED {promo['winner']['role']} "
                      f"({promo['winner']['id']}) at {epochs_accum}ep", flush=True)
            else:
                accum = inc_state
                stale_rounds += 1
                if patience_rounds > 0 and stale_rounds >= patience_rounds:
                    search_tip, stale_rounds = champ_src, 0
                    rollback_count += 1
                    tip_note = f"search tip rolled back to the champion (patience {patience_rounds})"
                else:
                    # Exploration may drift: the next window starts from the best confirmed source,
                    # whether or not it passed the guardrails.
                    tip_rec = max(confirm, key=lambda c: c["gate_mean"])
                    search_tip = next(c["src"] for c in cand if c["role"] == tip_rec["role"])
                    tip_note = (f"search tip -> {tip_rec['role']} ({tip_rec['id']}, "
                                f"gate_mean={tip_rec['gate_mean']:.4f})")
                print(f"[{arm}] round {t}: no promotion ({promo['reason']}); {tip_note}", flush=True)

            stale_searches, next_search_at = advance_router(t, promo["promoted"], stale_searches, spec)
            next_search_at, lock_reason = apply_search_budget(next_search_at, searches_done, spec)

            # ---- 7) switch audit: the structural branch closes after this round ----
            # Before freezing on the deployed structure, compare it with the anchor at the confirm
            # budget on C(t); if it no longer holds up, roll back to the anchor (rebuilt at the
            # accumulated budget) and freeze there.
            if next_search_at >= NEVER_SEARCH and bool(cfg["exp"]["switch_audit"]):
                print(f"[{arm}] round {t}: structural branch closes ({lock_reason})", flush=True)
                if champ_src.strip() == anchor_src.strip():
                    audit = {"performed": False, "rolled_back_to_anchor": False,
                             "note": "the champion is the anchor"}
                else:
                    if "switch_audit_anchor" not in confirmed:
                        st_a, ood_a = train_official(anchor_src, ep_conf, confirm_seed, cur, cfg)
                        g_a = gate_score(anchor_src, st_a, splits, cfg, seed)
                        _save_state(arm_dir / f"round{t}" / "confirm" / "switch_audit_anchor.pt",
                                    anchor_src, st_a)
                        confirmed["switch_audit_anchor"] = {"role": "switch_audit_anchor",
                                                            "confirm_ood": ood_a, **g_a}
                        save_progress(arm_dir, state)
                        del st_a
                        torch.cuda.empty_cache()
                    keys = ("gate_mean", "gate_worst", "clean", "gate_families")
                    champ_rec = next(c for c in confirm if c["role"] == "incumbent")
                    dec = decide_anchor_fallback({k: champ_rec[k] for k in keys},
                                                 {k: confirmed["switch_audit_anchor"][k] for k in keys},
                                                 gates)
                    audit = {"performed": True, "verdict": dec["verdict"], "reason": dec["reason"],
                             "conditions": dec["conditions"],
                             "champion_gate_mean": champ_rec["gate_mean"],
                             "anchor_gate_mean": confirmed["switch_audit_anchor"]["gate_mean"],
                             "audit_epochs": float(ep_conf),
                             "rolled_back_to_anchor": bool(dec["anchor_preferred"])}
                    if dec["anchor_preferred"]:
                        if epochs_accum == ep_conf:
                            st = torch.load(arm_dir / f"round{t}" / "confirm" /
                                            "switch_audit_anchor.pt", map_location="cpu")["state_dict"]
                        else:
                            st, spent = rebuild_on_accumulated_schedule(
                                anchor_src, arm_dir, t, ep_boot, ep_cont, seed + 61000 + t, cfg)
                            if spent != epochs_accum:
                                raise RuntimeError(f"rebuild spent {spent}ep but the accumulated "
                                                   f"budget is {epochs_accum}ep")
                            _save_state(arm_dir / f"round{t}" / "switch_rollback_rebuilt.pt",
                                        anchor_src, st)
                            audit["audit_epochs"] += float(epochs_accum)
                        champ_src, accum, search_tip = anchor_src, st, anchor_src
                        print(f"[{arm}] round {t}: switch audit -- evolved structure lost to the "
                              f"anchor ({dec['reason']}); rolled back to the anchor", flush=True)
                    else:
                        print(f"[{arm}] round {t}: switch audit -- evolved structure holds up "
                              f"({audit['champion_gate_mean']:.4f} vs "
                              f"{audit['anchor_gate_mean']:.4f})", flush=True)
        else:
            accum = inc_state

        # ---- cost ledger (anchor-epoch equivalents; LLM calls counted separately) ----
        # A rebuild is charged whenever it ran, including one refused at the deployment check.
        rebuilt_ran = promo.get("promoted") or (promo.get("deploy_confirmation") is not None)
        rebuild_ep = epochs_accum if (rebuilt_ran and epochs_accum != ep_conf) else 0
        round_cost = float(ep_cont)
        round_llm = int(iters_c) if source == "evolve" else 0
        if promo.get("searched"):
            round_cost += search_round_cost(search_stats, len(confirm), cfg, rebuild_ep)
            round_llm += int(search_stats.get("proposals", 0))
        if audit and audit.get("performed"):
            round_cost += float(audit["audit_epochs"])
        cost_epochs += round_cost
        llm_calls += round_llm

        _save_champion(arm_dir / f"round{t}", champ_src, accum,
                       {"role": "champion", "epochs": epochs_accum,
                        "gate": gate_score(champ_src, accum, splits, cfg, seed)})
        write_json(arm_dir / f"round{t}" / "archive.json", to_jsonable(archive.cells))
        row = {"arm": arm, "round": t, "wall_s": round(time.time() - t_round),
               "epochs_accum": epochs_accum, "challenger": chall, "curriculum": cmeta,
               "incumbent": {"ood": inc_ood, **inc_gate}, "search": search_stats,
               "confirm": confirm,
               "confirm_rejected": list((current.get("confirm_rejected") or {}).values()),
               "promotion": promo,
               "champion_params": count_params(champ_src, accum),
               "champion_is_anchor": champ_src.strip() == anchor_src.strip(),
               "rollback": {"stale_rounds": stale_rounds, "rollback_count": rollback_count,
                            "patience_rounds": patience_rounds,
                            "tip_is_champion": bool(search_tip.strip() == champ_src.strip())},
               "structure": {"action": "search" if do_search else "id", "policy": spec,
                             "routed_from_next_search_at": routed_from,
                             "stale_searches": stale_searches, "next_search_at": next_search_at,
                             "branch_closed": bool(next_search_at >= NEVER_SEARCH),
                             "searches_done": searches_done, "lock_reason": lock_reason,
                             "switch_audit": audit},
               "cost": {"round_epochs": round_cost, "cost_epochs": cost_epochs,
                        "round_llm_calls": round_llm, "llm_calls": llm_calls,
                        "search_epochs_this_round": round_cost - ep_cont}}
        round_log.append(row)
        write_json(arm_dir / "round_log.json", round_log)
        # T2 is sealed during `run`: no T2 quantity may appear in the round log.
        assert not any("t2" in k.lower() for k in _flat_keys(row)), "T2 leaked into the round log"
        assert_frozen_round_is_contained(row, ep_cont, iters_c,
                                         anchor_only=(spec["structure"] == "never"),
                                         no_challenger=(source != "evolve"))

        state.update({"phase": "round_done", "round": t, "champ_src": champ_src,
                      "accum": accum, "epochs_accum": epochs_accum,
                      "archive_cells": archive.cells, "search_tip": search_tip,
                      "stale_rounds": stale_rounds, "rollback_count": rollback_count,
                      "next_search_at": next_search_at, "stale_searches": stale_searches,
                      "searches_done": searches_done, "lock_reason": lock_reason,
                      "cost_epochs": cost_epochs, "llm_calls": llm_calls,
                      "round_log": round_log, "current": {}})
        save_progress(arm_dir, state)

    state["phase"] = "complete"
    save_progress(arm_dir, state)
    return round_log


# ================================ decisions ========================================================

def decide_promotion(confirm, gates, ep_conf, epochs_accum):
    """Promote the best mutation only if every condition holds against the incumbent source.

    All sources are retrained from scratch for `ep_conf` epochs on the same C(t) with the same
    seed, so the comparison is between structures at an equal training budget:

      1. gate_mean  > incumbent + delta_promote
      2. clean     >= incumbent - epsilon_clean
      3. gate_worst >= incumbent - epsilon_worst
      4. no family drops by more than epsilon_family

    Ties between mutations are broken by gate mean, then two-worst mean, then fewer parameters,
    then the smaller source hash.
    """
    inc = next((c for c in confirm if c["role"] == "incumbent"), None)
    muts = [c for c in confirm if c["role"] != "incumbent"]
    out = {"promoted": False, "reason": None, "winner": None,
           "epochs_confirm": int(ep_conf), "epochs_accum_before": int(epochs_accum),
           "delta_promote": float(gates["delta_promote"]),
           "epsilon_clean": float(gates["epsilon_clean"]),
           "epsilon_worst": float(gates["epsilon_worst"]),
           "epsilon_family": float(gates["epsilon_family"]),
           "screen_top1_overturned": None, "conditions": []}
    if inc is None or not muts:
        out["reason"] = "no valid mutation in the shortlist"
        return out

    # Did the best 8-epoch screen survive full-fidelity retraining?  None on an exact tie.
    ranked = sorted(confirm, key=lambda c: -c["gate_mean"])
    top_screen = max(muts, key=lambda c: (c["screen"] if c["screen"] is not None else -1e9))
    out["screen_top1_overturned"] = (
        None if ranked[0]["gate_mean"] == top_screen["gate_mean"]
        else bool(ranked[0]["role"] != top_screen["role"]))
    out["confirm_ranking"] = [c["role"] for c in ranked]

    best = min(muts, key=lambda c: (-c["gate_mean"], -c["gate_worst"], c.get("params", 0),
                                    c.get("source_sha256", "")))
    fam_drop = {f: best["gate_families"][f] - inc["gate_families"][f] for f in inc["gate_families"]}
    worst_fam, worst_drop = min(fam_drop.items(), key=lambda kv: kv[1])
    conds = [
        ("beats_incumbent_by_delta",
         best["gate_mean"] > inc["gate_mean"] + float(gates["delta_promote"]),
         {"mutation": best["gate_mean"], "incumbent": inc["gate_mean"]}),
        ("clean_not_degraded",
         best["clean"] >= inc["clean"] - float(gates["epsilon_clean"]),
         {"mutation": best["clean"], "incumbent": inc["clean"]}),
        ("worst_not_degraded",
         best["gate_worst"] >= inc["gate_worst"] - float(gates["epsilon_worst"]),
         {"mutation": best["gate_worst"], "incumbent": inc["gate_worst"]}),
        ("no_family_regressed",
         worst_drop >= -float(gates["epsilon_family"]),
         {"worst_family": worst_fam, "drop": worst_drop}),
    ]
    out["conditions"] = [{"name": n, "passed": bool(ok), **vals} for n, ok, vals in conds]
    failed = [n for n, ok, _ in conds if not ok]
    out["promoted"] = not failed
    out["reason"] = (f"all {len(conds)} conditions passed" if not failed
                     else "failed: " + ", ".join(failed))
    out["primary_margin"] = best["gate_mean"] - inc["gate_mean"] - float(gates["delta_promote"])
    out["winner"] = ({k: best[k] for k in ("role", "id", "screen", "cell", "gate_mean",
                                           "gate_worst", "clean")} if not failed else None)
    out["candidate_considered"] = {k: best[k] for k in ("role", "id", "gate_mean", "gate_worst",
                                                        "clean")}
    return out


def decide_anchor_fallback(champ, anchor, gates, epsilon_anchor=EPSILON_ANCHOR):
    """Switch audit: does the deployed structure still hold up against the anchor?

    Both are trained from scratch at the confirm budget on the same C(t) with the same seed.  The
    champion is kept if it is not below the anchor and passes the same guardrails; otherwise the
    arm rolls back to the anchor.
    """
    fam_drop = {f: champ["gate_families"][f] - anchor["gate_families"][f]
                for f in anchor["gate_families"]}
    worst_fam, worst_drop = min(fam_drop.items(), key=lambda kv: kv[1])
    conds = [
        ("champion_not_below_anchor",
         champ["gate_mean"] >= anchor["gate_mean"] - float(epsilon_anchor),
         {"champion": champ["gate_mean"], "anchor": anchor["gate_mean"]}),
        ("worst_not_degraded",
         champ["gate_worst"] >= anchor["gate_worst"] - float(gates["epsilon_worst"]),
         {"champion": champ["gate_worst"], "anchor": anchor["gate_worst"]}),
        ("clean_not_degraded",
         champ["clean"] >= anchor["clean"] - float(gates["epsilon_clean"]),
         {"champion": champ["clean"], "anchor": anchor["clean"]}),
        ("no_family_regressed",
         worst_drop >= -float(gates["epsilon_family"]),
         {"worst_family": worst_fam, "drop": worst_drop}),
    ]
    failed = [n for n, ok, _ in conds if not ok]
    return {"verdict": "structure_holds_up" if not failed else "anchor_preferred",
            "anchor_preferred": bool(failed),
            "reason": ("all conditions passed" if not failed else "failed: " + ", ".join(failed)),
            "conditions": [{"name": n, "passed": bool(ok), **v} for n, ok, v in conds]}


def assert_frozen_round_is_contained(row, ep_cont, iters_c, anchor_only=False, no_challenger=False):
    """A round with the structural branch closed must cost exactly one weight continuation.

    This makes "a frozen DuoEvo round is an Evolve-NoStruct round" a checked property: no search,
    no confirm, no promotion, `continue_epochs` charged, and the Challenger budget only when the
    arm evolves its own curriculum.
    """
    if row.get("structure", {}).get("action") != "id":
        return
    bad = []
    if row.get("search"):
        bad.append(f"search block is non-empty: {sorted(row['search'])[:4]}")
    if row.get("confirm"):
        bad.append(f"{len(row['confirm'])} confirm trainings on a frozen round")
    if row["promotion"].get("searched"):
        bad.append("promotion.searched is True")
    if row["promotion"].get("promoted"):
        bad.append("promotion.promoted is True")
    if anchor_only and not row.get("champion_is_anchor", True):
        bad.append(f"champion is no longer the anchor ({row.get('champion_params')} params)")
    cost = row.get("cost", {})
    if float(cost.get("round_epochs", -1)) != float(ep_cont):
        bad.append(f"round_epochs {cost.get('round_epochs')} != continue_epochs {ep_cont}")
    if float(cost.get("search_epochs_this_round", -1)) != 0.0:
        bad.append(f"search_epochs_this_round {cost.get('search_epochs_this_round')} != 0")
    want_llm = 0 if no_challenger else int(iters_c)
    if int(cost.get("round_llm_calls", -1)) != want_llm:
        bad.append(f"round_llm_calls {cost.get('round_llm_calls')} != {want_llm}")
    if bad:
        raise AssertionError(f"round {row.get('round')} of {row.get('arm')} has the structural "
                             f"branch closed but is not a frozen round: " + "; ".join(bad))


def _flat_keys(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{prefix}{k}"
            yield from _flat_keys(v, f"{prefix}{k}.")
    elif isinstance(obj, list):
        for v in obj:
            yield from _flat_keys(v, prefix)


def _save_state(path: Path, src: str, state):
    ensure_dir(Path(path).parent)
    torch.save({"program_src": src, "state_dict": state}, path)


def _save_champion(round_dir: Path, src: str, state, meta: dict):
    ensure_dir(round_dir)
    (round_dir / "champion.py").write_text(src, encoding="utf-8")
    _save_state(round_dir / "champion.pt", src, state)
    write_json(round_dir / "champion.json", to_jsonable(meta))


# ================================ commands =========================================================

def load_everything(args):
    cfg = load_config(args.config, args.seed)
    seed = int(cfg["project"]["seed"])
    seed_everything(seed)
    splits = load_splits(cfg)
    cfg["data"]["n_classes"] = int(splits.meta["n_classes"])
    global _GLOBAL_CFG
    _GLOBAL_CFG = cfg
    probe = carve_edge_probe(splits, n_per_class=int(cfg["noise"]["vedge_per_class"]), seed=seed)
    out_root = ensure_dir(Path(args.out) / f"{cfg['data']['dataset']}_s{seed}")
    return cfg, splits, probe, out_root


def cmd_run(args):
    cfg, splits, probe, out_root = load_everything(args)
    arms = [a.strip() for a in args.arms.split(",")] if args.arms else list(cfg["exp"]["arms"])
    for a in arms:
        arm_spec(a, cfg)                                # fail on a bad arm before any training
    write_json(out_root / "config_used.json", to_jsonable(cfg))
    llm = None
    if any(arm_spec(a, cfg)["structure"] != "never" for a in arms):
        yaml_path = CONFIGS / "llm" / "solver.yaml"
        llm, meta = make_llm(yaml_path)
        (out_root / "solver_prompt_system.txt").write_text(meta["system"], encoding="utf-8")
        print(f"structure-search LLM: {meta['model']}", flush=True)

    summary_path = out_root / "summary.json"
    results = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    for arm in arms:
        results[arm] = run_arm(arm, cfg, splits, probe, out_root, llm, resume=args.resume)
        write_json(summary_path, to_jsonable(results))
    print(f"\nresults -> {out_root}\nT2 is still sealed; run `evaluate` once every arm has finished.",
          flush=True)


def cmd_evaluate(args):
    """Score every saved champion checkpoint on T2 (forward passes only)."""
    cfg, splits, probe, out_root = load_everything(args)
    seed = int(cfg["project"]["seed"])
    rows = []
    for arm_dir in sorted(p for p in out_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        rounds = [p for p in arm_dir.glob("round*") if p.is_dir() and p.name[5:].isdigit()]
        for rd in sorted(rounds, key=lambda p: int(p.name[5:])):
            ckpt = rd / "champion.pt"
            if not ckpt.exists():
                continue
            payload = torch.load(ckpt, map_location="cpu")
            model = build_model(payload["program_src"], payload["state_dict"])
            t2_mean, t2_worst, t2_cells = t2_score(model, splits, cfg, seed)
            meta = (json.loads((rd / "champion.json").read_text(encoding="utf-8"))
                    if (rd / "champion.json").exists() else {})
            rows.append({"arm": arm_dir.name, "round": int(rd.name[5:]),
                         "epochs_accum": meta.get("epochs"),
                         "t2_mean": t2_mean, "t2_worst_cell": t2_worst,
                         "t2_cells": {k: round(v, 4) for k, v in t2_cells.items()}})
            print(f"[T2] {arm_dir.name} {rd.name}: mean={t2_mean:.4f} worst={t2_worst:.4f}",
                  flush=True)
            del model
            torch.cuda.empty_cache()
    write_json(out_root / "t2_eval.json", rows)

    # Arms are budget-matched on `epochs_accum` (every arm reaches boot + continue * rounds whether
    # or not it promotes), so trajectories are compared at equal training epochs.
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], {})[r["epochs_accum"]] = r
    summary = {}
    for arm in sorted(by_arm):
        log_path = out_root / arm / "round_log.json"
        log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else []
        summary[arm] = {
            "trajectory": {str(e): {"t2_mean": r["t2_mean"], "t2_worst_cell": r["t2_worst_cell"]}
                           for e, r in sorted(by_arm[arm].items())},
            "search_rounds": [x["round"] for x in log if x["promotion"].get("searched")],
            "cost_epochs": log[-1]["cost"]["cost_epochs"] if log else None,
            "llm_calls": log[-1]["cost"]["llm_calls"] if log else None,
        }
    write_json(out_root / "t2_summary.json", summary)

    print(f"\n{'arm':<16} {'epochs':>7} {'T2 mean':>8} {'T2 worst':>9} {'searches':>9} {'LLM calls':>10}")
    for arm in sorted(by_arm):
        last = by_arm[arm][max(by_arm[arm])]
        s = summary[arm]
        print(f"{arm:<16} {last['epochs_accum']:>7} {last['t2_mean']:>8.4f} "
              f"{last['t2_worst_cell']:>9.4f} {len(s['search_rounds']):>9} {str(s['llm_calls']):>10}")
    print(f"\n-> {out_root / 't2_eval.json'}", flush=True)


def cmd_calibrate(args):
    """Run-to-run spread of the gate for one fixed source.

    Trains the anchor `--reps` times at `confirm_epochs` on one frozen curriculum.  With sigma_M
    and sigma_W the across-training standard deviations of `gate_mean` and `gate_worst`, the
    proposed thresholds are delta_promote = 2 sigma_M and epsilon_worst = 3 sigma_W.  Pass
    `--curriculum-dir` to reuse a curriculum: building one runs the Challenger (LLM calls), so two
    calibrations are only comparable on the same curriculum.
    """
    cfg, splits, probe, out_root = load_everything(args)
    seed = int(cfg["project"]["seed"])
    epochs = int(cfg["exp"]["confirm_epochs"])
    cap = int(cfg["exp"]["archive_materialize_cap"])
    cal_dir = ensure_dir(out_root / "_calibration")
    src = SEED_SOLVER.read_text(encoding="utf-8")

    if args.curriculum_dir:
        state_dir = Path(args.curriculum_dir)
        cur = load_curriculum_npz(state_dir / "data.npz")
        cmeta = {"reused_from": str(state_dir), "data_sha256": _sha256(state_dir / "data.npz")}
    else:
        # One Challenger round against the boot model, then the standard materialization.
        cur0, sd0, _ = freeze_curriculum(0, cal_dir, splits, GenMapElites(), cfg, seed,
                                         mix_random_frac=1.0, archive_cap=cap)
        set_search_env(sd0, cfg)
        boot, _ = train_official(src, int(cfg["exp"]["boot_epochs"]), seed + 20000, cur0, cfg)
        freeze_solver(build_model(src, boot), src, probe, cfg, ensure_dir(cal_dir / "_gen_state"),
                      seed, int(cfg["data"]["n_classes"]), int(cfg["data"]["window_len"]))
        archive = GenMapElites()
        for r in challenger_evolve(1, cal_dir, cal_dir / "_gen_state",
                                   int(cfg["exp"]["challenger_oe_iters"])):
            archive.insert(r["program_src"], float(r["reward"]), r["feats"])
        cur, state_dir, cmeta = freeze_curriculum(
            1, cal_dir, splits, archive, cfg, seed,
            mix_random_frac=float(cfg["exp"]["curriculum_mix_random_frac"]), archive_cap=cap)
        print(f"[calibrate] curriculum frozen at {state_dir}; pass --curriculum-dir {state_dir} "
              f"to reuse it", flush=True)
    set_search_env(state_dir, cfg)

    reps = []
    for i in range(int(args.reps)):
        train_seed = seed + 500000 + 1000 * i
        st, ood = train_official(src, epochs, train_seed, cur, cfg)
        g = gate_score(src, st, splits, cfg, seed)
        reps.append({"rep": i, "train_seed": train_seed, "confirm_ood": ood, **g})
        print(f"[calibrate] rep {i}: gate_mean={g['gate_mean']:.4f} worst={g['gate_worst']:.4f} "
              f"clean={g['clean']:.4f}", flush=True)
        del st
        torch.cuda.empty_cache()

    M = np.array([r["gate_mean"] for r in reps])
    W = np.array([r["gate_worst"] for r in reps])
    C = np.array([r["clean"] for r in reps])
    out = {"n_reps": len(reps), "epochs": epochs, "curriculum": cmeta, "reps": reps,
           "gate_mean": {"mean": float(M.mean()), "std": float(M.std(ddof=1))},
           "gate_worst": {"mean": float(W.mean()), "std": float(W.std(ddof=1))},
           "clean": {"mean": float(C.mean()), "std": float(C.std(ddof=1))},
           "proposed": {"delta_promote": round(2 * float(M.std(ddof=1)), 4),
                        "epsilon_worst": round(3 * float(W.std(ddof=1)), 4)}}
    write_json(out_root / "calibration.json", out)
    print(f"\n[calibrate] std(M)={M.std(ddof=1):.4f} std(W)={W.std(ddof=1):.4f} -> "
          f"delta_promote={out['proposed']['delta_promote']} "
          f"epsilon_worst={out['proposed']['epsilon_worst']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="DuoEvo co-evolution driver")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, help_text in (("run", cmd_run, "run the co-evolution arms"),
                                ("evaluate", cmd_evaluate, "score saved checkpoints on T2"),
                                ("calibrate", cmd_calibrate, "estimate the promotion thresholds")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", required=True, help="dataset config, e.g. configs/cwru.yaml")
        p.add_argument("--seed", type=int, default=11)
        p.add_argument("--out", default="runs")
        if name == "run":
            p.add_argument("--arms", default=None, help="comma-separated subset of the configured arms")
            p.add_argument("--resume", action="store_true", help="continue an interrupted run")
        if name == "calibrate":
            p.add_argument("--reps", type=int, default=5)
            p.add_argument("--curriculum-dir", default=None,
                           help="reuse a frozen curriculum directory instead of building one")
        p.set_defaults(func=fn)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    # OpenEvolve evaluates candidates in a spawned worker pool; the parent has already initialized
    # CUDA, so `fork` would fail.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
