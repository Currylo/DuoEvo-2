"""Crash-safe progress for `duoevo.py run --resume`.

Each arm keeps one `resume.pt` that is rewritten atomically after every phase of a round
(challenger, curriculum, incumbent, search, each confirm training).  A resumed run refuses to
continue if the arm, seed or experiment configuration differs from the checkpoint.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


class ResumeError(RuntimeError):
    pass


def atomic_torch_save(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def open_progress(arm_dir, arm: str, seed: int, exp_cfg: dict, resume: bool):
    """The saved state of an interrupted arm, or None for a fresh start."""
    path = Path(arm_dir) / "resume.pt"
    if not resume:
        if path.exists():
            raise ResumeError(f"resume checkpoint already exists: {path} (pass --resume)")
        return None
    if not path.exists():
        raise ResumeError(f"no resume checkpoint: {path}")

    state = torch.load(path, map_location="cpu")
    if state.get("version") != 1 or state.get("arm") != arm or state.get("seed") != seed:
        raise ResumeError("resume identity does not match this run")
    if state.get("exp_cfg") != exp_cfg:
        raise ResumeError("resume configuration does not match")
    return state


def save_progress(arm_dir, state: dict) -> None:
    atomic_torch_save(Path(arm_dir) / "resume.pt", state)


def save_rng(path, rng) -> None:
    atomic_torch_save(path, rng.getstate())


def restore_rng(path, rng, completed: int) -> None:
    path = Path(path)
    if completed == 0:
        return
    if not path.exists():
        raise ResumeError(f"missing structure-search RNG checkpoint: {path}")
    rng.setstate(torch.load(path, map_location="cpu"))
