"""Dataset loaders: PU, CWRU and JNU as windowed train / dev / test / cross splits.

All signals are brought to `data.sample_rate_hz` (12 kHz), z-normalized per source record and cut
into `window_len` windows with the dataset's `stride`.  `cross` is a held-out operating condition
and is never used for training, selection or the reported test.

  PU     each operating condition is recorded as 20 repetitions; whole repetitions are assigned
         to train / dev / test (`train_reps`, `dev_reps`, `test_reps`).
  CWRU   one record per fault and load; each record is split 60/20/20 in time order.
  JNU    one record per fault and speed; each record is split 60/20/20 in time order.

Windows overlap when stride < window_len, so a time-ordered split drops the
`window_len // stride - 1` windows after each boundary; no dev or test window shares a raw sample
with a training window.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

SPLITS = ("train", "dev", "test", "cross")


@dataclass
class BearingSplits:
    train_X: np.ndarray
    train_y: np.ndarray
    dev_X: np.ndarray
    dev_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    cross_X: np.ndarray
    cross_y: np.ndarray
    meta: dict


def load_splits(cfg: dict) -> BearingSplits:
    name = cfg["data"]["dataset"]
    loaders = {"pu": load_pu, "cwru": load_cwru, "jnu": load_jnu}
    if name not in loaders:
        raise ValueError(f"unknown dataset {name!r}; expected one of {sorted(loaders)}")
    return loaders[name](cfg)


# ---------------- shared helpers ----------------

def _resample(sig: np.ndarray, raw_hz: int, target_hz: int) -> np.ndarray:
    if int(raw_hz) == int(target_hz):
        return sig
    frac = Fraction(int(target_hz), int(raw_hz)).limit_denominator(1000)
    sig = np.asarray(sig, dtype=np.float64).ravel()
    return resample_poly(sig, frac.numerator, frac.denominator)


def _window(sig: np.ndarray, win_len: int, stride: int) -> np.ndarray:
    """Per-record z-normalization + sliding windows -> (n, win_len) float32."""
    sig = np.asarray(sig, dtype=np.float64).ravel()
    if len(sig) < win_len:
        return np.zeros((0, win_len), dtype=np.float32)
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    n = max(0, (len(sig) - win_len) // stride + 1)
    if n <= 0:
        return np.zeros((0, win_len), dtype=np.float32)
    return np.stack([sig[i * stride: i * stride + win_len] for i in range(n)]).astype(np.float32)


def _time_split(windows: np.ndarray, win_len: int, stride: int, train_frac=0.6, dev_frac=0.2):
    """60/20/20 time-ordered split that drops the windows overlapping each boundary."""
    n = len(windows)
    a, b = int(n * train_frac), int(n * (train_frac + dev_frac))
    g = max(0, int(win_len) // max(1, int(stride)) - 1)
    if g and (b - a <= g or n - b <= g):         # never let the purge empty a split
        g = max(0, min(b - a - 1, n - b - 1, g))
    return windows[:a], windows[a + g:b], windows[b + g:]


def _push(by, split, windows, cls):
    if len(windows):
        by[split]["X"].append(windows)
        by[split]["y"].append(np.full(len(windows), cls, dtype=np.int64))


def _stratified_cap(X: np.ndarray, y: np.ndarray, cap: int, rng: np.random.Generator):
    if len(y) <= cap:
        return X, y
    classes = sorted(set(y.tolist()))
    per_class = max(1, cap // len(classes))
    idxs = []
    for cls in classes:
        cls_idx = np.flatnonzero(y == cls)
        idxs.extend(rng.choice(cls_idx, min(per_class, len(cls_idx)), replace=False).tolist())
    if len(idxs) < cap:
        rest = np.setdiff1d(np.arange(len(y)), np.asarray(idxs), assume_unique=False)
        idxs.extend(rng.choice(rest, min(cap - len(idxs), len(rest)), replace=False).tolist())
    rng.shuffle(idxs)
    idx = np.asarray(idxs, dtype=int)
    return X[idx], y[idx]


def _finalize(by, cfg, split_id, class_names, extra_meta) -> BearingSplits:
    dcfg = cfg["data"]
    win = int(dcfg["window_len"])
    arrays = {}
    for split in SPLITS:
        xs, ys = by[split]["X"], by[split]["y"]
        arrays[f"{split}_X"] = (np.concatenate(xs, axis=0).astype(np.float32) if xs
                                else np.zeros((0, win), np.float32))
        arrays[f"{split}_y"] = (np.concatenate(ys, axis=0).astype(np.int64) if ys
                                else np.zeros((0,), np.int64))
    if dcfg.get("max_windows_per_split"):            # smoke runs only
        rng = np.random.default_rng(int(cfg["project"]["seed"]))
        for split in SPLITS:
            arrays[f"{split}_X"], arrays[f"{split}_y"] = _stratified_cap(
                arrays[f"{split}_X"], arrays[f"{split}_y"], int(dcfg["max_windows_per_split"]), rng)
    n_classes = len(class_names)
    for split in SPLITS:
        present = sorted(set(arrays[f"{split}_y"].tolist()))
        if present != list(range(n_classes)):
            raise RuntimeError(f"[{dcfg['dataset']}] {split} split has classes {present}, "
                               f"expected 0..{n_classes - 1}")
    meta = {"dataset": dcfg["dataset"], "split_id": split_id,
            "sample_rate_hz": int(dcfg["sample_rate_hz"]), "window_len": win,
            "stride": int(dcfg["stride"]), "class_names": list(class_names), "n_classes": n_classes,
            "sizes": {s: int(len(arrays[f"{s}_y"])) for s in SPLITS},
            "n_per_class": {s: {str(c): int((arrays[f"{s}_y"] == c).sum())
                                for c in range(n_classes)} for s in SPLITS},
            **extra_meta}
    return BearingSplits(meta=meta, **arrays)


# ---------------- PU (Paderborn) ----------------
# Files: <root>/<bearing>/<condition>_<bearing>_<repetition>.mat, vibration channel at 64 kHz.

_REP_RE = re.compile(r"_(\d+)\.mat$", re.IGNORECASE)


def _pu_files(root: Path, bearing: str, condition: str) -> list[tuple[int, Path]]:
    bearing_dir = root / bearing
    if not bearing_dir.is_dir():
        return []
    prefix = f"{condition}_{bearing}_"
    out = []
    for path in bearing_dir.iterdir():
        if path.name.lower().endswith(".mat") and path.name.startswith(prefix):
            match = _REP_RE.search(path.name)
            if match:
                out.append((int(match.group(1)), path))
    return sorted(out, key=lambda item: item[0])


def _pu_vibration(path: Path) -> np.ndarray | None:
    try:
        mat = sio.loadmat(str(path))
    except Exception:
        return None
    for key, obj in mat.items():
        if key.startswith("_"):
            continue
        try:
            channels = obj["Y"][0][0]
        except Exception:
            continue
        n_ch = channels.shape[1] if len(channels.shape) > 1 else 1
        for i in range(n_ch):
            ch = channels[0, i] if len(channels.shape) > 1 else channels[i]
            try:
                if "vibration" in str(ch["Name"][0]).lower():
                    return np.asarray(ch["Data"][0], dtype=np.float64).ravel()
            except Exception:
                continue
    return None


def load_pu(cfg: dict) -> BearingSplits:
    dcfg = cfg["data"]
    root = Path(dcfg["root"])
    win, stride = int(dcfg["window_len"]), int(dcfg["stride"])
    fs, raw_fs = int(dcfg["sample_rate_hz"]), int(dcfg["raw_rate_hz"])
    classes = list(dcfg["classes"].items())
    rep_split = {}
    for split in ("train", "dev", "test"):
        for rep in dcfg[f"{split}_reps"]:
            rep_split[int(rep)] = split
    by = {s: {"X": [], "y": []} for s in SPLITS}

    def windows(path):
        sig = _pu_vibration(path)
        if sig is None or len(sig) < win:
            return np.zeros((0, win), dtype=np.float32)
        return _window(_resample(sig, raw_fs, fs), win, stride)

    for cls, (_, bearings) in enumerate(classes):
        for bearing in bearings:
            for condition in dcfg["train_conditions"]:
                for rep, path in _pu_files(root, bearing, condition):
                    if rep in rep_split:
                        _push(by, rep_split[rep], windows(path), cls)
            for _, path in _pu_files(root, bearing, dcfg["cross_condition"]):
                _push(by, "cross", windows(path), cls)

    h = hashlib.sha256()
    for key in ("window_len", "stride", "train_reps", "dev_reps", "test_reps", "train_conditions",
                "cross_condition", "classes"):
        h.update(str(dcfg[key]).encode("utf-8") + b"|")
    return _finalize(by, cfg, f"pu_{h.hexdigest()[:16]}", [name for name, _ in classes],
                     {"cross_condition": dcfg["cross_condition"]})


# ---------------- CWRU ----------------
# 12k drive-end fault data plus the normal baseline.  Files: IR014_2.mat, B021_0.mat,
# OR007@3_0.mat and 97-100.mat (normal); the trailing digit is the motor load (0-3).
# Ten classes: Normal + {IR, B, OR} x {0.007, 0.014, 0.021 inch}; 0.028-inch files are not used.

CWRU_CLASSES = ["Normal", "IR007", "IR014", "IR021", "B007", "B014", "B021",
                "OR007", "OR014", "OR021"]
_CWRU_NORMAL_LOAD = {97: 0, 98: 1, 99: 2, 100: 3}
_CWRU_DIAMETER = {"007": 0, "014": 1, "021": 2}


def _cwru_class(base: str, is_normal: bool):
    if is_normal:
        return 0
    dia = re.search(r"(007|014|021)", base)
    if not dia:
        return None
    d = _CWRU_DIAMETER[dia.group(1)]
    for prefix, offset in (("IR", 1), ("B", 4), ("OR", 7)):
        if base.startswith(prefix):
            return offset + d
    return None


def _cwru_load(fname: str, is_normal: bool):
    base = fname.split(".")[0]
    if is_normal and base.isdigit():
        return _CWRU_NORMAL_LOAD.get(int(base), 0)
    m = re.search(r"_(\d)\.mat$", fname)
    return int(m.group(1)) if m else None


def _cwru_drive_end(path: Path):
    try:
        mat = sio.loadmat(str(path))
    except Exception:
        return None
    for k in mat:
        if k.endswith("_DE_time"):
            return np.asarray(mat[k], dtype=np.float64).ravel()
    return None


def load_cwru(cfg: dict) -> BearingSplits:
    dcfg = cfg["data"]
    root = Path(dcfg["root"])
    win, stride = int(dcfg["window_len"]), int(dcfg["stride"])
    cross_load = int(dcfg["cross_load"])
    by = {s: {"X": [], "y": []} for s in SPLITS}
    files = []
    for d, is_normal in ((root / "12k Drive End Bearing Fault Data", False),
                         (root / "Normal Baseline Data", True)):
        if d.is_dir():
            files.extend((p, is_normal) for p in sorted(d.glob("*.mat")))

    for path, is_normal in files:
        cls = _cwru_class(path.name.split(".")[0], is_normal)
        load = _cwru_load(path.name, is_normal)
        if cls is None or load is None:
            continue
        sig = _cwru_drive_end(path)
        if sig is None:
            continue
        w = _window(sig, win, stride)
        if load == cross_load:
            _push(by, "cross", w, cls)
        else:
            for split, part in zip(("train", "dev", "test"), _time_split(w, win, stride)):
                _push(by, split, part, cls)
    return _finalize(by, cfg, f"cwru_{win}_{stride}_load{cross_load}_{len(CWRU_CLASSES)}c",
                     CWRU_CLASSES, {"cross_condition": f"load {cross_load}"})


# ---------------- JNU ----------------
# Files: n600_3_2.csv / ib600_2.csv / ob800_2.csv / tb1000_2.csv, single column at 50 kHz.
# Prefix n / ib / ob / tb = healthy / inner race / outer race / ball; the number is the speed (rpm).

JNU_CLASSES = ["Healthy", "InnerRace", "OuterRace", "Ball"]


def _jnu_class_and_speed(fname: str):
    base = fname.lower()
    m = re.search(r"(600|800|1000)", base)
    speed = int(m.group(1)) if m else None
    for prefix, cls in (("ib", 1), ("ob", 2), ("tb", 3), ("n", 0)):
        if base.startswith(prefix):
            return cls, speed
    return None, speed


def load_jnu(cfg: dict) -> BearingSplits:
    dcfg = cfg["data"]
    root = Path(dcfg["root"])
    win, stride = int(dcfg["window_len"]), int(dcfg["stride"])
    fs, raw_fs = int(dcfg["sample_rate_hz"]), int(dcfg["raw_rate_hz"])
    cross_speed = int(dcfg["cross_speed"])
    by = {s: {"X": [], "y": []} for s in SPLITS}
    for path in sorted(root.glob("*.csv")):
        cls, speed = _jnu_class_and_speed(path.name)
        if cls is None or speed is None:
            continue
        try:
            sig = np.loadtxt(str(path))
        except Exception:
            continue
        w = _window(_resample(sig, raw_fs, fs), win, stride)
        if speed == cross_speed:
            _push(by, "cross", w, cls)
        else:
            for split, part in zip(("train", "dev", "test"), _time_split(w, win, stride)):
                _push(by, split, part, cls)
    return _finalize(by, cfg, f"jnu_{win}_{stride}_speed{cross_speed}_{len(JNU_CLASSES)}c",
                     JNU_CLASSES, {"cross_condition": f"{cross_speed} rpm"})


# ---------------- Challenger scoring subset ----------------

@dataclass
class EdgeProbe:
    vedge_X: np.ndarray        # class-balanced dev subset the Challenger is scored on
    vedge_y: np.ndarray
    probe_X: np.ndarray        # a disjoint class-balanced dev subset, reserved
    probe_y: np.ndarray


def carve_edge_probe(splits: BearingSplits, n_per_class: int = 200, seed: int = 0) -> EdgeProbe:
    """Two disjoint, class-balanced subsets of dev, fixed by `seed` before evolution starts."""
    rng = np.random.default_rng(seed)
    X, y = splits.dev_X, splits.dev_y
    classes = sorted(set(y.tolist()))
    k = min(n_per_class, min(int((y == c).sum()) for c in classes) // 2)
    if k < 1:
        raise RuntimeError("dev split too small to carve the Challenger scoring subset")
    v_idx, p_idx = [], []
    for c in classes:
        chosen = rng.choice(np.flatnonzero(y == c), 2 * k, replace=False)
        v_idx.extend(chosen[:k].tolist())
        p_idx.extend(chosen[k:].tolist())
    v_idx, p_idx = np.array(sorted(v_idx)), np.array(sorted(p_idx))
    return EdgeProbe(vedge_X=X[v_idx], vedge_y=y[v_idx], probe_X=X[p_idx], probe_y=y[p_idx])
