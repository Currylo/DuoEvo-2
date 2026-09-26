"""Noise DSL: typed primitives, programs built from them, and SNR-locked injection.

A DSL program is {"components": [{"prim": name, "params": {...}}, ...], "weights": [...],
"snr_db": s}.  `apply_program` mixes the components, then `inject_noise` rescales the mixture so
every window receives it at exactly `snr_db`.  Random programs form the random half of each
curriculum; `fixed_program` defines the canonical families of the promotion gate.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as ss


Program = dict


RANDOM_PRIMITIVES = (
    "awgn",
    "colored",
    "arma",
    "lowpass",
    "bandpass",
    "sinusoid",
    "line_vfd",
    "impulse_resonance",
    "impulse_train",
    "resonant_ringdown",
    "order_harmonics",
    "cyclostationary",
    "burst",
    "amplitude_bursts",
)

PRIMITIVE_ALIASES = {
    "colored_noise": "colored",
    "arma_noise": "arma",
    "line_or_vfd_components": "line_vfd",
    "bandpass_resonance": "bandpass",
    "cyclostationary_modulation": "cyclostationary",
}


ALLOWED_PARAM_KEYS = {
    "awgn": set(),
    "colored": {"beta", "exponent"},
    "arma": {"a1", "a2", "ar_coeffs", "ma_coeffs"},
    "lowpass": {"cutoff_hz", "cutoff", "cutoff_freq_hz", "f_cut_hz"},
    "bandpass": {
        "band_lo_hz", "low_hz", "lo_hz", "f_low_hz", "lowcut", "lowcut_hz",
        "band_hi_hz", "high_hz", "hi_hz", "f_high_hz", "highcut", "highcut_hz",
        "center_hz", "center_freq_hz", "center", "freq_hz",
        "bandwidth_hz", "bw_hz", "q", "quality",
    },
    "sinusoid": {"freq_hz", "frequency_hz", "f0_hz", "base_freq_hz", "n_harm", "n_harmonics", "num_harmonics"},
    "line_vfd": {"line_hz", "f_line", "freq_hz", "f0", "f0_hz", "n_harm", "n_harmonics", "num_harmonics", "amp_drift", "drift", "modulation"},
    "impulse_resonance": {"rate_hz", "res_hz", "freq_hz", "decay_s", "decay", "jitter", "irregularity"},
    "impulse_train": {"rate_hz", "rate", "res_hz", "freq_hz", "freq", "decay_s", "decay", "jitter", "irregularity"},
    "resonant_ringdown": {"freq_hz", "freq", "res_hz", "damping", "damping_ratio", "n_excite", "n_events", "events"},
    "order_harmonics": {
        "f0_hz", "f0", "base_freq_hz", "order_hz", "order_freq",
        "n_harm", "n_harmonics", "num_harmonics", "orders",
        "drift", "speed_drift", "sideband", "sideband_amp", "decay_per_order",
    },
    "cyclostationary": {
        "carrier_hz", "carrier", "fc_hz", "freq_hz", "bw_hz", "bandwidth_hz",
        "mod_hz", "mod_freq_hz", "mod_rate_hz", "depth", "mod_depth",
    },
    "burst": {
        "band_lo_hz", "low_hz", "lo_hz", "f_low_hz",
        "band_hi_hz", "high_hz", "hi_hz", "f_high_hz",
        "duty", "duty_cycle", "burst_probability",
        "block_len", "block_size", "burst_len", "duration_s", "burst_duration_s",
        "burst_rate_hz", "rate_hz", "rate", "jitter", "irregularity", "amplitude", "amplitude_factor",
    },
    "amplitude_bursts": {
        "duty", "duty_cycle", "n_bursts", "bursts", "burst_rate_hz", "rate_hz", "rate",
        "burst_duration_s", "burst_length_s", "duration_s", "modulation_depth",
        "amplitude_factor", "amplitude",
        "band_lo_hz", "low_hz", "lo_hz", "f_low_hz",
        "band_hi_hz", "high_hz", "hi_hz", "f_high_hz",
        "tail_df", "df", "student_df",
        "burst_width_samples", "width_samples", "width_min_samples", "width_max_samples",
        "gap_start_samples", "gap_end_samples", "gap_drift", "arrival_drift",
        "jitter", "irregularity",
    },
}

LIST_PARAM_KEYS = {"ar_coeffs", "ma_coeffs", "orders"}


def _canonical_prim(prim: str) -> str:
    return PRIMITIVE_ALIASES.get(str(prim), str(prim))


def _unit_rms_batch(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.sqrt(np.mean(x * x, axis=-1, keepdims=True)) + 1e-12)


def inject_noise(X: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add zero-mean, unit-RMS-normalized `noise` to each row of `X` at `snr_db`."""
    X = np.asarray(X, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float64)
    if noise.ndim == 1:
        noise = np.tile(noise[None, :], (len(X), 1))
    out = np.empty_like(X, dtype=np.float32)
    for i in range(len(X)):
        n = noise[i].astype(np.float64)
        n = n - n.mean()
        n = n / (np.sqrt(np.mean(n * n)) + 1e-12)
        sp = float(np.mean(X[i].astype(np.float64) ** 2))
        out[i] = (X[i] + n * np.sqrt(sp / (10.0 ** (snr_db / 10.0)))).astype(np.float32)
    return out


def apply_program(program: Program, X: np.ndarray, fs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    lint_program(program)
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    noise = np.zeros_like(X, dtype=np.float64)
    weights = np.asarray(program.get("weights", []), dtype=np.float64)
    components = program["components"]
    if len(weights) != len(components):
        weights = np.ones(len(components), dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)
    for weight, comp in zip(weights, components):
        noise += float(weight) * _component_noise(comp, X.shape, fs, rng)
    noisy = inject_noise(X, noise, float(program["snr_db"]))
    return noisy.astype(np.float32), noise.astype(np.float32)


def lint_program(program: Program) -> None:
    if not isinstance(program, dict):
        raise ValueError("program must be a dict")
    comps = program.get("components")
    if not isinstance(comps, list) or not comps:
        raise ValueError("program.components must be a non-empty list")
    if len(comps) > 4:
        raise ValueError("program has too many components")
    snr = float(program.get("snr_db", -4.0))
    if not (-12.0 <= snr <= 6.0):
        raise ValueError("snr_db out of bounds")
    for comp in comps:
        prim = _canonical_prim(comp.get("prim"))
        if prim not in RANDOM_PRIMITIVES:
            raise ValueError(f"unknown primitive {prim}")
        params = comp.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"{prim}.params must be a dict")
        allowed = ALLOWED_PARAM_KEYS.get(prim, set())
        for key, value in params.items():
            if key not in allowed:
                raise ValueError(f"{prim}: unknown param {key}")
            if key in LIST_PARAM_KEYS:
                if not isinstance(value, (list, tuple)) or not all(isinstance(v, (int, float)) for v in value):
                    raise ValueError(f"{prim}.{key} must be a numeric list")
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{prim}.{key} must be numeric")


def _param(params: dict, names: tuple[str, ...], default):
    for name in names:
        if name in params:
            return params[name]
    return default


def random_program(rng: np.random.Generator, snr_db: float = -4.0, max_components: int = 4) -> Program:
    """1 to `max_components` random primitives with random mixing weights."""
    n = int(rng.integers(1, max_components + 1))
    comps = [_random_component(rng) for _ in range(n)]
    weights = rng.uniform(0.2, 1.0, size=n).tolist()
    return {"components": comps, "weights": weights, "snr_db": float(snr_db)}


def fixed_program(name: str, snr_db: float, rng: np.random.Generator | None = None) -> Program:
    rng = np.random.default_rng() if rng is None else rng
    if name == "awgn":
        comps = [{"prim": "awgn", "params": {}}]
    elif name == "lowpass":
        comps = [{"prim": "lowpass", "params": {"cutoff_hz": float(rng.uniform(800, 3500))}}]
    elif name == "pink":
        comps = [{"prim": "colored", "params": {"beta": 1.0}}]
    elif name == "mixed_static":
        comps = [
            {"prim": "awgn", "params": {}},
            {"prim": "lowpass", "params": {"cutoff_hz": 1800.0}},
            {"prim": "colored", "params": {"beta": 1.0}},
        ]
    elif name == "impulse_resonance":
        comps = [{"prim": "impulse_resonance", "params": {"rate_hz": 85.0, "res_hz": 3400.0, "decay_s": 0.003}}]
    elif name == "order_vfd":
        comps = [
            {"prim": "order_harmonics", "params": {"f0_hz": 27.0, "n_harm": 5}},
            {"prim": "sinusoid", "params": {"freq_hz": 2200.0, "n_harm": 3}},
        ]
    elif name == "burst_band":
        comps = [{"prim": "burst", "params": {"band_lo_hz": 1200.0, "band_hi_hz": 3300.0, "duty": 0.30}}]
    elif name == "complex_mixed":
        comps = [
            {"prim": "colored", "params": {"beta": 0.7}},
            {"prim": "impulse_resonance", "params": {"rate_hz": 92.0, "res_hz": 3600.0, "decay_s": 0.0035}},
            {"prim": "burst", "params": {"band_lo_hz": 1000.0, "band_hi_hz": 3200.0, "duty": 0.25}},
        ]
    else:
        raise ValueError(f"unknown fixed noise family {name}")
    return {"components": comps, "weights": [1.0] * len(comps), "snr_db": float(snr_db)}


def _spectral_peak_count(mean_psd: np.ndarray, freqs: np.ndarray) -> int:
    try:
        if float(mean_psd.max()) <= 0.0 or len(mean_psd) < 3:
            return 0
        p_db = 10.0 * np.log10(mean_psd + 1e-18)
        threshold = float(p_db.max() - 6.0)
        df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
        distance = max(1, int(round(80.0 / max(df, 1e-6))))
        peaks, _ = ss.find_peaks(p_db, height=threshold, distance=distance)
        return int(len(peaks))
    except Exception:
        return 0


def _spectral_type(low: float, mid: float, high: float, peak_count: int, env_kurt: float) -> str:
    if float(env_kurt) >= 8.0:
        return "impulsive"
    if int(peak_count) >= 3:
        return "harmonic"
    if low > 0.55 and (mid + high) < 0.45:
        return "lowpass"
    if int(peak_count) in (1, 2):
        return "bandpass"
    return "fullband"


def _dominant_band(low: float, mid: float, high: float) -> str:
    vals = {"low": float(low), "mid": float(mid), "high": float(high)}
    return max(vals, key=vals.get)


def _component_noise(comp: dict, shape: tuple[int, int], fs: int, rng: np.random.Generator) -> np.ndarray:
    prim = _canonical_prim(comp["prim"])
    params = comp.get("params", {})
    n, L = shape
    if prim == "awgn":
        return rng.standard_normal(shape)
    if prim == "colored":
        beta = _param(params, ("beta",), None)
        if beta is None and "exponent" in params:
            beta = abs(float(params["exponent"]))
        beta = float(1.0 if beta is None else beta)
        return np.stack([_colored_noise(L, beta, rng) for _ in range(n)])
    if prim == "arma":
        return _arma_noise(n, L, rng, params)
    if prim == "lowpass":
        raw = rng.standard_normal(shape)
        cutoff = float(_param(params, ("cutoff_hz", "cutoff", "cutoff_freq_hz", "f_cut_hz"), 2000.0))
        return _filter_batch(raw, fs, "lowpass", cutoff=cutoff)
    if prim == "bandpass":
        raw = rng.standard_normal(shape)
        lo = _param(params, ("band_lo_hz", "low_hz", "lo_hz", "f_low_hz", "lowcut", "lowcut_hz"), None)
        hi = _param(params, ("band_hi_hz", "high_hz", "hi_hz", "f_high_hz", "highcut", "highcut_hz"), None)
        if lo is None or hi is None:
            center = float(_param(params, ("center_hz", "center_freq_hz", "center", "freq_hz"), 2000.0))
            bw = _param(params, ("bandwidth_hz", "bw_hz"), None)
            if bw is None:
                q = float(_param(params, ("q", "quality"), 8.0))
                bw = max(center / max(q, 1e-6), 50.0)
            lo = center - float(bw) / 2.0
            hi = center + float(bw) / 2.0
        return _filter_batch(raw, fs, "bandpass", lo=max(5.0, float(lo)), hi=min(fs / 2.0 - 5.0, float(hi)))
    if prim == "sinusoid":
        freq = float(_param(params, ("freq_hz", "frequency_hz", "f0_hz", "base_freq_hz"), 50.0))
        n_harm = int(_param(params, ("n_harm", "n_harmonics", "num_harmonics"), 1))
        return _sinusoids(n, L, fs, rng, freq, n_harm)
    if prim == "line_vfd":
        return _line_vfd(n, L, fs, rng, params)
    if prim == "impulse_resonance":
        return _impulse_resonance(n, L, fs, rng, params)
    if prim == "impulse_train":
        return _impulse_train(n, L, fs, rng, params)
    if prim == "resonant_ringdown":
        return _resonant_ringdown(n, L, fs, rng, params)
    if prim == "order_harmonics":
        return _order_harmonics(n, L, fs, rng, params)
    if prim == "cyclostationary":
        return _cyclostationary(n, L, fs, rng, params)
    if prim == "burst":
        return _burst(n, L, fs, rng, params)
    if prim == "amplitude_bursts":
        return _amplitude_bursts(n, L, fs, rng, params)
    raise ValueError(f"unknown primitive {prim}")


def _random_component(rng: np.random.Generator) -> dict:
    prim = rng.choice(RANDOM_PRIMITIVES)
    if prim == "awgn":
        params = {}
    elif prim == "colored":
        params = {"beta": float(rng.uniform(0.0, 2.5))}
    elif prim == "arma":
        params = {"a1": float(rng.uniform(-0.85, 0.85)), "a2": float(rng.uniform(-0.75, 0.75))}
    elif prim == "lowpass":
        params = {"cutoff_hz": float(rng.uniform(600.0, 4200.0))}
    elif prim == "bandpass":
        params = {"center_hz": float(rng.uniform(300.0, 5200.0)), "q": float(rng.uniform(2.0, 20.0))}
    elif prim == "sinusoid":
        params = {"freq_hz": float(rng.uniform(40.0, 2600.0)), "n_harm": int(rng.integers(1, 6))}
    elif prim == "line_vfd":
        params = {
            "line_hz": float(rng.uniform(40.0, 80.0)),
            "n_harm": int(rng.integers(1, 7)),
            "amp_drift": float(rng.uniform(0.0, 0.30)),
        }
    elif prim == "impulse_resonance":
        params = {
            "rate_hz": float(rng.uniform(10.0, 220.0)),
            "res_hz": float(rng.uniform(800.0, 5200.0)),
            "decay_s": float(rng.uniform(0.001, 0.010)),
            "jitter": float(rng.uniform(0.05, 0.45)),
        }
    elif prim == "impulse_train":
        params = {
            "rate_hz": float(rng.uniform(2.0, 360.0)),
            "jitter": float(rng.uniform(0.05, 0.60)),
            "decay_s": float(rng.uniform(0.001, 0.018)),
            "res_hz": float(rng.uniform(800.0, 5200.0)),
        }
    elif prim == "resonant_ringdown":
        params = {
            "freq_hz": float(rng.uniform(500.0, 5400.0)),
            "damping": float(rng.uniform(0.004, 0.18)),
            "n_excite": int(rng.integers(1, 21)),
        }
    elif prim == "order_harmonics":
        params = {
            "f0_hz": float(rng.uniform(10.0, 80.0)),
            "n_harm": int(rng.integers(2, 8)),
            "drift": float(rng.uniform(0.0, 0.05)),
            "sideband": float(rng.uniform(0.0, 0.35)),
        }
    elif prim == "cyclostationary":
        params = {
            "carrier_hz": float(rng.uniform(500.0, 5200.0)),
            "mod_hz": float(rng.uniform(5.0, 300.0)),
            "depth": float(rng.uniform(0.15, 1.0)),
        }
    elif prim == "burst":
        lo = float(rng.uniform(300.0, 2500.0))
        hi = float(rng.uniform(lo + 300.0, 5400.0))
        params = {
            "band_lo_hz": lo,
            "band_hi_hz": hi,
            "duty": float(rng.uniform(0.08, 0.55)),
            "jitter": float(rng.uniform(0.10, 0.75)),
            "block_len": int(rng.integers(48, 320)),
        }
    elif prim == "amplitude_bursts":
        params = {"duty": float(rng.uniform(0.05, 0.55)), "n_bursts": int(rng.integers(1, 13))}
        if rng.random() < 0.65:
            lo = float(rng.choice([rng.uniform(500.0, 1200.0), rng.uniform(900.0, 1800.0), rng.uniform(1500.0, 2600.0)]))
            hi = float(rng.uniform(max(lo + 700.0, 2200.0), 5600.0))
            params["band_lo_hz"] = lo
            params["band_hi_hz"] = hi
    else:
        params = {}
    return {"prim": str(prim), "params": params}


def _colored_noise(L: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    freqs = np.fft.rfftfreq(L)
    freqs[0] = 1.0
    amp = 1.0 / np.power(freqs, beta / 2.0)
    amp[0] = 0.0
    phase = rng.uniform(0.0, 2.0 * np.pi, len(freqs))
    mag = rng.standard_normal(len(freqs)) * amp
    x = np.fft.irfft(mag * np.exp(1j * phase), n=L)
    return x / (x.std() + 1e-12)


def _arma_noise(n: int, L: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    ar = params.get("ar_coeffs")
    a1 = float(ar[0]) if isinstance(ar, (list, tuple)) and len(ar) >= 1 else float(_param(params, ("a1",), 0.5))
    a2 = float(ar[1]) if isinstance(ar, (list, tuple)) and len(ar) >= 2 else float(_param(params, ("a2",), -0.2))
    a1 = float(np.clip(a1, -0.95, 0.95))
    a2 = float(np.clip(a2, -0.90, 0.90))
    if a2 - abs(a1) <= -0.98:
        a1, a2 = 0.5, -0.2
    out = np.zeros((n, L), dtype=np.float64)
    exc = rng.standard_normal((n, L))
    for i in range(n):
        for t in range(L):
            y1 = out[i, t - 1] if t >= 1 else 0.0
            y2 = out[i, t - 2] if t >= 2 else 0.0
            out[i, t] = np.clip(exc[i, t] + a1 * y1 + a2 * y2, -1e6, 1e6)
    return _unit_rms_batch(np.nan_to_num(out))


def _filter_batch(x: np.ndarray, fs: int, kind: str, **kwargs) -> np.ndarray:
    nyq = fs / 2.0
    if kind == "lowpass":
        wn = np.clip(kwargs["cutoff"] / nyq, 1e-4, 0.99)
        b, a = ss.butter(6, wn, btype="low")
    elif kind == "bandpass":
        lo = np.clip(kwargs["lo"] / nyq, 1e-4, 0.98)
        hi = np.clip(kwargs["hi"] / nyq, lo + 1e-4, 0.99)
        b, a = ss.butter(4, [lo, hi], btype="band")
    else:
        raise ValueError(kind)
    return ss.filtfilt(b, a, x.astype(np.float64), axis=-1)


def _sinusoids(n: int, L: int, fs: int, rng: np.random.Generator, freq_hz: float, n_harm: int) -> np.ndarray:
    t = np.arange(L) / fs
    out = np.zeros((n, L), dtype=np.float64)
    for i in range(n):
        for h in range(1, n_harm + 1):
            f = freq_hz * h
            if f >= fs / 2.0:
                continue
            out[i] += (1.0 / h) * np.sin(2.0 * np.pi * f * t + rng.uniform(0.0, 2.0 * np.pi))
    return out


def _line_vfd(n: int, L: int, fs: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    line_hz = float(np.clip(_param(params, ("line_hz", "f_line", "freq_hz", "f0", "f0_hz"), 60.0), 40.0, 80.0))
    n_harm = int(np.clip(_param(params, ("n_harm", "n_harmonics", "num_harmonics"), 3), 1, 6))
    amp_drift = float(np.clip(_param(params, ("amp_drift", "drift", "modulation"), 0.05), 0.0, 0.3))
    t = np.arange(L) / fs
    out = np.zeros((n, L), dtype=np.float64)
    for i in range(n):
        env = 1.0 + amp_drift * np.sin(2.0 * np.pi * rng.uniform(0.2, 1.2) * t + rng.uniform(0, 2 * np.pi))
        for h in range(1, n_harm + 1):
            f = h * line_hz
            if f >= fs / 2.0:
                continue
            out[i] += (1.0 / h) * env * np.sin(2.0 * np.pi * f * t + rng.uniform(0.0, 2.0 * np.pi))
    return out


def _impulse_resonance(n: int, L: int, fs: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    rate = float(params.get("rate_hz", 80.0))
    res = float(params.get("res_hz", 3200.0))
    decay = float(params.get("decay_s", 0.003))
    jitter = float(np.clip(_param(params, ("jitter", "irregularity"), 0.10), 0.0, 0.8))
    klen = int(np.clip(6.0 * decay * fs, 16, L))
    kt = np.arange(klen) / fs
    kernel = np.exp(-kt / decay) * np.sin(2.0 * np.pi * res * kt)
    spacing = fs / max(rate, 1e-6)
    out = np.zeros((n, L), dtype=np.float64)
    for i in range(n):
        pos = rng.uniform(0, spacing)
        while pos < L:
            p = int(round(pos))
            length = min(klen, L - p)
            if length > 0:
                out[i, p : p + length] += rng.uniform(0.6, 1.4) * kernel[:length]
            pos += spacing * max(0.15, 1.0 + jitter * rng.standard_normal())
    return out


def _impulse_train(n: int, L: int, fs: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    rate = float(np.clip(_param(params, ("rate_hz", "rate"), 30.0), 2.0, 400.0))
    jitter = float(np.clip(_param(params, ("jitter", "irregularity"), 0.25), 0.0, 0.8))
    decay_s = float(np.clip(_param(params, ("decay_s", "decay"), 0.006), 0.0005, 0.030))
    res_hz = float(np.clip(_param(params, ("res_hz", "freq_hz", "freq"), 3000.0), 200.0, fs * 0.45))
    klen = int(np.clip(6.0 * decay_s * fs, 16, min(L, 512)))
    kt = np.arange(klen) / fs
    kernel = np.exp(-kt / decay_s) * np.sin(2.0 * np.pi * res_hz * kt)
    spacing = fs / rate
    out = np.zeros((n, L), dtype=np.float64)
    for i in range(n):
        pos = rng.uniform(0, spacing)
        while pos < L:
            p = int(round(pos))
            length = min(klen, L - p)
            if length > 0:
                amp = rng.choice([-1.0, 1.0]) * rng.lognormal(mean=0.0, sigma=0.35)
                out[i, p:p + length] += amp * kernel[:length]
            pos += spacing * max(0.10, 1.0 + jitter * rng.standard_normal())
    return out


def _resonant_ringdown(n: int, L: int, fs: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    freq = float(np.clip(_param(params, ("freq_hz", "freq", "res_hz"), 3000.0), 200.0, fs * 0.45))
    damping = float(np.clip(_param(params, ("damping", "damping_ratio"), 0.04), 0.002, 0.25))
    n_excite = int(np.clip(_param(params, ("n_excite", "n_events", "events"), 5), 1, 24))
    out = np.zeros((n, L), dtype=np.float64)
    max_len = min(L, 1024)
    for i in range(n):
        for _ in range(n_excite):
            s = int(rng.integers(0, L))
            length = min(max_len, L - s)
            if length <= 1:
                continue
            t = np.arange(length) / fs
            env = np.exp(-damping * 2.0 * np.pi * freq * t)
            out[i, s:s + length] += rng.uniform(0.5, 1.5) * env * np.sin(2.0 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    return out


def _order_harmonics(n: int, L: int, fs: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    f0 = float(_param(params, ("f0_hz", "f0", "base_freq_hz", "order_hz", "order_freq"), 30.0))
    orders = params.get("orders")
    if isinstance(orders, (list, tuple)) and orders:
        n_harm = int(np.clip(max(int(v) for v in orders if isinstance(v, (int, float))), 1, 8))
    else:
        n_harm = int(_param(params, ("n_harm", "n_harmonics", "num_harmonics"), 5))
    drift = float(np.clip(_param(params, ("drift", "speed_drift"), 0.0), 0.0, 0.08))
    sideband = float(np.clip(_param(params, ("sideband", "sideband_amp"), 0.0), 0.0, 0.6))
    t = np.arange(L) / fs
    out = np.zeros((n, L), dtype=np.float64)
    for i in range(n):
        speed = 1.0 + drift * np.sin(2.0 * np.pi * rng.uniform(0.2, 0.8) * t + rng.uniform(0, 2 * np.pi))
        for h in range(1, n_harm + 1):
            f = h * f0
            if f >= fs / 2.0:
                continue
            phase = rng.uniform(0.0, 2.0 * np.pi)
            out[i] += (1.0 / h) * np.sin(2.0 * np.pi * f * t * speed + phase)
            if sideband > 0:
                sb = min(fs / 2.0 - 5.0, f + f0)
                out[i] += sideband * (1.0 / h) * np.sin(2.0 * np.pi * sb * t + phase)
    return out


def _cyclostationary(n: int, L: int, fs: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    carrier = float(np.clip(_param(params, ("carrier_hz", "carrier", "fc_hz", "freq_hz"), 2500.0), 200.0, fs * 0.45))
    mod_hz = float(np.clip(_param(params, ("mod_hz", "mod_freq_hz", "mod_rate_hz"), 37.0), 3.0, 400.0))
    depth = float(np.clip(_param(params, ("depth", "mod_depth"), 0.6), 0.0, 1.0))
    t = np.arange(L) / fs
    out = np.zeros((n, L), dtype=np.float64)
    for i in range(n):
        env = 1.0 + depth * np.sin(2.0 * np.pi * mod_hz * t + rng.uniform(0.0, 2.0 * np.pi))
        out[i] = env * np.sin(2.0 * np.pi * carrier * t + rng.uniform(0.0, 2.0 * np.pi))
    return out


def _burst(n: int, L: int, fs: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    lo = float(_param(params, ("band_lo_hz", "low_hz", "lo_hz", "f_low_hz"), 1000.0))
    hi = float(_param(params, ("band_hi_hz", "high_hz", "hi_hz", "f_high_hz"), 3200.0))
    duty = float(np.clip(_param(params, ("duty", "duty_cycle", "burst_probability"), 0.3), 0.05, 0.8))
    raw = rng.standard_normal((n, L))
    base = _filter_batch(raw, fs, "bandpass", lo=lo, hi=hi)
    out = np.zeros((n, L), dtype=np.float64)
    duration = _param(params, ("duration_s", "burst_duration_s"), None)
    default_block = int(float(duration) * fs) if duration is not None else L // 12
    rate = _param(params, ("burst_rate_hz", "rate_hz", "rate"), None)
    if rate is not None:
        default_block = int(np.clip(fs / max(float(rate), 1e-6), 16, L))
    block = int(np.clip(_param(params, ("block_len", "block_size", "burst_len"), default_block), 32, L))
    jitter = float(np.clip(_param(params, ("jitter", "irregularity"), 0.0), 0.0, 0.8))
    for i in range(n):
        start = 0
        while start < L:
            if rng.random() < duty:
                cur_block = int(np.clip(block * rng.uniform(1.0 - jitter, 1.0 + jitter), 16, L))
                end = min(L, start + cur_block)
                win = np.hanning(end - start) if end - start > 1 else np.ones(1)
                out[i, start:end] = base[i, start:end] * win
                start += cur_block
            else:
                start += block
    return out


def _amplitude_bursts(n: int, L: int, fs: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    duration_s = _param(params, ("burst_duration_s", "burst_length_s", "duration_s"), None)
    rate_hz = _param(params, ("burst_rate_hz", "rate_hz", "rate"), None)
    if "duty" in params or "duty_cycle" in params:
        duty = float(np.clip(_param(params, ("duty", "duty_cycle"), 0.25), 0.03, 0.8))
    elif duration_s is not None and rate_hz is not None:
        duty = float(np.clip(float(duration_s) * float(rate_hz), 0.03, 0.8))
    else:
        duty = 0.25
    if "n_bursts" in params or "bursts" in params:
        n_bursts = int(np.clip(_param(params, ("n_bursts", "bursts"), 4), 1, 16))
    elif rate_hz is not None:
        n_bursts = int(np.clip(float(rate_hz) * L / 12000.0, 1, 16))
    else:
        n_bursts = 4
    tail_df = float(np.clip(_param(params, ("tail_df", "df", "student_df"), 3.0), 1.2, 12.0))
    jitter = float(np.clip(_param(params, ("jitter", "irregularity"), 0.0), 0.0, 0.95))
    out = np.zeros((n, L), dtype=np.float64)
    base_len = _amplitude_burst_base_len(params, L, fs, duty, n_bursts)
    width_min = _param(params, ("width_min_samples",), None)
    width_max = _param(params, ("width_max_samples",), None)
    gap_drift = float(np.clip(_param(params, ("gap_drift", "arrival_drift"), 0.0), 0.0, 3.0))
    gap_start = _param(params, ("gap_start_samples",), None)
    gap_end = _param(params, ("gap_end_samples",), None)
    nonstationary_gaps = gap_drift > 0.0 or gap_start is not None or gap_end is not None or rate_hz is not None
    for i in range(n):
        raw = rng.standard_t(df=tail_df, size=L)
        lo = _param(params, ("band_lo_hz", "low_hz", "lo_hz", "f_low_hz"), None)
        hi = _param(params, ("band_hi_hz", "high_hz", "hi_hz", "f_high_hz"), None)
        if lo is not None and hi is not None:
            raw = _filter_batch(raw[None, :], fs, "bandpass", lo=float(lo), hi=float(hi))[0]
        mask = np.zeros(L, dtype=np.float64)
        if nonstationary_gaps:
            start_gap = float(gap_start) if gap_start is not None else max(8.0, 1.5 * base_len)
            if gap_end is not None:
                end_gap = float(gap_end)
            elif rate_hz is not None:
                end_gap = fs / max(float(rate_hz), 1e-6)
            else:
                end_gap = start_gap * (1.0 + 3.0 * gap_drift)
            pos = float(rng.uniform(0.0, max(1.0, start_gap)))
            for _ in range(n_bursts):
                if pos >= L:
                    break
                length = _sample_amplitude_burst_len(rng, base_len, L, jitter, width_min, width_max)
                start = int(np.clip(round(pos), 0, max(0, L - 1)))
                end = min(L, start + length)
                win = np.hanning(end - start) if end - start > 1 else np.ones(1)
                mask[start:end] = np.maximum(mask[start:end], win)
                progress = float(np.clip(pos / max(L, 1), 0.0, 1.0))
                gap_scale = max(1.0, start_gap + (end_gap - start_gap) * progress)
                gap = rng.exponential(gap_scale)
                gap *= max(0.10, 1.0 + jitter * rng.standard_normal())
                pos += length + gap
        else:
            for _ in range(n_bursts):
                length = _sample_amplitude_burst_len(rng, base_len, L, jitter, width_min, width_max)
                start = int(rng.integers(0, max(1, L - length + 1)))
                win = np.hanning(length) if length > 1 else np.ones(1)
                mask[start:start + length] = np.maximum(mask[start:start + length], win)
        out[i] = raw * mask
    return out


def _amplitude_burst_base_len(params: dict, L: int, fs: int, duty: float, n_bursts: int) -> int:
    width = _param(params, ("burst_width_samples", "width_samples"), None)
    if width is not None:
        return int(np.clip(float(width), 4, L))
    width_min = _param(params, ("width_min_samples",), None)
    width_max = _param(params, ("width_max_samples",), None)
    if width_min is not None and width_max is not None:
        return int(np.clip(0.5 * (float(width_min) + float(width_max)), 4, L))
    duration_s = _param(params, ("burst_duration_s", "burst_length_s", "duration_s"), None)
    if duration_s is not None:
        return int(np.clip(float(duration_s) * fs, 4, L))
    return max(8, int(round(float(duty) * L / max(1, int(n_bursts)))))


def _sample_amplitude_burst_len(
    rng: np.random.Generator,
    base_len: int,
    L: int,
    jitter: float,
    width_min,
    width_max,
) -> int:
    if width_min is not None and width_max is not None:
        lo = int(np.clip(float(width_min), 4, L))
        hi = int(np.clip(float(width_max), lo, L))
        return int(rng.integers(lo, hi + 1))
    if jitter > 0.0:
        scale = rng.uniform(max(0.20, 1.0 - jitter), 1.0 + jitter)
    else:
        scale = rng.uniform(0.35, 2.0)
    return int(np.clip(base_len * scale, 4, L))
