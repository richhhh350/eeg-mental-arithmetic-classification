"""Fixed per-window antialias resampling and traceable stage3 inputs."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path

import numpy as np
from scipy.signal import firwin, freqz, resample_poly

from .cli import load_prepared, new_output, preprocess_windows
from .data import load_recording
from .features import differential_entropy
from .provenance import digest, environment, file_digest, read_csv, read_json, source_fingerprint, write_csv, write_json


def stage3_environment():
    result = environment()
    result.update(compute="CPU; deterministic PyTorch; per-worker thread count in config")
    for name in ("torch", "filelock", "fsspec", "networkx", "sympy", "typing-extensions"):
        result["packages"][name] = importlib.metadata.version(name)
    return result


def antialias_filter(config):
    spec = config["resampling"]
    if (config["input_fs_hz"], config["output_fs_hz"], spec["up"], spec["down"]) != (500, 128, 32, 125):
        raise ValueError("This frozen stage implements only500->128 Hz")
    return firwin(spec["fir_taps"], spec["fir_cutoff_nyquist_fraction"], window=(spec["window"], spec["beta"]))


def resample_windows(x, config):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (19, 1000) or not np.isfinite(x).all():
        raise ValueError("Expected finite (windows,19,1000) input")
    spec = config["resampling"]
    result = resample_poly(x, spec["up"], spec["down"], axis=-1,
                           window=antialias_filter(config), padtype=spec["padtype"])
    result -= result.mean(axis=-1, keepdims=True)
    if result.shape != (len(x), 19, 256) or not np.isfinite(result).all():
        raise ValueError("Unexpected resampled dimensions or values")
    return result


def grid_matched_de(x):
    """500 Hz / NFFT1000 yields the same 0.5Hz bins as128 Hz / NFFT256."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (19, 1000) or not np.isfinite(x).all():
        raise ValueError("Invalid grid-control input")
    psd = abs(np.fft.rfft(x, axis=-1)) ** 2 / (1000 * 500)
    psd[..., 1:-1] *= 2
    f = np.fft.rfftfreq(1000, 1 / 500)
    power = np.stack([psd[..., (f >= a) & (f <= b)].sum(-1) * 0.5
                      for a, b in ((4, 8), (8, 13), (13, 30))], axis=-1)
    result = np.full_like(power, -50)
    mask = power > 1e-15
    result[mask] = 0.5 * np.log2(2 * np.pi * np.e * power[mask])
    return result.reshape(len(x), 57)


def filter_diagnostics(config):
    coefficients = antialias_filter(config)
    frequencies = np.array([4, 8, 13, 30, 45, 64, 80, 100, 150, 200], dtype=float)
    _, response = freqz(coefficients, worN=frequencies, fs=500 * 32)
    t = np.arange(1000) / 500
    tests = []
    for frequency in (10, 30, 100, 150):
        x = np.tile(np.sin(2 * np.pi * frequency * t), (1, 19, 1))
        output = resample_windows(x, config)[0, 0]
        interior_rms = np.sqrt(np.mean(output[16:-16] ** 2))
        full_rms = np.sqrt(np.mean(output ** 2))
        tests.append({"input_frequency_hz": frequency, "interior_rms": float(interior_rms),
                      "full_window_rms": float(full_rms), "interior_amplitude_db": float(20 * np.log10(max(interior_rms * np.sqrt(2), 1e-20)))})
    if not all(abs(r["interior_amplitude_db"]) < 0.2 for r in tests[:2]):
        raise AssertionError("Passband sinusoid check failed")
    if not all(r["interior_amplitude_db"] < -40 for r in tests[2:]):
        raise AssertionError("Stopband/alias rejection check failed")
    return {"response": [{"frequency_hz": float(f), "gain_db": float(20 * np.log10(max(abs(h), 1e-20)))} for f, h in zip(frequencies, response)],
            "synthetic_sinusoids": tests, "checks_passed": True,
            "half_support_original_samples": (len(coefficients) - 1) / (2 * 32),
            "half_support_seconds": (len(coefficients) - 1) / (2 * 32 * 500),
            "boundary_note": "Interior synthetic check omits16 output samples at each edge; real windows retain all256 samples and reflection can still affect edge estimates."}


def prepare(data_dir, config_path, reference_prepared, output):
    config = read_json(config_path)
    ref_x, y, groups, rows, base, reference = load_prepared(reference_prepared)
    if read_json(config["stage1_config"]) != base or base["window_seconds"] != 2 or base["stride_seconds"] != 2:
        raise ValueError("Stage1 reference configuration changed")
    expected = {r["filename"]: r["sha256"] for r in reference["fingerprint"]["records"]}
    if {p.name for p in Path(data_dir).glob("*.edf")} != set(expected):
        raise ValueError("EDF inventory changed")
    output = new_output(output)
    write_json(output / "config.json", config)
    write_json(output / "filter_diagnostics.json", filter_diagnostics(config))
    chunks, de500, degrid, de128, current_ids = [], [], [], [], []
    for filename in sorted(expected):
        data, meta = load_recording(Path(data_dir) / filename, base["crop_start_seconds"], base["crop_seconds"])
        if meta["sha256"] != expected[filename]:
            raise ValueError(f"Recording changed: {filename}")
        x = preprocess_windows(np.stack([data[:, a:a + 1000] for a in range(0, 30000, 1000)]), base)
        down = resample_windows(x, config)
        chunks.append(down.astype(np.float32))
        de500.append(differential_entropy(x))
        degrid.append(grid_matched_de(x))
        # Features use float32 values cast back tofloat64, matching exact CNN input.
        de128.append(differential_entropy(chunks[-1], fs=128))
        recording = Path(filename).stem
        current_ids.extend(f"{recording}:{a}:{a + 1000}" for a in range(0, 30000, 1000))
        print(f"Resampled {recording}", flush=True)
    if not np.array_equal(np.concatenate(de500), ref_x) or current_ids != [r["window_id"] for r in rows]:
        raise AssertionError("Stage1 features/IDs did not reproduce")
    np.save(output / "windows128.npy", np.concatenate(chunks), allow_pickle=False)
    np.savez_compressed(output / "features.npz", de500=ref_x, de500_grid=np.concatenate(degrid), de128=np.concatenate(de128), y=y, groups=groups)
    write_csv(output / "windows.csv", rows)
    np.save(output / "antialias_fir.npy", antialias_filter(config), allow_pickle=False)
    contrasts = []
    for name, a, b in (("grid_change_at500", np.concatenate(degrid), ref_x), ("resample_matched_grid", np.concatenate(de128), np.concatenate(degrid)), ("combined_change", np.concatenate(de128), ref_x)):
        d = a - b
        contrasts.append({"contrast": name, "median_abs_de": float(np.median(abs(d))), "p95_abs_de": float(np.quantile(abs(d), 0.95)), "max_abs_de": float(abs(d).max()), "rms_de": float(np.sqrt(np.mean(d * d)))})
    write_json(output / "feature_sensitivity.json", {"descriptive_only": True, "comparisons": contrasts, "stage1_exact_parity": True})
    fingerprint = {"config": config, "stage1_dataset_id": reference["dataset_id"], "records": reference["fingerprint"]["records"], "source_files": source_fingerprint(), "environment": stage3_environment()}
    manifest = {"dataset_id": digest(fingerprint), "fingerprint": fingerprint, "shape": [len(rows), 19, 256], "created_utc": datetime.now(timezone.utc).isoformat(), "files": {p.name: file_digest(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(output / "manifest.json", manifest)
    print(f"Prepared {len(rows)} windows; dataset_id={manifest['dataset_id']}", flush=True)


def load_stage3(path):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if digest(manifest["fingerprint"]) != manifest["dataset_id"]:
        raise ValueError("Stage3 fingerprint mismatch")
    for name, sha in manifest["files"].items():
        if file_digest(path / name) != sha:
            raise ValueError(f"Stage3 input changed: {name}")
    config = read_json(path / "config.json")
    if config != manifest["fingerprint"]["config"]:
        raise ValueError("Stage3 config mismatch")
    with np.load(path / "features.npz", allow_pickle=False) as z:
        features = {k: z[k] for k in z.files}
    x = np.load(path / "windows128.npy", allow_pickle=False)
    rows = read_csv(path / "windows.csv")
    if (x.shape != tuple(manifest["shape"]) or not np.isfinite(x).all()
            or not np.array_equal(features["y"], [int(r["label"]) for r in rows])
            or not np.array_equal(features["groups"], [r["subject"] for r in rows])):
        raise ValueError("Stage3 arrays and metadata mismatch")
    return x, features, rows, config, manifest
