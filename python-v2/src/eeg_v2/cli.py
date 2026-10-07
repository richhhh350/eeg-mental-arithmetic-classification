"""Commands: prepare, run and independently verify saved predictions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time
import traceback

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .data import CHANNELS, load_recording, quality_summary
from .evaluation import fit_source_model, metrics
from .features import differential_entropy
from .provenance import digest, environment, file_digest, read_csv, read_json, source_fingerprint, write_csv, write_json
from .splits import assert_no_interval_overlap, chronological_blocks, window_intervals


def preprocess_windows(windows, config):
    """Only instantaneous spatial reference and within-current-window centering."""
    result = np.asarray(windows, dtype=np.float64).copy()
    if config["preprocessing"]["additional_temporal_filter"] or config["preprocessing"]["additional_ica"]:
        raise ValueError("Stage1 implements neither extra temporal filtering nor ICA")
    if config["preprocessing"]["average_reference"]:
        result -= result.mean(axis=1, keepdims=True)
    if config["preprocessing"]["window_demean"]:
        result -= result.mean(axis=2, keepdims=True)
    return result


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def representative_plots(examples, output, fs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch
    fig, axes = plt.subplots(3, 2, figsize=(12, 7), constrained_layout=True)
    fig2, axes2 = plt.subplots(3, 2, figsize=(12, 7), constrained_layout=True)
    for index, (recording, data) in enumerate(examples):
        ax, psd = axes.flat[index], axes2.flat[index]
        signal = data[list(CHANNELS).index("Fz")]
        ax.plot(np.arange(fs * 10) / fs, signal[:fs * 10], lw=0.65)
        ax.set(title=recording + " / supplied Fz", xlabel="Time (s)", ylabel="uV")
        freq, power = welch(signal, fs=fs, nperseg=fs * 2)
        psd.semilogy(freq, np.maximum(power, 1e-20))
        psd.set(title=recording + " / supplied Fz", xlabel="Frequency (Hz)", ylabel="uV²/Hz", xlim=(0, 80))
    fig.savefig(output / "qc_waveforms.png", dpi=160)
    fig2.savefig(output / "qc_psd.png", dpi=160)
    plt.close("all")


def prepare(data_dir, config, output):
    config = read_json(config)
    output = new_output(output)
    write_json(output / "config.json", config)
    fs = config["fs_hz"]
    crop_offset = round(config["crop_start_seconds"] * fs)
    length = round(config["crop_seconds"] * fs)
    size = round(config["window_seconds"] * fs)
    stride = round(config["stride_seconds"] * fs)
    expected = {f"Subject{i:02d}_{c}.edf" for i in range(config["subjects"]) for c in (1, 2)}
    actual = {p.name for p in Path(data_dir).glob("*.edf")}
    if actual != expected:
        raise ValueError(f"EDF inventory mismatch: missing={expected-actual}, extra={actual-expected}")
    features, rows, s1_rows, records, qc, examples = [], [], [], [], [], []
    preprocess_hash = digest(config["preprocessing"])
    for filename in sorted(expected):
        data, meta = load_recording(Path(data_dir) / filename, config["crop_start_seconds"], config["crop_seconds"])
        if data.shape != (19, length):
            raise ValueError(f"Unexpected shape in {filename}: {data.shape}")
        if not np.isfinite(data).all() or np.any(np.std(data, axis=1) == 0):
            raise ValueError(f"Nonfinite or wholly flat channel: {filename}")
        subject, condition = Path(filename).stem.split("_")
        label = int(condition) - 1
        recording = Path(filename).stem
        records.append(meta)
        qc.append({"recording_id": recording, **quality_summary(data, fs)})
        if subject in config["smoke_subjects"]:
            examples.append((recording, data))
        intervals = window_intervals(0, length, size, stride)
        windows = np.stack([data[:, a:b] for a, b in intervals])
        features.append(differential_entropy(preprocess_windows(windows, config), fs, config["bands_hz"]))
        for a, b in intervals:
            rows.append({"window_id": f"{recording}:{a+crop_offset}:{b+crop_offset}",
                         "subject": subject, "recording_id": recording, "label": label,
                         "condition": "rest" if label == 0 else "arithmetic", "split": "S2_pool",
                         "start_sample": a + crop_offset, "stop_sample": b + crop_offset,
                         "fs_hz": fs, "window_samples": size, "stride_samples": stride,
                         "preprocessing_sha256": preprocess_hash})
        blocks = chronological_blocks(length, fs, config["s1_gap_seconds"], config["s1_ratios"])
        for split, (start, stop) in blocks.items():
            intervals_s1 = window_intervals(start, stop, size, stride)
            if not intervals_s1:
                raise ValueError(f"No complete S1 windows: {recording}/{split}")
            for a, b in intervals_s1:
                s1_rows.append({"subject": subject, "recording_id": recording, "label": label,
                                "split": split, "start_sample": a + crop_offset,
                                "stop_sample": b + crop_offset, "block_start_sample": start + crop_offset,
                                "block_stop_sample": stop + crop_offset, "fs_hz": fs})
        print(f"Prepared {recording}: {len(intervals)} windows", flush=True)
    assert_no_interval_overlap(s1_rows)
    x = np.concatenate(features)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite extracted features")
    np.savez_compressed(output / "features.npz", X=x,
                        y=np.array([r["label"] for r in rows]),
                        groups=np.array([r["subject"] for r in rows]))
    write_csv(output / "windows.csv", rows)
    write_csv(output / "s1_windows.csv", s1_rows)
    write_json(output / "quality.json", qc)
    write_json(output / "records.json", records)
    fingerprint = {"config": config, "source_files": source_fingerprint(), "environment": environment(),
                   "records": [{"filename": r["filename"], "sha256": r["sha256"]} for r in records]}
    manifest = {"dataset_id": digest(fingerprint), "fingerprint": fingerprint,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "n_windows": len(rows), "n_features": x.shape[1], "n_s1_windows": len(s1_rows),
                "feature_names": [f"{ch}/{band}/DE_log2" for ch in CHANNELS for band in config["band_names"]],
                "files": {name: file_digest(output / name) for name in
                          ("features.npz", "windows.csv", "s1_windows.csv", "quality.json", "records.json", "config.json")}}
    write_json(output / "manifest.json", manifest)
    representative_plots(examples, output, fs)
    print(f"Prepared dataset {manifest['dataset_id']}: {x.shape}", flush=True)


def load_prepared(prepared):
    prepared = Path(prepared)
    manifest = read_json(prepared / "manifest.json")
    for name, expected in manifest["files"].items():
        if file_digest(prepared / name) != expected:
            raise ValueError(f"Prepared artifact was modified: {name}")
    config = read_json(prepared / "config.json")
    if digest(manifest["fingerprint"]) != manifest["dataset_id"] or config != manifest["fingerprint"]["config"]:
        raise ValueError("Manifest/config fingerprint mismatch")
    with np.load(prepared / "features.npz", allow_pickle=False) as arrays:
        x, y, groups = arrays["X"], arrays["y"], arrays["groups"]
    rows = read_csv(prepared / "windows.csv")
    if len(rows) != len(x) or not np.array_equal(y, [int(r["label"]) for r in rows]) or not np.array_equal(groups, [r["subject"] for r in rows]):
        raise ValueError("Feature and window metadata are not aligned")
    return x, y, groups, rows, config, manifest


def run(prepared, output, subjects=None):
    x, y, groups, rows, config, manifest = load_prepared(prepared)
    targets = subjects or sorted(np.unique(groups).tolist())
    if len(targets) != len(set(targets)) or not set(targets) <= set(groups):
        raise ValueError("Invalid or repeated outer subject IDs")
    output = new_output(output)
    run_meta = {"dataset_id": manifest["dataset_id"], "config": config, "subjects": targets,
                "source_files": source_fingerprint(), "environment": environment(),
                "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "run.json", run_meta)
    predictions, results = [], []
    for target in targets:
        tick = time.perf_counter()
        try:
            train, test = groups != target, groups == target
            with threadpool_limits(limits=1):
                model, selection = fit_source_model(x[train], y[train], groups[train], config)
                if target in selection["fit_subjects"]:
                    raise AssertionError("Target participant entered training")
                probability = model.predict_proba(x[test])[:, 1]
            score = metrics(y[test], probability)
            # Dummy prevalence is estimated from sources; no test labels determine it.
            dummy = metrics(y[test], np.full(test.sum(), y[train].mean()))
            selection["fit_window_ids"] = [r["window_id"] for r, use in zip(rows, train) if use]
            selection["test_window_ids"] = [r["window_id"] for r, use in zip(rows, test) if use]
            write_json(output / f"{target}_selection.json", selection)
            joblib.dump(model, output / f"{target}_model.joblib")
            for index, p in zip(np.flatnonzero(test), probability):
                predictions.append({**rows[index], "outer_subject": target, "probability_arithmetic": float(p),
                                    "prediction": int(p >= .5), "selected_C": selection["selected_C"]})
            results.append({"subject": target, "status": "ok", "selected_C": selection["selected_C"],
                            **score, "dummy_balanced_accuracy": dummy["balanced_accuracy"],
                            "elapsed_seconds": time.perf_counter() - tick})
            print(f"Completed {target} ({len(results)}/{len(targets)})", flush=True)
        except Exception as error:
            results.append({"subject": target, "status": "failed", "error": repr(error),
                            "elapsed_seconds": time.perf_counter() - tick})
            (output / f"{target}_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"FAILED {target}: {error}", flush=True)
        write_csv(output / "subject_metrics.csv", results)
        if predictions:
            write_csv(output / "predictions.csv", predictions)
    complete = all(r["status"] == "ok" for r in results)
    summary = {"requested_folds": len(targets), "completed_folds": sum(r["status"] == "ok" for r in results),
               "status": "complete" if complete else "incomplete", "dataset_id": manifest["dataset_id"]}
    if complete:
        for name in ("balanced_accuracy", "accuracy", "macro_f1", "positive_class_f1", "roc_auc", "dummy_balanced_accuracy"):
            summary[f"mean_subject_{name}"] = float(np.mean([r[name] for r in results]))
        values = [r["balanced_accuracy"] for r in results]
        summary["subject_ba_min"] = min(values)
        summary["subject_ba_max"] = max(values)
        summary["subject_ba_median"] = float(np.median(values))
    write_json(output / "summary.json", summary)
    if not complete:
        raise RuntimeError("Incomplete run; inspect per-subject failure logs")
    verify(output)
    plot_results(results, output)


def verify(output):
    output = Path(output)
    predictions = read_csv(output / "predictions.csv")
    results = read_csv(output / "subject_metrics.csv")
    meta = read_json(output / "run.json")
    expected = set(meta["subjects"])
    if set(r["subject"] for r in results) != expected or len(results) != len(expected):
        raise AssertionError("Missing or duplicated subjects")
    if len({p["window_id"] for p in predictions}) != len(predictions):
        raise AssertionError("Duplicate predictions")
    if {p["outer_subject"] for p in predictions} != expected:
        raise AssertionError("Unexpected prediction subjects")
    for p in predictions:
        if p["subject"] != p["outer_subject"] or int(p["prediction"]) != int(float(p["probability_arithmetic"]) >= .5):
            raise AssertionError("Inconsistent prediction ownership or class")
    checked = []
    for row in results:
        if row["status"] != "ok":
            raise AssertionError("Failed fold is not a successful baseline")
        subject = row["subject"]
        pred = [p for p in predictions if p["outer_subject"] == subject]
        selection = read_json(output / f"{subject}_selection.json")
        if subject in selection["fit_subjects"] or set(selection["fit_window_ids"]) & set(selection["test_window_ids"]):
            raise AssertionError("Train/test overlap")
        if set(p["window_id"] for p in pred) != set(selection["test_window_ids"]):
            raise AssertionError("Missing test window")
        for candidate in selection["candidates"]:
            for fold in candidate["folds"]:
                if set(fold["fit_subjects"]) & set(fold["validation_subjects"]) or subject in fold["fit_subjects"] + fold["validation_subjects"]:
                    raise AssertionError("Nested participant leakage")
        recalculated = metrics([int(p["label"]) for p in pred], [float(p["probability_arithmetic"]) for p in pred])
        for name, value in recalculated.items():
            if not np.isclose(float(row[name]), value, rtol=0, atol=1e-12):
                raise AssertionError(f"Saved metric mismatch: {subject}/{name}")
        checked.append(subject)
    write_json(output / "verification.json", {"passed": True, "folds_verified": len(checked),
               "windows_verified": len(predictions), "checks": ["unique_windows", "complete_test_ids", "recomputed_metrics",
                                                                 "source_target_disjoint", "inner_subject_disjoint"]})
    print(f"Verified {len(checked)} folds and {len(predictions)} predictions", flush=True)


def plot_results(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    scores = np.array([r["balanced_accuracy"] for r in results])
    ax.bar([r["subject"].removeprefix("Subject") for r in results], scores, color="#24788a")
    ax.axhline(.5, ls="--", color="#b14d3d", label="Dummy BA = 0.50")
    ax.axhline(scores.mean(), color="#21364d", label=f"Equal-subject mean = {scores.mean():.3f}")
    ax.set(ylim=(0, 1.03), xlabel="Held-out participant", ylabel="Balanced accuracy", title="Stage 1: nested LOSO / DE + logistic regression")
    ax.legend(loc="upper right")
    fig.savefig(output / "subject_balanced_accuracy.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--config", default="configs/stage1.json")
    p.add_argument("--output", required=True)
    r = sub.add_parser("run")
    r.add_argument("--prepared", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--subjects", nargs="+")
    v = sub.add_parser("verify")
    v.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"prepare": prepare, "run": run, "verify": verify}[command](**args)


if __name__ == "__main__":
    main()
