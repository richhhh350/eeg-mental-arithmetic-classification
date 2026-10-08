"""Frozen traditional baselines and one-factor exploratory ablations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .cli import load_prepared, new_output, preprocess_windows
from .data import load_recording
from .provenance import digest, environment, file_digest, read_csv, read_json, source_fingerprint, write_csv, write_json
from .splits import assert_no_interval_overlap, chronological_blocks, window_intervals
from .stage2_features import extract_features
from .stage2_models import fit_grouped, fit_temporal, predict_scores, score_metrics


def window_rows(recording, subject, label, intervals, role, fs, offset=0):
    return [{"window_id": f"{recording}:{a+offset}:{b+offset}", "subject": subject,
             "recording_id": recording, "label": label, "split": role,
             "start_sample": a + offset, "stop_sample": b + offset, "fs_hz": fs}
            for a, b in intervals]


def _arrays(feature_chunks, rows):
    x = np.concatenate(feature_chunks)
    if len(x) != len(rows) or not np.isfinite(x).all():
        raise ValueError("Invalid feature array")
    return x, np.array([r["label"] for r in rows]), np.array([r["subject"] for r in rows])


def prepare(data_dir, config, reference_prepared, output):
    config = read_json(config)
    ref_x, ref_y, ref_groups, ref_rows, base, reference = load_prepared(reference_prepared)
    declared_base = read_json(config["stage1_config"])
    if declared_base != base:
        raise ValueError("Stage1 config no longer matches prepared reference")
    output = new_output(output)
    write_json(output / "config.json", config)
    write_json(output / "stage1_config.json", base)
    fs, length = base["fs_hz"], round(base["crop_seconds"] * base["fs_hz"])
    offset = round(base["crop_start_seconds"] * fs)
    expected_sha = {r["filename"]: r["sha256"] for r in reference["fingerprint"]["records"]}
    if {p.name for p in Path(data_dir).glob("*.edf")} != set(expected_sha):
        raise ValueError("EDF inventory differs from stage1")
    pools = {name: {"train": ([], []), "eval": ([], [])} for name in config["variants"]}
    temporal = {role: ([], []) for role in ("train", "val", "test")}
    feature_names = {}
    for filename in sorted(expected_sha):
        data, meta = load_recording(Path(data_dir) / filename, base["crop_start_seconds"], base["crop_seconds"])
        if meta["sha256"] != expected_sha[filename]:
            raise ValueError(f"Source recording changed: {filename}")
        if not np.isfinite(data).all() or np.any(np.std(data, axis=1) == 0):
            raise ValueError(f"Invalid or flat input: {filename}")
        recording = Path(filename).stem
        subject, condition = recording.split("_")
        label = int(condition) - 1
        geometry = {}
        for name, spec in config["variants"].items():
            for role in ("train", "eval"):
                size = round(spec["window_seconds"] * fs)
                stride = round((spec["train_stride_seconds"] if role == "train" else spec["window_seconds"]) * fs)
                key = (size, stride)
                if key not in geometry:
                    intervals = window_intervals(0, length, size, stride)
                    windows = np.stack([data[:, a:b] for a, b in intervals])
                    geometry[key] = (preprocess_windows(windows, base), intervals)
                windows, intervals = geometry[key]
                x, names = extract_features(windows, fs=fs, kind=spec["feature"], bands=spec["bands"])
                if name in feature_names and feature_names[name] != names:
                    raise ValueError("Feature order changed between recordings")
                feature_names[name] = names
                chunks, rows = pools[name][role]
                chunks.append(x)
                rows.extend(window_rows(recording, subject, label, intervals, role, fs, offset))
        blocks = chronological_blocks(length, fs, base["s1_gap_seconds"], base["s1_ratios"])
        for role, (a, b) in blocks.items():
            intervals = window_intervals(a, b, round(2 * fs), round(2 * fs))
            if not intervals:
                raise ValueError("S1 block has no full windows")
            windows = np.stack([data[:, left:right] for left, right in intervals])
            x, _ = extract_features(preprocess_windows(windows, base), fs=fs, kind="de", bands=base["bands_hz"])
            chunks, rows = temporal[role]
            chunks.append(x)
            rows.extend(window_rows(recording, subject, label, intervals, role, fs, offset))
        print(f"Prepared stage2 {recording}", flush=True)
    summaries = {}
    for name, pool in pools.items():
        folder = new_output(output / name)
        arrays = {}
        counts = {}
        for role, (chunks, rows) in pool.items():
            x, y, groups = _arrays(chunks, rows)
            arrays.update({f"X_{role}": x, f"y_{role}": y, f"groups_{role}": groups})
            write_csv(folder / f"{role}_windows.csv", rows)
            counts[role] = len(rows)
            if name == "de2":
                if not (np.array_equal(x, ref_x) and np.array_equal(y, ref_y) and np.array_equal(groups, ref_groups)
                        and [r["window_id"] for r in rows] == [r["window_id"] for r in ref_rows]):
                    raise AssertionError("Stage1 reference features or windows changed")
        np.savez_compressed(folder / "arrays.npz", **arrays)
        write_json(folder / "feature_names.json", feature_names[name])
        summaries[name] = {"counts": counts, "n_features": arrays["X_train"].shape[1]}
    folder = new_output(output / "s1")
    arrays, all_rows = {}, []
    for role, (chunks, rows) in temporal.items():
        x, y, groups = _arrays(chunks, rows)
        arrays.update({f"X_{role}": x, f"y_{role}": y, f"groups_{role}": groups})
        all_rows.extend(rows)
        write_csv(folder / f"{role}_windows.csv", rows)
    assert_no_interval_overlap(all_rows)
    np.savez_compressed(folder / "arrays.npz", **arrays)
    summaries["s1"] = {"counts": {role: len(temporal[role][1]) for role in temporal}, "n_features": 57}
    fingerprint = {"config": config, "stage1_config": base, "source_files": source_fingerprint(),
                   "environment": environment(), "source_dataset_id": reference["dataset_id"],
                   "records": reference["fingerprint"]["records"]}
    files = {p.relative_to(output).as_posix(): file_digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
    manifest = {"dataset_id": digest(fingerprint), "fingerprint": fingerprint, "files": files,
                "variants": summaries, "stage1_exact_feature_parity": True,
                "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"dataset_id": manifest["dataset_id"], "variants": summaries}), flush=True)


def load_variant(prepared, variant):
    prepared = Path(prepared)
    manifest = read_json(prepared / "manifest.json")
    if digest(manifest["fingerprint"]) != manifest["dataset_id"]:
        raise ValueError("Preparation fingerprint mismatch")
    # Verify all variant files and frozen configuration, not unrelated arrays.
    for name, sha in manifest["files"].items():
        if name.startswith(variant + "/") or name in ("config.json", "stage1_config.json"):
            if file_digest(prepared / name) != sha:
                raise ValueError(f"Prepared file changed: {name}")
    config = read_json(prepared / "config.json")
    if config != manifest["fingerprint"]["config"]:
        raise ValueError("Frozen config mismatch")
    with np.load(prepared / variant / "arrays.npz", allow_pickle=False) as npz:
        arrays = {key: npz[key] for key in npz.files}
    roles = ("train", "val", "test") if variant == "s1" else ("train", "eval")
    rows = {role: read_csv(prepared / variant / f"{role}_windows.csv") for role in roles}
    for role in roles:
        if not (np.array_equal(arrays[f"y_{role}"], [int(r["label"]) for r in rows[role]])
                and np.array_equal(arrays[f"groups_{role}"], [r["subject"] for r in rows[role]])
                and len(arrays[f"X_{role}"]) == len(rows[role])):
            raise ValueError("Arrays and metadata are misaligned")
    return arrays, rows, config, manifest


def experiment_specs(config):
    return [{**spec, "protocol": "S2"} for spec in config["s2_experiments"]] + [
        {"id": f"s1_de_{kind}", "kind": kind, "variant": "s1", "protocol": "S1"}
        for kind in config["s1_models"]]


def execute_experiment(prepared, output, spec, subjects=None):
    arrays, rows, config, manifest = load_variant(prepared, spec["variant"])
    output = new_output(output)
    target_role = "test" if spec["protocol"] == "S1" else "eval"
    targets = subjects or sorted(np.unique(arrays[f"groups_{target_role}"]).tolist())
    if len(targets) != len(set(targets)) or not set(targets) <= set(arrays[f"groups_{target_role}"]):
        raise ValueError("Invalid outer subject list")
    metadata = {"experiment": spec, "config": config, "dataset_id": manifest["dataset_id"],
                "subjects": targets, "source_files": source_fingerprint(), "environment": environment(),
                "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "run.json", metadata)
    results, predictions = [], []
    for target in targets:
        tick = time.perf_counter()
        try:
            target_mask = arrays[f"groups_{target_role}"] == target
            if spec["protocol"] == "S2":
                train = arrays["groups_train"] != target
                val = arrays["groups_eval"] != target
                with threadpool_limits(limits=1):
                    model, selection = fit_grouped(arrays["X_train"][train], arrays["y_train"][train], arrays["groups_train"][train],
                        arrays["X_eval"][val], arrays["y_eval"][val], arrays["groups_eval"][val],
                        spec["kind"], config["candidates"][spec["kind"]], config["inner_folds"], config["seed"])
            else:
                train = arrays["groups_train"] == target
                val = arrays["groups_val"] == target
                with threadpool_limits(limits=1):
                    model, selection = fit_temporal(arrays["X_train"][train], arrays["y_train"][train],
                        arrays["X_val"][val], arrays["y_val"][val], spec["kind"],
                        config["candidates"][spec["kind"]], config["seed"])
                selection["fit_subjects"] = [target]
                selection["validation_window_ids"] = [r["window_id"] for r, use in zip(rows["val"], val) if use]
            with threadpool_limits(limits=1):
                pred, score, score_kind = predict_scores(model, arrays[f"X_{target_role}"][target_mask])
            measured = score_metrics(arrays[f"y_{target_role}"][target_mask], pred, score)
            selection["fit_window_ids"] = [r["window_id"] for r, use in zip(rows["train"], train) if use]
            selection["test_window_ids"] = [r["window_id"] for r, use in zip(rows[target_role], target_mask) if use]
            selection["score_kind"] = score_kind
            write_json(output / f"{target}_selection.json", selection)
            joblib.dump(model, output / f"{target}_model.joblib")
            for idx, yhat, value in zip(np.flatnonzero(target_mask), pred, score):
                predictions.append({**rows[target_role][idx], "outer_subject": target, "prediction": int(yhat),
                                    "score": float(value), "score_kind": score_kind})
            results.append({"subject": target, "status": "ok", **measured,
                            "selected_params": json.dumps(selection["selected_params"], sort_keys=True),
                            "fit_windows": int(train.sum()), "test_windows": int(target_mask.sum()),
                            "elapsed_seconds": time.perf_counter() - tick})
        except Exception as error:
            results.append({"subject": target, "status": "failed", "error": repr(error),
                            "elapsed_seconds": time.perf_counter() - tick})
            (output / f"{target}_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_csv(output / "subject_metrics.csv", results)
        if predictions:
            write_csv(output / "predictions.csv", predictions)
        print(f"{spec['id']} {target}: {results[-1]['status']} ({len(results)}/{len(targets)})", flush=True)
    complete = all(r["status"] == "ok" for r in results)
    summary = {"experiment": spec, "requested_folds": len(targets),
               "completed_folds": sum(r["status"] == "ok" for r in results),
               "status": "complete" if complete else "incomplete", "dataset_id": manifest["dataset_id"]}
    if complete:
        for metric in ("balanced_accuracy", "accuracy", "macro_f1", "positive_class_f1", "roc_auc"):
            summary[f"mean_subject_{metric}"] = float(np.mean([r[metric] for r in results]))
        summary["elapsed_seconds"] = sum(r["elapsed_seconds"] for r in results)
        summary["min_subject_ba"] = min(r["balanced_accuracy"] for r in results)
        summary["max_subject_ba"] = max(r["balanced_accuracy"] for r in results)
    write_json(output / "summary.json", summary)
    if not complete:
        raise RuntimeError(f"{spec['id']} incomplete; inspect failure logs")
    verify_experiment(output, prepared)


def verify_experiment(output, prepared):
    output = Path(output)
    meta = read_json(output / "run.json")
    spec = meta["experiment"]
    arrays, pools, config, manifest = load_variant(prepared, spec["variant"])
    if meta["dataset_id"] != manifest["dataset_id"] or meta["config"] != config:
        raise AssertionError("Run and prepared dataset mismatch")
    test_role = "test" if spec["protocol"] == "S1" else "eval"
    predictions, results = read_csv(output / "predictions.csv"), read_csv(output / "subject_metrics.csv")
    expected = set(meta["subjects"])
    if len(results) != len(expected) or {r["subject"] for r in results} != expected:
        raise AssertionError("Missing or duplicated subject metrics")
    if len({p["window_id"] for p in predictions}) != len(predictions) or {p["outer_subject"] for p in predictions} != expected:
        raise AssertionError("Missing or duplicated predictions")
    candidates_checked = 0
    recomputed = []
    all_subjects = set(arrays["groups_train"].tolist())
    for row in results:
        if row["status"] != "ok":
            raise AssertionError("Failed fold cannot be verified as successful")
        target = row["subject"]
        selected = read_json(output / f"{target}_selection.json")
        test_rows = [r for r in pools[test_role] if r["subject"] == target]
        fit_rows = [r for r in pools["train"] if (r["subject"] != target if spec["protocol"] == "S2" else r["subject"] == target)]
        actual = [p for p in predictions if p["outer_subject"] == target]
        if set(selected["test_window_ids"]) != {r["window_id"] for r in test_rows} or set(selected["fit_window_ids"]) != {r["window_id"] for r in fit_rows}:
            raise AssertionError("Wrong fitting or testing windows")
        if {p["window_id"] for p in actual} != {r["window_id"] for r in test_rows}:
            raise AssertionError("Incomplete held-out predictions")
        expected_rows = {r["window_id"]: r for r in test_rows}
        for p in actual:
            source = expected_rows[p["window_id"]]
            if p["subject"] != target or any(p[key] != source[key] for key in source):
                raise AssertionError("Prediction metadata mismatch")
            threshold = 0 if p["score_kind"] == "decision_margin" else .5
            if p["score_kind"] != selected["score_kind"] or int(p["prediction"]) != int(float(p["score"]) >= threshold):
                raise AssertionError("Wrong decision threshold or score type")
        if spec["protocol"] == "S2":
            sources = all_subjects - {target}
            if set(selected["fit_subjects"]) != sources:
                raise AssertionError("Target used to fit S2")
            for candidate in selected["candidates"]:
                validation_seen = []
                for fold in candidate["folds"]:
                    fit_ids, val_ids = set(fold["fit_subjects"]), set(fold["validation_subjects"])
                    if fit_ids & val_ids or fit_ids | val_ids != sources:
                        raise AssertionError("Inner participant leakage")
                    validation_seen.extend(fold["validation_subjects"])
                    candidates_checked += 1
                if len(validation_seen) != len(sources) or set(validation_seen) != sources:
                    raise AssertionError("Inner validation does not cover each source once")
        else:
            val_rows = [r for r in pools["val"] if r["subject"] == target]
            if set(selected["validation_window_ids"]) != {r["window_id"] for r in val_rows}:
                raise AssertionError("S1 validation mismatch")
            assert_no_interval_overlap([{**r, "start_sample": int(r["start_sample"]),
                                         "stop_sample": int(r["stop_sample"])}
                                        for r in fit_rows + val_rows + test_rows])
        measured = score_metrics([int(p["label"]) for p in actual], [int(p["prediction"]) for p in actual], [float(p["score"]) for p in actual])
        for name, value in measured.items():
            if not np.isclose(float(row[name]), value, rtol=0, atol=1e-12):
                raise AssertionError(f"Metric mismatch: {target}/{name}")
        recomputed.append(measured)
    summary = read_json(output / "summary.json")
    for name in ("balanced_accuracy", "accuracy", "macro_f1", "positive_class_f1", "roc_auc"):
        if not np.isclose(summary[f"mean_subject_{name}"], np.mean([r[name] for r in recomputed]), rtol=0, atol=1e-12):
            raise AssertionError("Aggregate mismatch")
    write_json(output / "verification.json", {"passed": True, "folds": len(results), "windows": len(predictions),
               "inner_candidate_folds_checked": candidates_checked, "checks": ["manifest_hash", "window_alignment", "fit_test_boundaries",
               "inner_subject_isolation", "saved_score_threshold", "independently_recomputed_metrics"]})


def run(prepared, output, experiments=None, subjects=None):
    config = read_json(Path(prepared) / "config.json")
    specs = experiment_specs(config)
    if experiments:
        if not set(experiments) <= {s["id"] for s in specs}:
            raise ValueError("Unknown experiment")
        specs = [s for s in specs if s["id"] in experiments]
    output = new_output(output)
    write_json(output / "batch.json", {"config": config, "experiments": specs, "subjects": subjects or "all36",
               "created_utc": datetime.now(timezone.utc).isoformat()})
    failed = []
    for spec in specs:
        try:
            execute_experiment(prepared, output / spec["id"], spec, subjects)
        except Exception as error:
            failed.append({"experiment": spec["id"], "error": repr(error)})
            print(f"Experiment failure: {spec['id']}: {error}", flush=True)
    write_json(output / "batch_status.json", {"experiments_requested": len(specs), "failed": failed,
               "status": "complete" if not failed else "incomplete"})
    if failed:
        raise RuntimeError("One or more experiments failed; no complete stage2 result")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--config", default="configs/stage2.json")
    p.add_argument("--reference-prepared", default="outputs/prepared-stage1")
    p.add_argument("--output", required=True)
    r = sub.add_parser("run")
    r.add_argument("--prepared", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--experiments", nargs="+")
    r.add_argument("--subjects", nargs="+")
    v = sub.add_parser("verify")
    v.add_argument("--prepared", required=True)
    v.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"prepare": prepare, "run": run, "verify": verify_experiment}[command](**args)


if __name__ == "__main__":
    main()
