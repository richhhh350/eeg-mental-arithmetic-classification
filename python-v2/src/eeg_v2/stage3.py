"""Stage3 CLI: prepare, bridge, run, verify. All fits are source-only."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .cli import new_output
from .eegnet import ChannelScaler, EEGNet, infer, seed_all
from .provenance import digest, file_digest, read_csv, read_json, source_fingerprint, write_csv, write_json
from .stage2_models import fit_grouped, predict_scores, score_metrics
from .stage3_data import load_stage3, prepare, stage3_environment
from .stage3_training import fit_source_eegnet


def run_fingerprint(config, manifest):
    return {"config": config, "dataset_id": manifest["dataset_id"],
            "source_files": source_fingerprint(), "environment": stage3_environment()}


def _targets(features, subjects):
    result = subjects or sorted(set(features["groups"]))
    if len(set(result)) != len(result) or not set(result) <= set(features["groups"]):
        raise ValueError("Invalid target subject selection")
    return list(result)


def _summary(metrics, expected):
    complete = len(metrics) == expected and all(r["status"] == "ok" for r in metrics)
    result = {"requested_folds": expected, "completed_folds": sum(r["status"] == "ok" for r in metrics),
              "status": "complete" if complete else "incomplete"}
    if complete:
        for key in ("balanced_accuracy", "accuracy", "macro_f1", "positive_class_f1", "roc_auc"):
            result[f"mean_subject_{key}"] = float(np.mean([float(r[key]) for r in metrics]))
    return result


def bridge(prepared, output, subjects=None):
    _, features, rows, config, manifest = load_stage3(prepared)
    output = new_output(output)
    targets = _targets(features, subjects)
    write_json(output / "run.json", {"fingerprint": run_fingerprint(config, manifest), "subjects": targets})
    summaries = []
    for spec in config["bridges"]:
        folder = new_output(output / spec["id"])
        x, y, groups = features[spec["feature"]], features["y"], features["groups"]
        metrics, predictions = [], []
        for target in targets:
            tick = time.perf_counter()
            try:
                source, test = groups != target, groups == target
                with threadpool_limits(limits=1):
                    model, selection = fit_grouped(x[source], y[source], groups[source], x[source], y[source], groups[source],
                                                   spec["kind"], config["candidates"][spec["kind"]], config["inner_folds"], 42)
                    pred, score, kind = predict_scores(model, x[test])
                selection.update(fit_window_ids=[r["window_id"] for r, m in zip(rows, source) if m], test_window_ids=[r["window_id"] for r, m in zip(rows, test) if m])
                write_json(folder / f"{target}_selection.json", selection)
                joblib.dump(model, folder / f"{target}_model.joblib")
                predictions.extend({**r, "predicted": int(p), "score": float(s), "score_kind": kind} for r, p, s in zip([r for r, m in zip(rows, test) if m], pred, score))
                metrics.append({"subject": target, "status": "ok", **score_metrics(y[test], pred, score), "elapsed_seconds": time.perf_counter() - tick})
            except Exception as exc:
                write_json(folder / f"{target}_failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
                metrics.append({"subject": target, "status": "failed", "error": repr(exc)})
            write_csv(folder / "subject_metrics.csv", metrics)
            if predictions:
                write_csv(folder / "predictions.csv", predictions)
        summary = {"experiment": spec, **_summary(metrics, len(targets))}
        write_json(folder / "summary.json", summary)
        summaries.append(summary)
        print(f"Bridge {spec['id']}: {summary['status']} ({len(metrics)} subjects)", flush=True)
    write_json(output / "summary.json", {"experiments": summaries, "status": "complete" if all(r["status"] == "complete" for r in summaries) else "incomplete"})
    if any(r["status"] != "complete" for r in summaries):
        raise RuntimeError("Bridge contains failures")


def _check_completed(folder, fingerprint_id):
    completion = read_json(folder / "completed.json")
    if completion["fingerprint_id"] != fingerprint_id:
        raise ValueError("Completed fold has a different run fingerprint")
    for name, sha in completion["files"].items():
        if file_digest(folder / name) != sha:
            raise ValueError(f"Completed fold artifact changed: {name}")


def _fit_one(prepared, output, target, seed, fingerprint_id):
    folder = Path(output) / f"seed{seed}" / target
    tick = time.perf_counter()
    try:
        x, features, rows, config, manifest = load_stage3(prepared)
        if digest(run_fingerprint(config, manifest)) != fingerprint_id:
            raise ValueError("Worker environment or code differs from parent")
        source, test = features["groups"] != target, features["groups"] == target
        folder.mkdir(parents=True, exist_ok=False)
        def progress(value):
            write_json(folder / "progress.json", {"target": target, "seed": seed, **value})
        model, scaler, selection = fit_source_eegnet(x[source], features["y"][source], features["groups"][source], config, seed, progress)
        selection.update(fit_window_ids=[r["window_id"] for r, m in zip(rows, source) if m],
                         test_window_ids=[r["window_id"] for r, m in zip(rows, test) if m])
        # The fit is complete before the target array is transformed or scored.
        score = infer(model, scaler.transform(x[test]), config["training"]["batch_size"])
        pred = (score >= 0.5).astype(int)
        metrics = {"subject": target, "seed": seed, "status": "ok", **score_metrics(features["y"][test], pred, score),
                   "selected_epochs": selection["selected_epochs"], "elapsed_seconds": time.perf_counter() - tick}
        write_json(folder / "selection.json", selection)
        write_json(folder / "metrics.json", metrics)
        write_csv(folder / "predictions.csv", [{**r, "seed": seed, "predicted": int(p), "score": float(s), "score_kind": "arithmetic_probability"} for r, p, s in zip([r for r, m in zip(rows, test) if m], pred, score)])
        np.savez(folder / "scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
        torch.save(model.state_dict(), folder / "model.pt")
        write_json(folder / "completed.json", {"fingerprint_id": fingerprint_id, "target": target, "seed": seed,
                   "files": {name: file_digest(folder / name) for name in ("model.pt", "scaler.npz", "selection.json", "metrics.json", "predictions.csv")}})
        return metrics
    except Exception as exc:
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc(), "fingerprint_id": fingerprint_id})
        return {"subject": target, "seed": seed, "status": "failed", "error": repr(exc)}


def aggregate(output, targets, seeds):
    output = Path(output)
    all_metrics, all_predictions, by_seed = [], [], []
    for seed in seeds:
        metrics = []
        for target in targets:
            folder = output / f"seed{seed}" / target
            if (folder / "completed.json").exists():
                metric = read_json(folder / "metrics.json")
                all_predictions.extend(read_csv(folder / "predictions.csv"))
                metrics.append(metric)
            elif (folder / "failure.json").exists():
                metrics.append({"subject": target, "seed": seed, "status": "failed"})
        all_metrics.extend(metrics)
        by_seed.append({"seed": seed, **_summary(metrics, len(targets))})
    if all_metrics:
        write_csv(output / "subject_seed_metrics.csv", all_metrics)
    if all_predictions:
        write_csv(output / "predictions.csv", all_predictions)
    complete = all(s["status"] == "complete" for s in by_seed)
    summary = {"status": "complete" if complete else "incomplete", "subjects": targets, "seeds": seeds,
               "requested_folds": len(targets) * len(seeds), "completed_folds": sum(s["completed_folds"] for s in by_seed), "by_seed": by_seed}
    if complete:
        means = [{"subject": target, **{name: float(np.mean([float(r[name]) for r in all_metrics if r["subject"] == target]))
                 for name in ("balanced_accuracy", "accuracy", "macro_f1", "positive_class_f1", "roc_auc")}} for target in targets]
        write_csv(output / "subject_mean_metrics.csv", means)
        summary.update({f"mean_subject_seed_{name}": float(np.mean([r[name] for r in means])) for name in ("balanced_accuracy", "accuracy", "macro_f1", "positive_class_f1", "roc_auc")})
        summary["interpretation"] = "Mean of seed-specific participant metrics; not probability ensembling and not independent repeated participants."
    write_json(output / "summary.json", summary)
    return summary


def run(prepared, output, subjects=None, seeds=None, workers=1, resume=False):
    _, features, _, config, manifest = load_stage3(prepared)
    targets, seeds = _targets(features, subjects), seeds or config["seeds"]
    if not 1 <= workers <= 4 or len(set(seeds)) != len(seeds) or not set(seeds) <= set(config["seeds"]):
        raise ValueError("Invalid worker/seed request")
    output = Path(output)
    fingerprint = run_fingerprint(config, manifest)
    fingerprint_id = digest(fingerprint)
    if resume:
        old = read_json(output / "run.json")
        if old["fingerprint_id"] != fingerprint_id or old["fingerprint"] != fingerprint:
            raise ValueError("Resume requires unchanged data, config, code and environment")
    else:
        new_output(output)
        write_json(output / "run.json", {"fingerprint": fingerprint, "fingerprint_id": fingerprint_id, "started_utc": datetime.now(timezone.utc).isoformat()})
    write_json(output / "request.json", {"subjects": targets, "seeds": seeds, "workers": workers, "requested_utc": datetime.now(timezone.utc).isoformat()})
    tasks = []
    for target in targets:
        for seed in seeds:
            folder = output / f"seed{seed}" / target
            if (folder / "completed.json").exists():
                _check_completed(folder, fingerprint_id)
            elif folder.exists():
                raise ValueError(f"Partial/failed fold retained; use a new output directory: {folder}")
            else:
                tasks.append((target, seed))
    count = len(targets) * len(seeds) - len(tasks)
    aggregate(output, targets, seeds)
    if workers == 1:
        for target, seed in tasks:
            result = _fit_one(prepared, output, target, seed, fingerprint_id)
            count += 1
            aggregate(output, targets, seeds)
            print(f"EEGNet {target} seed{seed}: {result['status']} ({count}/{len(targets)*len(seeds)})", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            pending = {executor.submit(_fit_one, prepared, output, target, seed, fingerprint_id): (target, seed) for target, seed in tasks}
            for future in as_completed(pending):
                result = future.result()
                count += 1
                aggregate(output, targets, seeds)
                print(f"EEGNet {result['subject']} seed{result['seed']}: {result['status']} ({count}/{len(targets)*len(seeds)})", flush=True)
    summary = aggregate(output, targets, seeds)
    if summary["status"] != "complete":
        raise RuntimeError("EEGNet run incomplete; inspect failures")


def verify(prepared, output):
    x, features, rows, config, manifest = load_stage3(prepared)
    output = Path(output)
    run_info = read_json(output / "run.json")
    if run_info["fingerprint_id"] != digest(run_info["fingerprint"]) or run_info["fingerprint"]["dataset_id"] != manifest["dataset_id"] or run_info["fingerprint"]["config"] != config:
        raise ValueError("Run and prepared inputs differ")
    request = read_json(output / "request.json")
    checked, max_error = 0, 0.0
    for target in request["subjects"]:
        source, test = features["groups"] != target, features["groups"] == target
        expected_scaler = ChannelScaler().fit(x[source])
        for seed in request["seeds"]:
            folder = output / f"seed{seed}" / target
            _check_completed(folder, run_info["fingerprint_id"])
            selection = read_json(folder / "selection.json")
            if set(selection["fit_subjects"]) != set(features["groups"][source]) or selection["target_used_for_selection"]:
                raise AssertionError("Outer source isolation failed")
            if selection["fit_window_ids"] != [r["window_id"] for r, m in zip(rows, source) if m] or selection["test_window_ids"] != [r["window_id"] for r, m in zip(rows, test) if m]:
                raise AssertionError("Fit/test window IDs differ")
            val_coverage = []
            for fold in selection["inner_folds"]:
                a, b = set(fold["fit_subjects"]), set(fold["validation_subjects"])
                if a & b or a | b != set(selection["fit_subjects"]):
                    raise AssertionError("Inner isolation failed")
                val_coverage.extend(b)
                best = max(r["mean_subject_validation_ba"] for r in fold["history"])
                first = next(r["epoch"] for r in fold["history"] if best - r["mean_subject_validation_ba"] <= 1e-12)
                if fold["best_epoch"] != first:
                    raise AssertionError("Epoch selection mismatch")
                for row in fold["history"]:
                    if set(row["validation_subject_ba"]) != b or not np.isclose(row["mean_subject_validation_ba"], np.mean(list(row["validation_subject_ba"].values())), atol=1e-12, rtol=0):
                        raise AssertionError("Unequal validation subject scoring")
            if sorted(val_coverage) != sorted(selection["fit_subjects"]):
                raise AssertionError("Incomplete validation coverage")
            if selection["selected_epochs"] != int(np.median([r["best_epoch"] for r in selection["inner_folds"]])) or len(selection["final_history"]) != selection["selected_epochs"]:
                raise AssertionError("Final epoch budget mismatch")
            with np.load(folder / "scaler.npz", allow_pickle=False) as z:
                if not np.array_equal(z["mean"], expected_scaler.mean_) or not np.array_equal(z["scale"], expected_scaler.scale_):
                    raise AssertionError("Scaler fit outside declared sources")
            seed_all(seed, config["training"]["threads"])
            model = EEGNet(**config["model"])
            model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
            score = infer(model, expected_scaler.transform(x[test]), config["training"]["batch_size"])
            predictions = read_csv(folder / "predictions.csv")
            if [r["window_id"] for r in predictions] != selection["test_window_ids"] or not np.array_equal([int(r["label"]) for r in predictions], features["y"][test]):
                raise AssertionError("Prediction rows mismatch")
            difference = float(np.max(abs(score - np.array([float(r["score"]) for r in predictions]))))
            max_error = max(max_error, difference)
            if difference > 1e-7 or not np.array_equal(score >= 0.5, [int(r["predicted"]) for r in predictions]):
                raise AssertionError("Prediction replay failed")
            metrics = read_json(folder / "metrics.json")
            measured = score_metrics(features["y"][test], (score >= 0.5).astype(int), score)
            if any(not np.isclose(metrics[k], v, atol=1e-12, rtol=0) for k, v in measured.items()):
                raise AssertionError("Stored metrics mismatch")
            checked += 1
    result = {"passed": True, "models": checked, "max_score_error": max_error,
              "source_scalers_verified": True, "inner_groups_and_epoch_selection_verified": True,
              "frozen_inference_state_verified": True}
    write_json(output / "verification.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--config", default="configs/stage3.json")
    p.add_argument("--reference-prepared", required=True)
    p.add_argument("--output", required=True)
    for name in ("bridge", "run", "verify"):
        p = commands.add_parser(name)
        p.add_argument("--prepared", required=True)
        p.add_argument("--output", required=True)
        if name != "verify":
            p.add_argument("--subjects", nargs="+")
        if name == "run":
            p.add_argument("--seeds", nargs="+", type=int)
            p.add_argument("--workers", type=int, default=1)
            p.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        args["config_path"] = args.pop("config")
        prepare(**args)
    elif command == "bridge":
        bridge(**args)
    elif command == "run":
        run(**args)
    else:
        verify(**args)


if __name__ == "__main__":
    main()
