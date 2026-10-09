"""Reproducible 36-participant finite-calibration experiment and replay audit."""
from __future__ import annotations
import argparse
import importlib.metadata
from pathlib import Path
import time
import traceback
import joblib
import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from eeg_v2.provenance import read_json, write_json, read_csv, write_csv, digest, file_digest, source_fingerprint
from eeg_v2.stage3_data import load_stage3, stage3_environment
from eeg_v2.stage2_models import score_metrics
from .calibration import experiments, window_masks, select_source, fit_final, predict, fit_adapter


def fingerprint(config, manifest, modules=("calibration.py", "runner.py")):
    return {"config": config, "dataset_id": manifest["dataset_id"], "legacy_code": source_fingerprint(),
            "code": {name: file_digest(Path(__file__).parent / name) for name in modules},
            "environment": stage3_environment(),
            "extra_packages": {p: importlib.metadata.version(p) for p in ("umap-learn", "numba", "llvmlite", "pynndescent")}}


def load_data(prepared, config_path="configs/stage4.json"):
    config = read_json(config_path)
    _, features, rows, _, manifest = load_stage3(prepared)
    if manifest["dataset_id"] != config["prepared_stage3_dataset_id"]:
        raise ValueError("Stage3 input identity changed")
    x, y, groups = features[config["feature"]], features["y"], features["groups"]
    assert x.shape == (2160, 57) and len(set(groups)) == 36
    calibration, test = window_masks(rows, config)
    for target in sorted(set(groups)):
        for label in (0, 1):
            cell = (groups == target) & (y == label)
            assert cell.sum() == 30 and (cell & test).sum() == 18
            for budget, mask in calibration.items():
                assert (cell & mask).sum() == budget // 2
    return x, y, groups, rows, calibration, test, config, manifest


def check_completed(folder, fid):
    record = read_json(folder / "completed.json")
    if record["fingerprint_id"] != fid:
        raise ValueError("Fold fingerprint mismatch")
    for name, sha in record["files"].items():
        if file_digest(folder / name) != sha:
            raise ValueError(f"Changed completed artifact {folder / name}")


def initialize_output(output, fp):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fid = digest(fp)
    if (output / "run.json").exists():
        if read_json(output / "run.json")["fingerprint_id"] != fid:
            raise ValueError("Resume configuration/code/environment/input differs")
    else:
        if list(output.iterdir()):
            raise ValueError("Nonempty unregistered output directory")
        write_json(output / "run.json", {"fingerprint": fp, "fingerprint_id": fid})
    return output, fid


def run(prepared="outputs/prepared-stage3", output="outputs/runs-stage4", config_path="configs/stage4.json", subjects=None):
    x, y, groups, rows, calibration, test, config, manifest = load_data(prepared, config_path)
    fp = fingerprint(config, manifest)
    output, fid = initialize_output(output, fp)
    targets = subjects or sorted(set(groups))
    if len(set(targets)) != len(targets) or not set(targets) <= set(groups):
        raise ValueError("Invalid targets")
    specs = experiments(config)
    for target in targets:
        folder = output / target
        if (folder / "completed.json").exists():
            check_completed(folder, fid)
            continue
        if folder.exists():
            raise FileExistsError("Archive incomplete evidence only after checking old process stopped: " + str(folder))
        folder.mkdir()
        tick = time.perf_counter()
        try:
            source, held = groups != target, groups == target
            suffix = held & test
            with threadpool_limits(limits=1):
                selection = select_source(x[source], y[source], groups[source], {b: m[source] for b, m in calibration.items()}, test[source], config)
                selection.update(fit_window_ids=[r["window_id"] for r, use in zip(rows, source) if use],
                                 test_window_ids=[r["window_id"] for r, use in zip(rows, suffix) if use],
                                 calibration_window_ids={str(b): [r["window_id"] for r, use in zip(rows, held & m) if use] for b, m in calibration.items()})
                models, predictions, metrics = {}, [], []
                for key, method, budget in specs:
                    fit_start = time.perf_counter()
                    prefix = None if not budget else x[held & calibration[budget]]
                    model = fit_final(x[source], y[source], prefix, method, selection["selected_C"][key])
                    fit_seconds = time.perf_counter() - fit_start
                    predict_start = time.perf_counter()
                    probability = predict(model, x[suffix])
                    prediction_seconds = time.perf_counter() - predict_start
                    models[key] = model
                    metrics.append({"subject": target, "experiment": key, "method": method, "budget_seconds_per_record": budget,
                                    "actual_target_seconds": 0 if not budget else len(prefix)*2, "status": "ok", "C": selection["selected_C"][key],
                                    **score_metrics(y[suffix], probability >= .5, probability), "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds,
                                    **model["adapter"].diagnostics})
                    predictions.extend({"experiment": key, **r, "predicted": int(p >= .5), "score": float(p)} for r, p in zip([r for r, use in zip(rows, suffix) if use], probability))
            selection["elapsed_seconds"] = time.perf_counter() - tick
            joblib.dump(models, folder / "models.joblib")
            write_json(folder / "selection.json", selection)
            write_csv(folder / "metrics.csv", metrics)
            write_csv(folder / "predictions.csv", predictions)
            write_json(folder / "completed.json", {"fingerprint_id": fid, "files": {n: file_digest(folder / n) for n in ("models.joblib", "selection.json", "metrics.csv", "predictions.csv")}})
            print(f"Stage4 {target}: 7 configurations complete ({selection['elapsed_seconds']:.1f}s)", flush=True)
        except Exception as exc:
            write_json(folder / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
            raise
        aggregate(output, config, fid)
    aggregate(output, config, fid)


def aggregate(output, config, fid):
    metrics, predictions = [], []
    for f in sorted(output.glob("Subject*/completed.json")):
        check_completed(f.parent, fid)
        metrics += read_csv(f.parent / "metrics.csv")
        predictions += read_csv(f.parent / "predictions.csv")
    n = len(metrics)
    summary = {"status": "complete" if n == 252 else "incomplete", "completed_models": n, "requested_models": 252, "predictions": len(predictions)}
    if n:
        write_csv(output / "subject_metrics.csv", metrics)
        write_csv(output / "predictions.csv", predictions)
    if n == 252:
        summary["experiments"] = [{"experiment": key, "subjects": 36,
             **{f"mean_subject_{m}": float(np.mean([float(r[m]) for r in metrics if r["experiment"] == key])) for m in ("balanced_accuracy", "macro_f1", "roc_auc")}}
             for key, _, _ in experiments(config)]
    write_json(output / "summary.json", summary)


def manual_metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    predicted = scores >= .5
    tp = int(np.sum((labels == 1) & predicted)); tn = int(np.sum((labels == 0) & ~predicted))
    fp = int(np.sum((labels == 0) & predicted)); fn = int(np.sum((labels == 1) & ~predicted))
    n0, n1 = tn+fp, tp+fn
    return {"balanced_accuracy": .5*(tp/n1+tn/n0), "macro_f1": .5*((2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)+(2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0)),
            "roc_auc": float((rankdata(scores)[labels == 1].sum()-n1*(n1+1)/2)/(n0*n1))}


def audit(prepared="outputs/prepared-stage3", output="outputs/runs-stage4", config_path="configs/stage4.json"):
    x, y, groups, rows, calibration, test, config, manifest = load_data(prepared, config_path)
    output = Path(output)
    run_info = read_json(output / "run.json")
    assert run_info["fingerprint"] == fingerprint(config, manifest)
    assert read_json(output / "summary.json")["status"] == "complete"
    errors, count = [], 0
    all_measured = []
    for target in sorted(set(groups)):
        folder = output / target
        check_completed(folder, run_info["fingerprint_id"])
        source, held = groups != target, groups == target
        suffix = held & test
        selection = read_json(folder / "selection.json")
        assert selection["fit_subjects"] == sorted(set(groups[source])) and not selection["target_used_for_selection"]
        assert selection["fit_window_ids"] == [r["window_id"] for r, use in zip(rows, source) if use]
        assert selection["test_window_ids"] == [r["window_id"] for r, use in zip(rows, suffix) if use]
        coverage = []
        for fold in selection["inner_folds"]:
            a, b = set(fold["fit_subjects"]), set(fold["validation_subjects"])
            assert not a & b and a | b == set(groups[source]); coverage += list(b)
        assert sorted(coverage) == sorted(set(groups[source]))
        models = joblib.load(folder / "models.joblib")
        pred_rows, recorded = read_csv(folder / "predictions.csv"), read_csv(folder / "metrics.csv")
        for key, method, budget in experiments(config):
            candidates = selection["candidates"][key]
            for candidate in candidates:
                assert sorted(r["subject"] for r in candidate["scores"]) == sorted(set(groups[source]))
                assert np.isclose(candidate["mean_subject_balanced_accuracy"], np.mean([r["balanced_accuracy"] for r in candidate["scores"]]), atol=1e-12)
            best = max(r["mean_subject_balanced_accuracy"] for r in candidates)
            assert selection["selected_C"][key] == next(r["C"] for r in candidates if best-r["mean_subject_balanced_accuracy"] <= 1e-12)
            model = models[key]
            np.testing.assert_allclose(model["scaler"].mean_, x[source].mean(0), atol=1e-12, rtol=0)
            np.testing.assert_allclose(model["scaler"].var_, x[source].var(0), atol=1e-12, rtol=0)
            if budget:
                ids = [r["window_id"] for r, use in zip(rows, held & calibration[budget]) if use]
                assert ids == selection["calibration_window_ids"][str(budget)] and not set(ids) & set(selection["test_window_ids"])
            with threadpool_limits(limits=1):
                expected = fit_adapter(model["scaler"].transform(x[source]), None if not budget else model["scaler"].transform(x[held & calibration[budget]]), method)
                for attr in ("offset", "transform_matrix", "destination"):
                    np.testing.assert_allclose(getattr(model["adapter"], attr), getattr(expected, attr), atol=1e-12, rtol=0)
                probability = predict(model, x[suffix])
            current = [r for r in pred_rows if r["experiment"] == key]
            assert [r["window_id"] for r in current] == selection["test_window_ids"]
            assert np.array_equal([int(r["label"]) for r in current], y[suffix])
            errors.append(float(np.max(abs(probability - [float(r["score"]) for r in current]))))
            measured = manual_metrics(y[suffix], probability)
            old = next(r for r in recorded if r["experiment"] == key)
            for name, value in measured.items():
                assert np.isclose(value, float(old[name]), atol=1e-12, rtol=0)
            assert model["classifier"].C == selection["selected_C"][key]
            assert np.array_equal([int(r["predicted"]) for r in current], probability >= .5)
            all_measured.append({"experiment": key, **measured}); count += 1
    assert count == 252 and max(errors) <= 1e-12
    summary = read_json(output / "summary.json")
    for entry in summary["experiments"]:
        for name in ("balanced_accuracy", "macro_f1", "roc_auc"):
            assert np.isclose(entry[f"mean_subject_{name}"], np.mean([r[name] for r in all_measured if r["experiment"] == entry["experiment"]]), atol=1e-12, rtol=0)
    result = {"passed": True, "models": count, "predictions": 9072, "max_replay_error": max(errors), "source_and_prefix_statistics_verified": True,
              "inner_subject_isolation_and_selection_verified": True, "common_suffix_and_calibration_disjointness_verified": True,
              "fingerprint_id": run_info["fingerprint_id"]}
    write_json(output / "audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "audit"])
    parser.add_argument("--prepared", default="outputs/prepared-stage3")
    parser.add_argument("--output", default="outputs/runs-stage4")
    parser.add_argument("--config-path", default="configs/stage4.json")
    parser.add_argument("--subjects", nargs="+")
    args = vars(parser.parse_args()); command = args.pop("command")
    if command == "audit": args.pop("subjects")
    globals()[command](**args)
