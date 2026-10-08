"""Audit saved runs independently of the experiment metric implementation."""
from collections import Counter
from pathlib import Path
import csv
import json

import joblib
import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def independently_score(rows):
    y = np.array([int(r["label"]) for r in rows])
    pred = np.array([int(r["prediction"]) for r in rows])
    score = np.array([float(r["score"]) for r in rows])
    tp, tn = np.sum((y == 1) & (pred == 1)), np.sum((y == 0) & (pred == 0))
    fp, fn = np.sum((y == 0) & (pred == 1)), np.sum((y == 1) & (pred == 0))
    f1_pos = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
    f1_neg = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.
    positive, negative = np.sum(y == 1), np.sum(y == 0)
    auc = (np.sum(rankdata(score)[y == 1]) - positive * (positive + 1) / 2) / (positive * negative)
    return {"balanced_accuracy": .5 * (tp / positive + tn / negative),
            "accuracy": (tp + tn) / len(y), "macro_f1": .5 * (f1_pos + f1_neg),
            "positive_class_f1": f1_pos, "roc_auc": auc, "n_windows": len(y)}


def main():
    root = Path(__file__).resolve().parents[1]
    runs, prepared = root / "outputs/runs-stage2", root / "outputs/prepared-stage2"
    analysis = root / "outputs/analysis-stage2"
    config = read_json(prepared / "config.json")
    assert read_json(runs / "batch_status.json")["status"] == "complete"
    all_subjects = {f"Subject{i:02d}" for i in range(36)}
    experiments = read_json(runs / "batch.json")["experiments"]
    total_predictions = total_inner = total_models = smoke_predictions = 0
    scores_max_error = 0.
    outputs = []
    pool_cache = {}
    for spec in experiments:
        folder = runs / spec["id"]
        metrics, predictions = read_csv(folder / "subject_metrics.csv"), read_csv(folder / "predictions.csv")
        assert len(metrics) == 36 and {r["subject"] for r in metrics} == all_subjects
        assert all(r["status"] == "ok" for r in metrics)
        assert len({r["window_id"] for r in predictions}) == len(predictions)
        variant = spec["variant"]
        if variant not in pool_cache:
            with np.load(prepared / variant / "arrays.npz", allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            roles = ("train", "val", "test") if variant == "s1" else ("train", "eval")
            pools = {role: read_csv(prepared / variant / f"{role}_windows.csv") for role in roles}
            pool_cache[variant] = arrays, pools
        arrays, pools = pool_cache[variant]
        role = "test" if spec["protocol"] == "S1" else "eval"
        summary = read_json(folder / "summary.json")
        recomputed, selections = [], []
        for measured in metrics:
            subject = measured["subject"]
            rows = [r for r in predictions if r["outer_subject"] == subject]
            mask = arrays[f"groups_{role}"] == subject
            train = arrays["groups_train"] == subject if spec["protocol"] == "S1" else arrays["groups_train"] != subject
            assert all(r["subject"] == subject for r in rows)
            assert Counter(int(r["label"]) for r in rows)[0] == Counter(int(r["label"]) for r in rows)[1]
            expected_ids = [r["window_id"] for r in pools[role] if r["subject"] == subject]
            assert [r["window_id"] for r in rows] == expected_ids
            numbers = independently_score(rows)
            recomputed.append(numbers)
            for key, value in numbers.items():
                assert np.isclose(float(measured[key]), value, atol=1e-12, rtol=0), (spec["id"], subject, key)
            selection = read_json(folder / f"{subject}_selection.json")
            candidates = selection["candidates"]
            best = max(c["mean_subject_balanced_accuracy"] for c in candidates)
            selected = next(c["params"] for c in candidates if c["mean_subject_balanced_accuracy"] >= best - 1e-12)
            assert selected == selection["selected_params"]
            selections.append(json.dumps(selected, sort_keys=True))
            if spec["protocol"] == "S2":
                sources = all_subjects - {subject}
                assert set(selection["fit_subjects"]) == sources
                for candidate in candidates:
                    seen, subject_scores = [], []
                    for fold in candidate["folds"]:
                        fit_ids, val_ids = set(fold["fit_subjects"]), set(fold["validation_subjects"])
                        assert fit_ids.isdisjoint(val_ids) and fit_ids | val_ids == sources
                        seen.extend(val_ids)
                        assert {s["subject"] for s in fold["scores"]} == val_ids
                        subject_scores.extend(s["balanced_accuracy"] for s in fold["scores"])
                        total_inner += 1
                    assert len(seen) == 35 and set(seen) == sources
                    assert np.isclose(np.mean(subject_scores), candidate["mean_subject_balanced_accuracy"], atol=1e-12, rtol=0)
            else:
                assert selection["final_fit"] == "train_only" and not selection["validation_merged_into_fit"]
                intervals = [r for role_rows in pools.values() for r in role_rows if r["subject"] == subject]
                for i, left in enumerate(intervals):
                    for right in intervals[i+1:]:
                        if left["recording_id"] == right["recording_id"] and left["split"] != right["split"]:
                            assert max(int(left["start_sample"]), int(right["start_sample"])) >= min(int(left["stop_sample"]), int(right["stop_sample"]))
            # Locally generated model files only; inspect the actual fit and reproduce saved scores.
            model = joblib.load(folder / f"{subject}_model.joblib")
            np.testing.assert_allclose(model["scaler"].mean_, arrays["X_train"][train].mean(axis=0), rtol=1e-12, atol=1e-12)
            with threadpool_limits(limits=1):
                reproduced = model.decision_function(arrays[f"X_{role}"][mask]) if spec["kind"] == "svm" else model.predict_proba(arrays[f"X_{role}"][mask])[:, 1]
            saved = np.array([float(r["score"]) for r in rows])
            error = float(np.max(np.abs(saved - reproduced)))
            assert error <= 1e-12
            scores_max_error = max(scores_max_error, error)
            total_models += 1
        for key in ("balanced_accuracy", "accuracy", "macro_f1", "positive_class_f1", "roc_auc"):
            assert np.isclose(summary[f"mean_subject_{key}"], np.mean([r[key] for r in recomputed]), rtol=0, atol=1e-12)
        smoke_root = "smoke-stage2-s1fix" if spec["protocol"] == "S1" else "smoke-stage2"
        smoke = read_csv(root / "outputs" / smoke_root / spec["id"] / "predictions.csv")
        lookup = {r["window_id"]: r for r in predictions}
        for row in smoke:
            assert float(row["score"]) == float(lookup[row["window_id"]]["score"])
            smoke_predictions += 1
        total_predictions += len(predictions)
        outputs.append({"experiment": spec["id"], "folds": 36, "predictions": len(predictions),
                        "mean_ba": summary["mean_subject_balanced_accuracy"], "selected_params_counts": dict(Counter(selections))})
    pca_checks = []
    base_arrays, _ = pool_cache["de2"]
    for target in config["pca_targets"]:
        folder = analysis / f"pca_{target}"
        summary = read_json(folder / "summary.json")
        assert set(summary["fit_subjects"]) == all_subjects - {target}
        model = joblib.load(folder / "source_pca_model.joblib")
        sources = base_arrays["groups_eval"] != target
        np.testing.assert_allclose(model["scaler"].mean_, base_arrays["X_eval"][sources].mean(axis=0), atol=1e-12, rtol=1e-12)
        assert model["pca"].n_samples_ == int(sources.sum()) == 2100
        coords = read_csv(folder / "coordinates.csv")
        np.testing.assert_allclose(model.transform(base_arrays["X_eval"]), [[float(r["pc1"]),float(r["pc2"])] for r in coords], atol=1e-12, rtol=0)
        pca_checks.append(target)
    report = {"passed": True, "method": "Independent confusion counts and rank-AUC; source-scaler means and score reproduction from all locally generated saved models; no retraining.",
              "experiments": len(experiments), "models_and_outer_folds_verified": total_models,
              "prediction_rows_verified": total_predictions, "inner_candidate_folds_verified": total_inner,
              "smoke_scores_exactly_reproduced": smoke_predictions, "saved_model_score_max_error": scores_max_error,
              "pca_source_fit_and_coordinates_verified": pca_checks, "details": outputs,
              "note": "Repeated windows across experiments are not additional independent participants; there are 36 participants."}
    (root / "outputs/stage2-independent-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
