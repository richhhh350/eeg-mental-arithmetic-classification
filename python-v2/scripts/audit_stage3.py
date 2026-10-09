"""Independent count/rank metric audit and saved-model replay for stage3."""
from __future__ import annotations
import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from eeg_v2.provenance import read_csv, read_json, write_json
from eeg_v2.stage3_data import load_stage3


def manual_metrics(y, predicted, score):
    y, predicted, score = np.asarray(y), np.asarray(predicted), np.asarray(score)
    tp = int(np.sum((y == 1) & (predicted == 1)))
    tn = int(np.sum((y == 0) & (predicted == 0)))
    fp = int(np.sum((y == 0) & (predicted == 1)))
    fn = int(np.sum((y == 1) & (predicted == 0)))
    f1_positive = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_negative = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    n1, n0 = int(np.sum(y == 1)), int(np.sum(y == 0))
    auc = (rankdata(score)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return {"balanced_accuracy": (tp / n1 + tn / n0) / 2, "accuracy": (tp + tn) / len(y),
            "macro_f1": (f1_positive + f1_negative) / 2, "positive_class_f1": f1_positive, "roc_auc": float(auc)}


def check_scores(rows, expected_y, recorded):
    y = np.array([int(r["label"]) for r in rows])
    pred = np.array([int(r["predicted"]) for r in rows])
    score = np.array([float(r["score"]) for r in rows])
    assert np.array_equal(y, expected_y)
    threshold = 0 if rows[0]["score_kind"] == "decision_margin" else 0.5
    assert np.array_equal(pred, (score >= threshold).astype(int))
    measured = manual_metrics(y, pred, score)
    for key, value in measured.items():
        assert np.isclose(value, float(recorded[key]), atol=1e-12, rtol=0), key
    return score, measured


def audit_bridge(prepared, bridge, stage1):
    _, features, rows, config, manifest = load_stage3(prepared)
    bridge = Path(bridge)
    subjects = sorted(set(features["groups"]))
    assert len(subjects) == 36
    assert read_json(bridge / "summary.json")["status"] == "complete"
    predictions_checked, models, max_error = 0, 0, 0.0
    for spec in config["bridges"]:
        folder = bridge / spec["id"]
        predictions = read_csv(folder / "predictions.csv")
        metrics = read_csv(folder / "subject_metrics.csv")
        assert {r["subject"] for r in metrics} == set(subjects)
        assert len(metrics) == 36 and len(predictions) == 2160
        x = features[spec["feature"]]
        for target in subjects:
            source, test = features["groups"] != target, features["groups"] == target
            pred_rows = [r for r in predictions if r["subject"] == target]
            assert [r["window_id"] for r in pred_rows] == [r["window_id"] for r, m in zip(rows, test) if m]
            score, measured = check_scores(pred_rows, features["y"][test], next(r for r in metrics if r["subject"] == target))
            model = joblib.load(folder / f"{target}_model.joblib")
            np.testing.assert_allclose(model.named_steps["scaler"].mean_, x[source].mean(0), atol=1e-12, rtol=0)
            with threadpool_limits(limits=1):
                replay = model.decision_function(x[test]) if spec["kind"] == "svm" else model.predict_proba(x[test])[:, 1]
            error = float(np.max(abs(score - replay)))
            assert error <= 1e-12
            max_error = max(max_error, error)
            selection = read_json(folder / f"{target}_selection.json")
            assert set(selection["fit_subjects"]) == set(features["groups"][source])
            assert selection["fit_window_ids"] == [r["window_id"] for r, m in zip(rows, source) if m]
            best = max(r["mean_subject_balanced_accuracy"] for r in selection["candidates"])
            first = next(r for r in selection["candidates"] if best - r["mean_subject_balanced_accuracy"] <= 1e-12)
            assert first["params"] == selection["selected_params"]
            for candidate in selection["candidates"]:
                validation = []
                scores = []
                for fold in candidate["folds"]:
                    a, b = set(fold["fit_subjects"]), set(fold["validation_subjects"])
                    assert not a & b and a | b == set(selection["fit_subjects"])
                    validation.extend(b)
                    scores.extend(r["balanced_accuracy"] for r in fold["scores"])
                assert sorted(validation) == sorted(selection["fit_subjects"])
                assert np.isclose(np.mean(scores), candidate["mean_subject_balanced_accuracy"], atol=1e-12, rtol=0)
            models += 1
            predictions_checked += len(pred_rows)
        summary = read_json(folder / "summary.json")
        assert np.isclose(np.mean([float(r["balanced_accuracy"]) for r in metrics]), summary["mean_subject_balanced_accuracy"], atol=1e-12, rtol=0)
    a = {r["window_id"]: float(r["probability_arithmetic"]) for r in read_csv(Path(stage1) / "predictions.csv")}
    b = {r["window_id"]: float(r["score"]) for r in read_csv(bridge / "s2_lr500_reference/predictions.csv")}
    assert a.keys() == b.keys() and max(abs(a[k] - b[k]) for k in a) == 0
    return {"passed": True, "traditional_models": models, "traditional_predictions": predictions_checked,
            "traditional_max_score_error": max_error, "stage1_reference_max_probability_error": 0.0,
            "dataset_id": manifest["dataset_id"]}


def audit(prepared, bridge, runs, stage1, output, bridge_only=False):
    result = audit_bridge(prepared, bridge, stage1)
    if not bridge_only:
        from eeg_v2.stage3 import verify
        verify(prepared, runs)
        runs = Path(runs)
        _, features, _, config, _ = load_stage3(prepared)
        summary = read_json(runs / "summary.json")
        assert summary["status"] == "complete" and summary["completed_folds"] == 108
        assert sorted(summary["seeds"]) == sorted(config["seeds"]) and len(summary["subjects"]) == 36
        count = 0
        metrics = []
        for seed in config["seeds"]:
            for target in summary["subjects"]:
                folder = runs / f"seed{seed}" / target
                recorded = read_json(folder / "metrics.json")
                pred_rows = read_csv(folder / "predictions.csv")
                _, measured = check_scores(pred_rows, features["y"][features["groups"] == target], recorded)
                metrics.append({"subject": target, "seed": seed, **measured})
                count += len(pred_rows)
        for seed_summary in summary["by_seed"]:
            for name in ("balanced_accuracy", "accuracy", "macro_f1", "positive_class_f1", "roc_auc"):
                assert np.isclose(seed_summary[f"mean_subject_{name}"], np.mean([r[name] for r in metrics if r["seed"] == seed_summary["seed"]]), atol=1e-12, rtol=0)
        for name in ("balanced_accuracy", "accuracy", "macro_f1", "positive_class_f1", "roc_auc"):
            expected = np.mean([np.mean([r[name] for r in metrics if r["subject"] == s]) for s in summary["subjects"]])
            assert np.isclose(summary[f"mean_subject_seed_{name}"], expected, atol=1e-12, rtol=0)
        result.update(eegnet_models=108, eegnet_predictions=count, eegnet_verification=read_json(runs / "verification.json"),
                      method="Manual confusion counts and rank-AUC; source-scaler checks, grouped selection checks, saved-model replay and independent aggregation. No retraining.")
    write_json(output, result)
    print(result, flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", default="outputs/prepared-stage3")
    p.add_argument("--bridge", default="outputs/bridge-stage3")
    p.add_argument("--runs", default="outputs/runs-stage3")
    p.add_argument("--stage1", default="outputs/loso-stage1")
    p.add_argument("--output", default="outputs/stage3-independent-audit.json")
    p.add_argument("--bridge-only", action="store_true")
    audit(**vars(p.parse_args()))
