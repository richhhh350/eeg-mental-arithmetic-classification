"""Numeric-only target calibration, source-only model selection."""
from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def matrix(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("Expected finite nonempty numeric feature matrix")
    return x


def symmetric_power(cov, exponent):
    cov = matrix(cov)
    if cov.shape[0] != cov.shape[1] or not np.allclose(cov, cov.T, atol=1e-12):
        raise ValueError("Expected symmetric covariance")
    eigenvalues, vectors = np.linalg.eigh(cov)
    if eigenvalues.min() <= 0:
        raise ValueError("Covariance must be positive definite")
    return (vectors * eigenvalues**exponent) @ vectors.T


def covariance(x):
    x = matrix(x)
    if len(x) < 2:
        raise ValueError("At least two calibration samples required")
    estimator = LedoitWolf(store_precision=False).fit(x)
    cov = estimator.covariance_.copy()
    ridge = 1e-6 * max(float(np.trace(cov) / cov.shape[0]), 1e-12)
    cov += np.eye(cov.shape[0]) * ridge
    return estimator.location_, cov, float(estimator.shrinkage_)


@dataclass
class Adapter:
    offset: np.ndarray
    transform_matrix: np.ndarray
    destination: np.ndarray
    diagnostics: dict

    def transform(self, x):
        x = matrix(x)
        if x.shape[1] != len(self.offset):
            raise ValueError("Feature count changed")
        return (x - self.offset) @ self.transform_matrix + self.destination


def fit_adapter(source_z, calibration_z, method):
    """No labels, record IDs, test data or condition metadata are accepted."""
    source_z = matrix(source_z)
    d = source_z.shape[1]
    if method == "none":
        if calibration_z is not None:
            raise ValueError("Zero calibration must not receive target data")
        return Adapter(np.zeros(d), np.eye(d), np.zeros(d), {"method": method, "calibration_windows": 0})
    if method not in ("standardize", "coral"):
        raise ValueError("Unknown calibration method")
    calibration_z = matrix(calibration_z)
    if calibration_z.shape[1] != d:
        raise ValueError("Calibration dimensions differ")
    sm, sc, ss = covariance(source_z)
    tm, tc, ts = covariance(calibration_z)
    if method == "standardize":
        sc, tc = np.diag(np.diag(sc)), np.diag(np.diag(tc))
    transform = symmetric_power(tc, -.5) @ symmetric_power(sc, .5)
    error = np.linalg.norm(transform.T @ tc @ transform - sc) / np.linalg.norm(sc)
    if error > 1e-7 or not np.isfinite(transform).all():
        raise FloatingPointError("Covariance transport check failed")
    return Adapter(tm, transform, sm, {
        "method": method, "calibration_windows": len(calibration_z),
        "source_shrinkage": ss, "target_shrinkage": ts,
        "source_condition_number": float(np.linalg.cond(sc)),
        "target_condition_number": float(np.linalg.cond(tc)),
        "regularized_covariance_transport_relative_error": float(error),
        "target_mean_shift_norm": float(np.linalg.norm(tm - sm)),
    })


def fit_lr(z, y, C):
    z, y = matrix(z), np.asarray(y)
    if y.shape != (len(z),) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Both source labels required")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return LogisticRegression(C=C, solver="lbfgs", max_iter=3000, random_state=42).fit(z, y)


def ba(y, probability):
    y = np.asarray(y)
    pred = np.asarray(probability) >= .5
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both evaluation labels required")
    return float(np.mean([np.mean(pred[y == label] == label) for label in (0, 1)]))


def experiments(config):
    return [("none_0", "none", 0)] + [(f"{m}_{b}", m, b) for m in ("standardize", "coral") for b in config["budgets_seconds_per_record"]]


def window_masks(rows, config):
    starts = np.array([int(r["start_sample"]) / float(r["fs_hz"]) for r in rows])
    stops = np.array([int(r["stop_sample"]) / float(r["fs_hz"]) for r in rows])
    if not np.allclose(stops - starts, config["window_seconds"]):
        raise ValueError("Window durations changed")
    test = (starts >= config["test_start_seconds"]) & (stops <= config["test_stop_seconds"])
    calibration = {b: (starts >= 0) & (stops <= b) for b in config["budgets_seconds_per_record"]}
    if any(np.any(mask & test) for mask in calibration.values()):
        raise ValueError("Calibration and evaluation overlap")
    return calibration, test


def select_source(x, y, groups, calibration, test, config):
    """Caller supplies ONLY outer-source arrays; inner targets stay out of fits."""
    x, y, groups = matrix(x), np.asarray(y), np.asarray(groups)
    if y.shape != (len(x),) or groups.shape != y.shape or test.shape != y.shape:
        raise ValueError("Source pool lengths differ")
    specs = experiments(config)
    records = {key: [{"C": c, "scores": []} for c in config["C_candidates"]] for key, _, _ in specs}
    splits = []
    for index, (fit, val) in enumerate(GroupKFold(config["inner_folds"]).split(x, y, groups)):
        fit_groups, val_groups = sorted(set(groups[fit])), sorted(set(groups[val]))
        assert not set(fit_groups) & set(val_groups)
        scaler = StandardScaler().fit(x[fit])
        source_z = scaler.transform(x[fit])
        # Cache only transformations independent of candidate C and all labels.
        pools = {}
        for target in val_groups:
            held = groups == target
            suffix = held & test
            target_z = scaler.transform(x[suffix])
            for key, method, budget in specs:
                prefix = None if method == "none" else scaler.transform(x[held & calibration[budget]])
                adapter = fit_adapter(source_z, prefix, method)
                pools[target, key] = adapter.transform(target_z), y[suffix]
        for ci, c in enumerate(config["C_candidates"]):
            model = fit_lr(source_z, y[fit], c)
            for (target, key), (z, labels) in pools.items():
                records[key][ci]["scores"].append({"fold": index, "subject": str(target), "balanced_accuracy": ba(labels, model.predict_proba(z)[:, 1])})
        splits.append({"fold": index, "fit_subjects": fit_groups, "validation_subjects": val_groups})
    selected = {}
    for key, candidates in records.items():
        for candidate in candidates:
            assert sorted(r["subject"] for r in candidate["scores"]) == sorted(set(groups))
            candidate["mean_subject_balanced_accuracy"] = float(np.mean([r["balanced_accuracy"] for r in candidate["scores"]]))
        best = max(r["mean_subject_balanced_accuracy"] for r in candidates)
        selected[key] = next(r["C"] for r in candidates if best - r["mean_subject_balanced_accuracy"] <= 1e-12)
    return {"selected_C": selected, "candidates": records, "inner_folds": splits,
            "fit_subjects": sorted(set(groups)), "target_used_for_selection": False}


def fit_final(source_x, source_y, target_calibration_x, method, C):
    """No test matrix or target task labels are exposed during final fitting."""
    scaler = StandardScaler().fit(matrix(source_x))
    source_z = scaler.transform(source_x)
    classifier = fit_lr(source_z, source_y, C)
    prefix_z = None if target_calibration_x is None else scaler.transform(matrix(target_calibration_x))
    adapter = fit_adapter(source_z, prefix_z, method)
    return {"scaler": scaler, "classifier": classifier, "adapter": adapter}


def predict(model, x):
    z = model["adapter"].transform(model["scaler"].transform(matrix(x)))
    return model["classifier"].predict_proba(z)[:, 1]
