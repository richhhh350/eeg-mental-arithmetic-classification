"""Nested LOSO with all fitted transforms confined to source participants."""
from __future__ import annotations

import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def metrics(y, probability):
    y = np.asarray(y)
    probability = np.asarray(probability, dtype=float)
    if y.ndim != 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError("Both conditions are required for evaluation")
    if probability.shape != y.shape or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid prediction probabilities")
    pred = (probability >= 0.5).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "positive_class_f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "n_windows": int(len(y)),
    }


def subject_scores(y, probability, groups):
    y, probability, groups = map(np.asarray, (y, probability, groups))
    return [{"subject": str(g), **metrics(y[groups == g], probability[groups == g])}
            for g in sorted(np.unique(groups))]


def make_model(c, config):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=float(c), solver=config["lr_solver"],
                                         max_iter=config["lr_max_iter"],
                                         random_state=config["seed"])),
    ])


def _fit(model, x, y):
    # A nonconverged fit must not be silently published as a valid result.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return model.fit(x, y)


def fit_source_model(x, y, groups, config):
    """No target X/y parameter: selection, scaling and final fit see sources only."""
    x, y, groups = map(np.asarray, (x, y, groups))
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(groups):
        raise ValueError("Inconsistent sample dimensions")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite source features")
    for g in np.unique(groups):
        if set(np.unique(y[groups == g])) != {0, 1}:
            raise ValueError(f"Source {g} lacks a condition")
    cv = GroupKFold(n_splits=config["inner_group_folds"])
    splits = list(cv.split(x, y, groups))
    candidates = []
    for c in sorted(config["lr_C"]):
        fold_details, all_subject_scores = [], []
        for index, (train, val) in enumerate(splits):
            train_groups = sorted(np.unique(groups[train]).tolist())
            val_groups = sorted(np.unique(groups[val]).tolist())
            if set(train_groups) & set(val_groups):
                raise AssertionError("Inner participant leakage")
            model = _fit(make_model(c, config), x[train], y[train])
            scores = subject_scores(y[val], model.predict_proba(x[val])[:, 1], groups[val])
            all_subject_scores.extend(scores)
            fold_details.append({"fold": index, "fit_subjects": train_groups,
                                 "validation_subjects": val_groups, "scores": scores})
        mean_ba = float(np.mean([s["balanced_accuracy"] for s in all_subject_scores]))
        candidates.append({"C": float(c), "mean_subject_balanced_accuracy": mean_ba,
                           "folds": fold_details})
    best_score = max(c["mean_subject_balanced_accuracy"] for c in candidates)
    chosen = next(c for c in candidates if c["mean_subject_balanced_accuracy"] >= best_score - 1e-12)
    model = _fit(make_model(chosen["C"], config), x, y)
    return model, {"selected_C": chosen["C"], "candidates": candidates,
                   "fit_subjects": sorted(np.unique(groups).tolist()),
                   "scaler_ddof": 0, "target_used_for_selection": False}
