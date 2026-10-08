"""Source-isolated model selection for Stage 2 traditional baselines."""
from __future__ import annotations

from copy import deepcopy
import warnings

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_model(kind, params, seed=42):
    """Build unfitted training-only preprocessing; SVM scores remain margins."""
    allowed = {"lr": {"C"}, "lda": {"shrinkage"}, "svm": {"C", "gamma"}, "dummy": set()}
    if kind not in allowed or not isinstance(params, dict) or set(params) != allowed[kind]:
        raise ValueError(f"Invalid model kind or parameters: {kind}/{params}")
    if kind == "lr":
        classifier = LogisticRegression(C=float(params["C"]), solver="lbfgs", max_iter=3000, random_state=seed)
    elif kind == "lda":
        classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=params["shrinkage"])
    elif kind == "svm":
        # sklearn 1.9 deprecates explicitly passing probability=False. Its default
        # still disables calibration; _fit verifies the effective setting.
        classifier = SVC(C=float(params["C"]), gamma=params["gamma"], kernel="rbf", random_state=seed)
    else:
        classifier = DummyClassifier(strategy="prior", random_state=seed)
    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


def _fit(model, x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        fitted = model.fit(x, y)
    classifier = fitted["classifier"]
    if isinstance(classifier, SVC) and getattr(classifier, "_effective_probability", classifier.probability) is not False:
        raise AssertionError("SVM probability calibration must remain disabled")
    return fitted


def predict_scores(model, x):
    """Return explicit binary decisions, continuous scores, and score semantics."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Invalid prediction features")
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Model must represent classes 0 and 1 in that order")
    if isinstance(model["classifier"], SVC):
        score = np.asarray(model.decision_function(x), dtype=float)
        threshold, score_kind = 0.0, "decision_margin"
    else:
        score = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
        threshold, score_kind = 0.5, "probability"
        if np.any((score < 0) | (score > 1)):
            raise ValueError("Invalid prediction probability")
    if score.shape != (len(x),) or not np.isfinite(score).all():
        raise ValueError("Invalid prediction scores")
    return (score >= threshold).astype(int), score, score_kind


def score_metrics(y, pred, score):
    """Evaluate either probabilities or margins without treating margins as probabilities."""
    y, pred, score = np.asarray(y), np.asarray(pred), np.asarray(score, dtype=float)
    if y.ndim != 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError("Both conditions are required for evaluation")
    if pred.shape != y.shape or not set(np.unique(pred)) <= {0, 1}:
        raise ValueError("Invalid binary predictions")
    if score.shape != y.shape or not np.isfinite(score).all():
        raise ValueError("Invalid prediction scores")
    return {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "positive_class_f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, score)), "n_windows": int(len(y))}


def _validate_pool(x, y, groups=None):
    x, y = np.asarray(x, dtype=float), np.asarray(y)
    if x.ndim != 2 or x.shape[1] == 0 or y.ndim != 1 or len(x) != len(y) or not len(x):
        raise ValueError("Inconsistent sample dimensions")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite features")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both conditions are required for fitting and validation")
    if groups is None:
        return x, y
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != len(y):
        raise ValueError("Inconsistent group dimensions")
    # Subject labels are identifiers, never model features.
    if groups.dtype.kind not in "US" or np.any(groups == ""):
        raise ValueError("Subject identifiers must be nonempty strings")
    for subject in np.unique(groups):
        if set(np.unique(y[groups == subject])) != {0, 1}:
            raise ValueError(f"Subject {subject} lacks a condition")
    return x, y, groups


def _candidate_list(candidates):
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ValueError("At least one candidate is required")
    if not all(isinstance(params, dict) for params in candidates):
        raise ValueError("Candidates must be parameter dictionaries")
    return deepcopy(list(candidates))


def _choose(records):
    best = max(record["mean_subject_balanced_accuracy"] for record in records)
    return next(record for record in records if record["mean_subject_balanced_accuracy"] >= best - 1e-12)


def fit_grouped(train_x, train_y, train_groups, validation_x, validation_y,
                validation_groups, kind, candidates, inner_folds=3, seed=42):
    """Select using disjoint source subjects; validation uses its own window pool.

    Both pools must contain the same source participants. In an inner fold,
    train-pool windows from validation participants are excluded entirely, even
    when train windows overlap. The outer held-out participant is not an input.
    """
    train_x, train_y, train_groups = _validate_pool(train_x, train_y, train_groups)
    validation_x, validation_y, validation_groups = _validate_pool(validation_x, validation_y, validation_groups)
    subjects = sorted(np.unique(train_groups).tolist())
    if subjects != sorted(np.unique(validation_groups).tolist()):
        raise ValueError("Training and validation pools must contain the same source subjects")
    if train_x.shape[1] != validation_x.shape[1]:
        raise ValueError("Training and validation feature dimensions differ")
    if isinstance(inner_folds, bool) or not isinstance(inner_folds, (int, np.integer)) or not 2 <= inner_folds <= len(subjects):
        raise ValueError("Invalid number of inner subject folds")
    params_list = _candidate_list(candidates)
    # Fold geometry is determined solely by the fixed evaluation pool. This
    # keeps the overlap ablation's validation subjects/windows comparable.
    folds = list(GroupKFold(n_splits=inner_folds).split(validation_x, validation_y, validation_groups))
    records = []
    for params in params_list:
        fold_records, scores_all = [], []
        for fold_index, (fit_eval_indices, validation_indices) in enumerate(folds):
            fit_subjects = sorted(np.unique(validation_groups[fit_eval_indices]).tolist())
            val_subjects = sorted(np.unique(validation_groups[validation_indices]).tolist())
            fit_indices = np.isin(train_groups, fit_subjects)
            if set(fit_subjects) & set(val_subjects) or not fit_indices.any():
                raise AssertionError("Invalid inner participant isolation")
            model = _fit(make_model(kind, params, seed), train_x[fit_indices], train_y[fit_indices])
            val_x, val_y = validation_x[validation_indices], validation_y[validation_indices]
            val_groups = validation_groups[validation_indices]
            pred, score, _ = predict_scores(model, val_x)
            scores = [{"subject": subject, **score_metrics(val_y[val_groups == subject], pred[val_groups == subject], score[val_groups == subject])}
                      for subject in val_subjects]
            scores_all.extend(scores)
            fold_records.append({"fold": fold_index, "fit_subjects": fit_subjects,
                                 "validation_subjects": val_subjects, "scores": scores})
        records.append({"params": params, "mean_subject_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in scores_all])),
                        "folds": fold_records})
    selected = _choose(records)
    model = _fit(make_model(kind, selected["params"], seed), train_x, train_y)
    return model, {"selected_params": deepcopy(selected["params"]), "candidates": records,
                   "fit_subjects": subjects, "scaler_ddof": 0, "target_used_for_selection": False,
                   "validation_unit": "equal_subject", "inner_fold_pool": "validation"}


def fit_temporal(train_x, train_y, validation_x, validation_y, kind, candidates, seed=42):
    """Select on the validation block and keep the selected train-only fit."""
    train_x, train_y = _validate_pool(train_x, train_y)
    validation_x, validation_y = _validate_pool(validation_x, validation_y)
    if train_x.shape[1] != validation_x.shape[1]:
        raise ValueError("Training and validation feature dimensions differ")
    records, models = [], []
    for params in _candidate_list(candidates):
        model = _fit(make_model(kind, params, seed), train_x, train_y)
        pred, score, _ = predict_scores(model, validation_x)
        measured = score_metrics(validation_y, pred, score)
        records.append({"params": params, "mean_subject_balanced_accuracy": measured["balanced_accuracy"],
                        "validation_metrics": measured})
        models.append(model)
    selected = _choose(records)
    index = next(i for i, record in enumerate(records) if record is selected)
    return models[index], {"selected_params": deepcopy(selected["params"]), "candidates": records,
                           "final_fit": "train_only", "fit_windows": len(train_y),
                           "validation_windows": len(validation_y), "scaler_ddof": 0,
                           "test_used_for_selection": False, "validation_merged_into_fit": False}
