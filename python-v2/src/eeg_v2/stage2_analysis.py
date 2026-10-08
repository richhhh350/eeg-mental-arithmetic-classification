"""Exploratory source-fitted visualizations and paired participant descriptions.

These summaries do not select models. Bootstrap intervals are conditional on
the observed participant predictions, not independent repetitions of LOSO.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib import colormaps
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .provenance import write_csv, write_json


def _cosine(left, right):
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= np.finfo(float).eps:
        return None
    return float(np.clip(np.dot(left, right) / denominator, -1, 1))


def _new_figure():
    figure = Figure(figsize=(12, 7), layout="constrained")
    FigureCanvasAgg(figure)
    return figure, figure.subplots()


def source_pca(X, groups, y, target, output_dir, seed=42):
    """Fit scaler and PCA on sources; transform the target without fitting.

    Labels are used only for the two plots and descriptive condition centroids.
    Full-dimensional descriptors use the source-fitted standardization rather
    than distances in the potentially misleading two-dimensional projection.
    """
    X = np.asarray(X, dtype=float)
    groups, y = np.asarray(groups), np.asarray(y)
    if X.ndim != 2 or groups.ndim != 1 or y.ndim != 1 or len(X) != len(groups) or len(X) != len(y):
        raise ValueError("Inconsistent sample dimensions")
    if X.shape[1] < 2 or not np.isfinite(X).all():
        raise ValueError("PCA requires at least two finite feature columns")
    if any(not isinstance(g, (str, np.str_)) or not str(g).strip() for g in groups):
        raise ValueError("Participant IDs must be nonempty strings")
    subjects = sorted(np.unique(groups).tolist())
    if target not in subjects or len(subjects) < 2:
        raise ValueError("A present target and at least one source subject are required")
    for subject in subjects:
        if set(np.unique(y[groups == subject])) != {0, 1}:
            raise ValueError(f"Participant {subject} lacks a condition")
    source_mask = groups != target
    if source_mask.sum() < 3:
        raise ValueError("At least three source windows are required")
    if not np.any(np.var(X[source_mask], axis=0) > 0):
        raise ValueError("Source features have no variation")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2, svd_solver="full", random_state=seed)),
    ])
    model.fit(X[source_mask])
    scaler, pca = model.named_steps["scaler"], model.named_steps["pca"]
    standardized = scaler.transform(X)
    coordinates = model.transform(X)
    fit_subjects = [s for s in subjects if s != target]
    condition_centres = {
        subject: [standardized[(groups == subject) & (y == label)].mean(axis=0) for label in (0, 1)]
        for subject in subjects
    }
    differences = {subject: centres[1] - centres[0] for subject, centres in condition_centres.items()}
    # Equal participant weighting, regardless of any differing window counts.
    source_direction = np.mean([differences[s] for s in fit_subjects], axis=0)
    source_centroid = np.mean([standardized[groups == s].mean(axis=0) for s in fit_subjects], axis=0)
    direction_rows, full_dimensions = [], []
    for subject in subjects:
        mask = groups == subject
        centres = [coordinates[mask & (y == label)].mean(axis=0) for label in (0, 1)]
        delta = centres[1] - centres[0]
        direction_rows.append({
            "subject": subject, "role": "target" if subject == target else "source",
            "rest_pc1": float(centres[0][0]), "rest_pc2": float(centres[0][1]),
            "arithmetic_pc1": float(centres[1][0]), "arithmetic_pc2": float(centres[1][1]),
            "delta_pc1": float(delta[0]), "delta_pc2": float(delta[1]),
            "full_standardized_condition_distance": float(np.linalg.norm(differences[subject])),
            "full_standardized_centroid_distance_to_sources": float(np.linalg.norm(standardized[mask].mean(axis=0) - source_centroid)),
            "condition_cosine_to_source_mean": _cosine(differences[subject], source_direction),
        })
        full_dimensions.append({"subject": subject, "arithmetic_minus_rest_standardized": differences[subject].tolist()})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {"model": "source_pca_model.joblib", "coordinates": "coordinates.csv",
             "directions": "subject_directions.csv", "full_dimensions": "full_dimension_descriptors.json",
             "subjects_plot": "pca_subjects.png", "conditions_plot": "pca_conditions.png",
             "summary": "summary.json"}
    if any((output_dir / filename).exists() for filename in files.values()):
        raise FileExistsError("Refusing to overwrite existing PCA artifacts")
    joblib.dump(model, output_dir / files["model"])
    write_csv(output_dir / files["coordinates"], [
        {"row_index": int(index), "subject": str(subject), "label": int(label),
         "role": "target" if subject == target else "source", "pc1": float(point[0]), "pc2": float(point[1])}
        for index, (subject, label, point) in enumerate(zip(groups, y, coordinates))
    ])
    write_csv(output_dir / files["directions"], direction_rows)
    write_json(output_dir / files["full_dimensions"], {
        "n_dimensions": int(X.shape[1]), "source_mean_condition_direction": source_direction.tolist(),
        "source_mean_centroid": source_centroid.tolist(), "subjects": full_dimensions,
        "definition": "Arithmetic minus rest in original feature dimensions, standardized using source windows only; source references weight participants equally.",
        "limitation": "Source participants contribute to their own source reference; these are descriptive geometries, not unbiased distance tests.",
    })

    palette = [colormaps["tab20"](i / 19) for i in range(20)] + [colormaps["tab20b"](i / 19) for i in range(20)]
    figure, axes = _new_figure()
    for index, subject in enumerate(subjects):
        mask = groups == subject
        held_out = subject == target
        color = palette[index] if index < len(palette) else colormaps["hsv"](index / len(subjects))
        axes.scatter(coordinates[mask, 0], coordinates[mask, 1], s=26 if held_out else 9,
                     alpha=.85 if held_out else .35, color=color,
                     marker="^" if held_out else "o", edgecolors="black" if held_out else "none",
                     linewidths=.4, label=f"{subject} (target)" if held_out else subject)
    axes.legend(fontsize=7, ncols=2, loc="upper left", bbox_to_anchor=(1.01, 1))
    axes.set_title(f"Source-fitted PCA by participant; held out: {target}")
    _finish_plot(figure, axes, pca, output_dir / files["subjects_plot"])

    figure, axes = _new_figure()
    for label, name, color in [(0, "Rest", "#2878B5"), (1, "Arithmetic", "#E07A1F")]:
        for held_out in (False, True):
            mask = (y == label) & ((groups == target) if held_out else source_mask)
            axes.scatter(coordinates[mask, 0], coordinates[mask, 1], color=color,
                         s=30 if held_out else 10, alpha=.85 if held_out else .2,
                         marker="^" if held_out else "o", edgecolors="black" if held_out else "none",
                         linewidths=.4, label=f"{name}: {'target' if held_out else 'sources'}")
    for row in direction_rows:
        held_out = row["subject"] == target
        axes.annotate("", xy=(row["arithmetic_pc1"], row["arithmetic_pc2"]),
                      xytext=(row["rest_pc1"], row["rest_pc2"]),
                      arrowprops={"arrowstyle": "->", "color": "black" if held_out else "#666666",
                                  "alpha": 1 if held_out else .28, "linewidth": 2 if held_out else .6})
    axes.legend(loc="best", fontsize=9)
    axes.set_title(f"Condition distributions; arrows: rest to arithmetic; black: {target}")
    _finish_plot(figure, axes, pca, output_dir / files["conditions_plot"])

    summary = {
        "target_subject": str(target), "fit_subjects": fit_subjects,
        "n_source_windows": int(source_mask.sum()), "n_target_windows": int((~source_mask).sum()),
        "n_original_features": int(X.shape[1]), "n_components": 2,
        "svd_solver": "full", "seed": int(seed), "target_used_for_fit": False,
        "labels_used_for_fit": False, "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "pca_mean": pca.mean_.tolist(), "pca_components": pca.components_.tolist(),
        "descriptor_space": "Original feature dimensions with source-fitted StandardScaler; no PCA truncation.",
        "direction_definition": "Per participant arithmetic centroid minus rest centroid; cosine compared with equally weighted mean source direction.",
        "limitation": "Exploratory visualization; projected overlap or separation is not classification evidence. No fit or model selection uses target windows or labels.",
        "files": files,
    }
    write_json(output_dir / files["summary"], summary)
    return summary


def _finish_plot(figure, axes, pca, path):
    axes.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}% source variance)")
    axes.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}% source variance)")
    axes.grid(alpha=.15)
    figure.savefig(path, dpi=150)
    figure.clear()


def _participant_scores(rows):
    result = {}
    for row in rows:
        subject = row.get("subject")
        if not isinstance(subject, str) or not subject.strip() or subject in result:
            raise ValueError("Missing or duplicated participant ID")
        if row.get("status", "ok") != "ok":
            raise ValueError(f"Failed participant cannot be dropped from a paired analysis: {subject}")
        try:
            score = float(row["balanced_accuracy"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid balanced accuracy for {subject}") from error
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid balanced accuracy for {subject}")
        result[subject] = score
    if len(result) < 2:
        raise ValueError("At least two participants are required")
    return result


def paired_descriptive(reference_rows, comparison_rows, seed=42, replicates=5000):
    """Describe comparison minus reference BA, paired by participant ID.

    The bootstrap resamples participant differences without refitting models.
    Intervals do not account for overlapping LOSO training sets or exploratory
    selection across variants. No p-values or significance decisions are made.
    """
    reference, comparison = _participant_scores(reference_rows), _participant_scores(comparison_rows)
    if set(reference) != set(comparison):
        raise ValueError("Paired analyses require exactly matching participant IDs")
    if isinstance(replicates, (bool, np.bool_)) or not isinstance(replicates, (int, np.integer)) or replicates < 1:
        raise ValueError("Bootstrap replicates must be a positive integer")
    subjects = sorted(reference)
    differences = np.array([comparison[s] - reference[s] for s in subjects], dtype=float)
    sd = float(np.std(differences, ddof=1))
    rng = np.random.default_rng(seed)
    bootstrap_means = differences[rng.integers(0, len(subjects), size=(replicates, len(subjects)))].mean(axis=1)
    interval = np.percentile(bootstrap_means * 100, [2.5, 97.5])
    tie = np.isclose(differences, 0, rtol=0, atol=1e-12)
    return {
        "n_subjects": len(subjects), "difference_direction": "comparison minus reference",
        "reference_mean_ba": float(np.mean([reference[s] for s in subjects])),
        "comparison_mean_ba": float(np.mean([comparison[s] for s in subjects])),
        "mean_difference_pp": float(100 * differences.mean()),
        "median_difference_pp": float(100 * np.median(differences)),
        "wins": int(np.sum((differences > 0) & ~tie)), "ties": int(tie.sum()),
        "losses": int(np.sum((differences < 0) & ~tie)),
        "paired_dz": float(differences.mean() / sd) if sd > np.finfo(float).eps else None,
        "paired_dz_definition": "Mean paired BA difference divided by sample SD of paired differences (ddof=1); null if SD is zero.",
        "bootstrap_mean_difference_ci95_pp": interval.tolist(),
        "bootstrap_replicates": int(replicates), "seed": int(seed),
        "bootstrap_unit": "paired participant difference",
        "bootstrap_method": "percentile, without model refitting",
        "interpretation": "Exploratory conditional descriptive interval for these predictions; overlapping LOSO training sets are not independent experiments. It is not a confirmatory population interval and does not adjust for comparing multiple variants.",
        "subject_differences": [{"subject": s, "reference_ba": reference[s], "comparison_ba": comparison[s],
                                 "difference_pp": float(100 * delta)} for s, delta in zip(subjects, differences)],
    }
