import json

import joblib
import numpy as np
import pytest

from eeg_v2.provenance import read_csv
from eeg_v2.stage2_analysis import paired_descriptive, source_pca


def _data():
    rng = np.random.default_rng(27)
    groups = np.repeat(["Subject00", "Subject17", "Subject35"], 8)
    y = np.tile([0, 0, 0, 0, 1, 1, 1, 1], 3)
    X = rng.normal(size=(24, 5)) + y[:, None] * np.array([1, -.5, 0, .3, -.1])
    return X, groups, y


def _rows(scores, subjects=None):
    return [{"subject": s, "balanced_accuracy": value, "status": "ok"}
            for s, value in zip(subjects or ["A", "B", "C", "D"], scores)]


def test_target_perturbation_cannot_change_source_fit(tmp_path):
    X, groups, y = _data()
    first = source_pca(X, groups, y, "Subject35", tmp_path / "first")
    changed = X.copy()
    changed[groups == "Subject35"] *= 1000
    changed[groups == "Subject35"] += 999
    second = source_pca(changed, groups, 1 - y, "Subject35", tmp_path / "second")
    for key in ("scaler_mean", "scaler_scale", "pca_mean", "pca_components", "explained_variance_ratio"):
        np.testing.assert_array_equal(first[key], second[key])
    model = joblib.load(tmp_path / "first" / first["files"]["model"])
    np.testing.assert_allclose(model.named_steps["scaler"].mean_, X[groups != "Subject35"].mean(axis=0))
    assert first["fit_subjects"] == ["Subject00", "Subject17"]
    assert first["target_used_for_fit"] is False
    rows = read_csv(tmp_path / "first" / first["files"]["coordinates"])
    np.testing.assert_allclose([[float(r["pc1"]), float(r["pc2"])] for r in rows], model.transform(X))
    json.dumps(first, allow_nan=False)


def test_pca_artifacts_and_full_dimension_descriptors(tmp_path):
    X, groups, y = _data()
    summary = source_pca(X, groups, y, "Subject35", tmp_path)
    for filename in summary["files"].values():
        assert (tmp_path / filename).is_file()
    directions = read_csv(tmp_path / summary["files"]["directions"])
    model = joblib.load(tmp_path / summary["files"]["model"])
    scaled = model.named_steps["scaler"].transform(X)
    for row in directions:
        subject = row["subject"]
        diff = scaled[(groups == subject) & (y == 1)].mean(axis=0) - scaled[(groups == subject) & (y == 0)].mean(axis=0)
        assert float(row["full_standardized_condition_distance"]) == pytest.approx(np.linalg.norm(diff))
    with pytest.raises(FileExistsError):
        source_pca(X, groups, y, "Subject35", tmp_path)


@pytest.mark.parametrize("change", ["no_target", "nonfinite", "missing_class", "only_target", "constant"])
def test_pca_rejects_invalid_inputs(tmp_path, change):
    X, groups, y = _data()
    target = "Subject35"
    if change == "no_target":
        target = "Subject99"
    elif change == "nonfinite":
        X[0, 0] = np.nan
    elif change == "missing_class":
        y[groups == "Subject00"] = 0
    elif change == "only_target":
        groups[:] = target
    elif change == "constant":
        X[:] = 0
    with pytest.raises(ValueError):
        source_pca(X, groups, y, target, tmp_path)


def test_paired_statistics_match_hand_computation_and_are_deterministic():
    reference = _rows([.5, .7, .6, .9])
    comparison = _rows([.6, .7, .4, 1.0])
    result = paired_descriptive(reference, comparison, seed=42, replicates=5000)
    assert result == paired_descriptive(reference[::-1], comparison[::-1], seed=42, replicates=5000)
    delta = np.array([.1, 0, -.2, .1])
    assert result["mean_difference_pp"] == pytest.approx(0, abs=1e-12)
    assert result["median_difference_pp"] == pytest.approx(5)
    assert (result["wins"], result["ties"], result["losses"]) == (2, 1, 1)
    assert result["paired_dz"] == pytest.approx(delta.mean() / delta.std(ddof=1), abs=1e-12)
    interval = result["bootstrap_mean_difference_ci95_pp"]
    assert -20 <= interval[0] < interval[1] <= 10
    assert result["subject_differences"][0]["difference_pp"] == pytest.approx(10)
    assert not any("p_value" in key for key in result)
    json.dumps(result, allow_nan=False)


def test_constant_difference_has_no_infinite_effect_size():
    result = paired_descriptive(_rows([.5, .5, .5, .5]), _rows([.6, .6, .6, .6]))
    assert result["paired_dz"] is None
    np.testing.assert_allclose(result["bootstrap_mean_difference_ci95_pp"], [10, 10])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("change", ["duplicate", "mismatch", "missing_id", "failed", "nan", "out_of_range", "one_subject"])
def test_invalid_pairs_fail_instead_of_dropping_participants(change):
    reference, comparison = _rows([.5, .6, .7, .8]), _rows([.6, .7, .8, .9])
    if change == "duplicate":
        comparison[-1]["subject"] = "A"
    elif change == "mismatch":
        comparison[-1]["subject"] = "Z"
    elif change == "missing_id":
        del comparison[-1]["subject"]
    elif change == "failed":
        comparison[-1]["status"] = "failed"
    elif change == "nan":
        comparison[-1]["balanced_accuracy"] = float("nan")
    elif change == "out_of_range":
        comparison[-1]["balanced_accuracy"] = 1.5
    elif change == "one_subject":
        reference, comparison = reference[:1], comparison[:1]
    with pytest.raises(ValueError):
        paired_descriptive(reference, comparison)


@pytest.mark.parametrize("replicates", [0, -1, 1.5, True])
def test_invalid_bootstrap_count(replicates):
    with pytest.raises(ValueError):
        paired_descriptive(_rows([.5, .6, .7, .8]), _rows([.6, .7, .8, .9]), replicates=replicates)
