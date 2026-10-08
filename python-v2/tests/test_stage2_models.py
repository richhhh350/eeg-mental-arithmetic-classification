import json
from pathlib import Path
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC

import eeg_v2.stage2_models as models
from eeg_v2.evaluation import fit_source_model


def toy_data():
    rng = np.random.default_rng(17)
    xs, ys, groups = [], [], []
    for number in range(6):
        n = 3 + number
        y = np.repeat([0, 1], n)
        x = rng.normal(size=(len(y), 4)) + y[:, None] * .6 + number * .1
        xs.append(x)
        ys.extend(y)
        groups.extend([f"Subject{number:02d}"] * len(y))
    return np.concatenate(xs), np.array(ys), np.array(groups)


@pytest.mark.parametrize("kind,params", [("lr", {"C": .1}), ("lda", {"shrinkage": "auto"}),
                                        ("svm", {"C": 1, "gamma": "scale"}), ("dummy", {})])
def test_models_score_semantics_and_train_scaler(kind, params):
    x, y, _ = toy_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = models._fit(models.make_model(kind, params), x, y)
    assert not any(issubclass(w.category, FutureWarning) for w in caught)
    pred, score, score_kind = models.predict_scores(model, x)
    np.testing.assert_allclose(model["scaler"].mean_, x.mean(axis=0))
    if kind == "svm":
        assert isinstance(model["classifier"], SVC)
        assert model["classifier"]._effective_probability is False
        assert score_kind == "decision_margin"
        np.testing.assert_array_equal(score, model.decision_function(x))
        np.testing.assert_array_equal(pred, score >= 0)
    else:
        assert score_kind == "probability"
        np.testing.assert_array_equal(score, model.predict_proba(x)[:, 1])
        np.testing.assert_array_equal(pred, score >= .5)
    assert models.score_metrics(y, pred, score)["n_windows"] == len(y)


def test_overlap_pool_keeps_validation_subjects_out_of_every_fit(monkeypatch):
    x, y, groups = toy_data()
    # Unequal duplication in the train pool must not define the validation folds.
    indices = np.concatenate([np.arange(len(y)), np.flatnonzero(groups == "Subject00"),
                              np.flatnonzero(groups == "Subject00")])
    fitted_x = []
    original_fit = models._fit
    def tracked_fit(model, features, labels):
        fitted_x.append(features.copy())
        return original_fit(model, features, labels)
    monkeypatch.setattr(models, "_fit", tracked_fit)
    fitted, selection = models.fit_grouped(x[indices], y[indices], groups[indices], x, y, groups,
                                          "lr", [{"C": .1}, {"C": 1}])
    assert len(fitted_x) == 7
    for index, candidate in enumerate(selection["candidates"]):
        scores = []
        val_seen = []
        for fold_index, fold in enumerate(candidate["folds"]):
            assert not set(fold["fit_subjects"]) & set(fold["validation_subjects"])
            val_seen.extend(fold["validation_subjects"])
            expected = x[indices][np.isin(groups[indices], fold["fit_subjects"])]
            np.testing.assert_array_equal(fitted_x[index * 3 + fold_index], expected)
            scores.extend(fold["scores"])
            for row in fold["scores"]:
                assert row["n_windows"] == np.sum(groups == row["subject"])
        assert sorted(val_seen) == sorted(set(groups))
        assert candidate["mean_subject_balanced_accuracy"] == np.mean([r["balanced_accuracy"] for r in scores])
    np.testing.assert_array_equal(fitted_x[-1], x[indices])
    np.testing.assert_allclose(fitted["scaler"].mean_, x[indices].mean(axis=0))
    _, standard = models.fit_grouped(x, y, groups, x, y, groups, "lr", [{"C": .1}])
    assert [f["validation_subjects"] for f in standard["candidates"][0]["folds"]] == [
        f["validation_subjects"] for f in selection["candidates"][0]["folds"]]


def test_lr_exact_stage1_parity():
    x, y, groups = toy_data()
    config = json.loads((Path(__file__).parents[1] / "configs/stage1.json").read_text())
    old, old_selection = fit_source_model(x, y, groups, config)
    new, new_selection = models.fit_grouped(x, y, groups, x, y, groups, "lr", [{"C": c} for c in config["lr_C"]])
    np.testing.assert_array_equal(old.predict_proba(x), new.predict_proba(x))
    np.testing.assert_array_equal(old["scaler"].mean_, new["scaler"].mean_)
    assert new_selection["selected_params"]["C"] == old_selection["selected_C"]
    for old_c, new_c in zip(old_selection["candidates"], new_selection["candidates"]):
        assert old_c["folds"] == new_c["folds"]
        assert old_c["mean_subject_balanced_accuracy"] == new_c["mean_subject_balanced_accuracy"]


def test_temporal_never_merges_validation(monkeypatch):
    x, y, _ = toy_data()
    train_x, train_y = x[:6], y[:6]
    val_x, val_y = x[6:14] + 1000, y[6:14]
    fits = []
    original_fit = models._fit
    def record(model, features, labels):
        fits.append(features.copy())
        return original_fit(model, features, labels)
    monkeypatch.setattr(models, "_fit", record)
    model, selection = models.fit_temporal(train_x, train_y, val_x, val_y, "lr", [{"C": .1}, {"C": 1}])
    assert len(fits) == 2
    for features in fits:
        np.testing.assert_array_equal(features, train_x)
    np.testing.assert_allclose(model["scaler"].mean_, train_x.mean(axis=0))
    assert selection["validation_merged_into_fit"] is False
    assert selection["fit_windows"] == len(train_y)
    assert selection["selected_params"] in [{"C": .1}, {"C": 1}]


def test_first_candidate_wins_numerical_tie():
    records = [{"params": {"C": 10}, "mean_subject_balanced_accuracy": .5},
               {"params": {"C": .1}, "mean_subject_balanced_accuracy": .5 + 5e-13}]
    assert models._choose(records) is records[0]
    records[1]["mean_subject_balanced_accuracy"] = .51
    assert models._choose(records) is records[1]


def test_metrics_accept_margin_and_reject_bad_inputs():
    result = models.score_metrics([0, 0, 1, 1], [0, 0, 1, 1], [-20, -2, 2, 20])
    assert result["roc_auc"] == result["balanced_accuracy"] == result["macro_f1"] == 1
    for y, pred, score in [([0, 0], [0, 0], [0, 0]), ([0, 1], [0, 2], [0, 1]),
                           ([0, 1], [0, 1], [np.nan, 1]), ([0, 1], [0], [0, 1])]:
        with pytest.raises(ValueError):
            models.score_metrics(y, pred, score)


def test_invalid_source_pools_fail():
    x, y, g = toy_data()
    with pytest.raises(ValueError, match="same source"):
        models.fit_grouped(x, y, g, x[g != "Subject00"], y[g != "Subject00"], g[g != "Subject00"], "lr", [{"C": 1}])
    with pytest.raises(ValueError, match="Nonfinite"):
        bad = x.copy()
        bad[0, 0] = np.nan
        models.fit_grouped(bad, y, g, x, y, g, "lr", [{"C": 1}])
    with pytest.raises(ValueError, match="lacks a condition"):
        bad_y = y.copy()
        bad_y[g == "Subject00"] = 0
        models.fit_grouped(x, bad_y, g, x, y, g, "lr", [{"C": 1}])
    for params in [{}, {"C": 1, "class_weight": "balanced"}]:
        with pytest.raises(ValueError, match="Invalid model"):
            models.make_model("lr", params)
    with pytest.raises(ValueError, match="At least one"):
        models.fit_grouped(x, y, g, x, y, g, "lr", [])
    with pytest.raises(ValueError, match="inner subject folds"):
        models.fit_grouped(x, y, g, x, y, g, "lr", [{"C": 1}], inner_folds=7)


def test_failure_and_nonconvergence_are_not_substituted(monkeypatch):
    x, y, g = toy_data()
    class Broken:
        def fit(self, *_):
            raise RuntimeError("injected failure")
    monkeypatch.setattr(models, "make_model", lambda *_: Broken())
    with pytest.raises(RuntimeError, match="injected"):
        models.fit_grouped(x, y, g, x, y, g, "lr", [{"C": 1}])
    class NotConverged:
        def fit(self, *_):
            warnings.warn("not converged", ConvergenceWarning)
    with pytest.raises(ConvergenceWarning):
        models._fit(NotConverged(), x, y)
