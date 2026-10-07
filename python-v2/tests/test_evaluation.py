import copy
import json
from pathlib import Path

import numpy as np
import pytest

from eeg_v2.cli import preprocess_windows
from eeg_v2.evaluation import fit_source_model, subject_scores
from eeg_v2.features import differential_entropy
from eeg_v2.splits import chronological_blocks, window_intervals


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / "configs/stage1.json").read_text())


def toy_data():
    rng = np.random.default_rng(17)
    groups = np.repeat([f"Subject{i:02d}" for i in range(7)], 20)
    y = np.tile(np.repeat([0, 1], 10), 7)
    x = rng.normal(size=(len(y), 5)) + y[:, None] * .5
    return x, y, groups


def test_outer_inputs_and_labels_cannot_change_source_fit(config):
    x, y, groups = toy_data()
    target = groups == "Subject06"
    model, details = fit_source_model(x[~target], y[~target], groups[~target], config)
    changed_x, changed_y = x.copy(), y.copy()
    changed_x[target] *= 100000
    changed_y[target] = 1 - changed_y[target]
    refit, second = fit_source_model(changed_x[~target], changed_y[~target], groups[~target], config)
    np.testing.assert_array_equal(model["scaler"].mean_, refit["scaler"].mean_)
    np.testing.assert_array_equal(model["classifier"].coef_, refit["classifier"].coef_)
    np.testing.assert_array_equal(model.predict_proba(x[target]), refit.predict_proba(x[target]))
    assert details == second
    assert "Subject06" not in details["fit_subjects"]
    for candidate in details["candidates"]:
        all_scores = []
        for fold in candidate["folds"]:
            assert not set(fold["fit_subjects"]) & set(fold["validation_subjects"])
            assert "Subject06" not in fold["fit_subjects"] + fold["validation_subjects"]
            all_scores.extend(s["balanced_accuracy"] for s in fold["scores"])
        assert candidate["mean_subject_balanced_accuracy"] == np.mean(all_scores)


def test_future_raw_perturbation_cannot_change_training_features(config):
    rng = np.random.default_rng(91)
    data = rng.normal(size=(19, 30000))
    blocks = chronological_blocks(30000)
    a, b = blocks["train"]
    def extract(signal):
        intervals = window_intervals(a, b, 1000, 1000)
        windows = np.stack([signal[:, start:stop] for start, stop in intervals])
        return differential_entropy(preprocess_windows(windows, config))
    before = extract(data)
    data[:, b:] = rng.normal(10000, 100, size=data[:, b:].shape)
    np.testing.assert_array_equal(before, extract(data))


def test_group_scores_do_not_weight_long_subjects_more():
    # A has two perfect samples; B has eight entirely wrong samples.
    y = np.array([0, 1] + [0, 1] * 4)
    prob = np.array([.1, .9] + [.9, .1] * 4)
    scores = subject_scores(y, prob, ["A"] * 2 + ["B"] * 8)
    assert np.mean([s["balanced_accuracy"] for s in scores]) == .5


def test_fit_failure_is_not_a_dummy_prediction(config, monkeypatch):
    import eeg_v2.evaluation as evaluation
    class Broken:
        def fit(self, x, y):
            raise RuntimeError("injected fit failure")
    monkeypatch.setattr(evaluation, "make_model", lambda *_: Broken())
    x, y, groups = toy_data()
    with pytest.raises(RuntimeError, match="injected"):
        fit_source_model(x, y, groups, config)


def test_unsupported_filter_is_not_silently_ignored(config):
    bad = copy.deepcopy(config)
    bad["preprocessing"]["additional_temporal_filter"] = True
    with pytest.raises(ValueError, match="neither"):
        preprocess_windows(np.zeros((1, 19, 1000)), bad)
