from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from eeg_v2.eegnet import ChannelScaler, EEGNet, infer, seed_all, train_epoch
from eeg_v2.provenance import digest, read_json
from eeg_v2.stage3_data import filter_diagnostics, grid_matched_de, resample_windows
from eeg_v2.stage3_training import balanced_cells, fit_source_eegnet, subject_ba


@pytest.fixture
def config():
    return read_json(Path(__file__).resolve().parents[1] / "configs/stage3.json")


def test_antialias_passband_stopband_and_boundaries(config):
    diagnostics = filter_diagnostics(config)
    assert diagnostics["checks_passed"]
    assert diagnostics["half_support_seconds"] == 0.078125
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 19, 1000))
    first = resample_windows(x, config)
    x[2] += 100 * rng.normal(size=x[2].shape)
    changed = resample_windows(x, config)
    np.testing.assert_array_equal(first[:2], changed[:2])
    assert not np.array_equal(first[2], changed[2])
    np.testing.assert_allclose(first.mean(-1), 0, atol=1e-15)


@pytest.mark.parametrize("shape", [(19, 1000), (1, 18, 1000), (1, 19, 999)])
def test_resampling_rejects_invalid_geometry(config, shape):
    with pytest.raises(ValueError):
        resample_windows(np.zeros(shape), config)


def test_matched_grid_known_band_power():
    signal = 3 * np.sin(2 * np.pi * 10 * np.arange(1000) / 500)
    result = grid_matched_de(np.tile(signal, (2, 19, 1))).reshape(2, 19, 3)
    expected = 0.5 * np.log2(2 * np.pi * np.e * 4.5)
    np.testing.assert_allclose(result[..., 1], expected, atol=1e-12)
    np.testing.assert_array_equal(result[..., [0, 2]], -50)


def test_eegnet_geometry_constraints_and_eval_state():
    seed_all(42, 1)
    model = EEGNet()
    assert sum(p.numel() for p in model.parameters()) == 1666
    x = np.random.default_rng(5).normal(size=(5, 19, 256)).astype(np.float32)
    assert model(torch.from_numpy(x)).shape == (5, 2)
    with torch.no_grad():
        model.spatial.weight.fill_(10)
        model.classifier.weight.fill_(10)
    model.constrain()
    assert torch.all(model.spatial.weight.flatten(1).norm(dim=1) <= 1.000001)
    assert torch.all(model.classifier.weight.norm(dim=1) <= 0.250001)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    first = infer(model, x, batch_size=5)
    np.testing.assert_allclose(first, infer(model, x, batch_size=1), atol=2e-7, rtol=0)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert all(not m.training for m in model.modules())
    assert np.all((first >= 0) & (first <= 1))
    with pytest.raises(ValueError):
        model(torch.zeros(2, 19, 255))


def test_scaler_is_train_only_and_handles_flat_channel():
    source = np.random.default_rng(3).normal(size=(8, 19, 256))
    source[:, 0] = 2
    scaler = ChannelScaler().fit(source)
    means = scaler.mean_.copy()
    normalized = scaler.transform(source)
    np.testing.assert_allclose(normalized[:, 1:].mean(axis=(0, 2)), 0, atol=1e-7)
    np.testing.assert_allclose(normalized[:, 1:].std(axis=(0, 2)), 1, atol=1e-6)
    assert scaler.scale_[0, 0, 0] == 1
    scaler.transform(source * 100 + 50)
    np.testing.assert_array_equal(means, scaler.mean_)
    with pytest.raises(ValueError):
        scaler.transform(np.full_like(source, np.nan))


def test_validation_score_weights_people_equally():
    y = np.array([0, 1] + [0] * 10 + [1] * 10)
    groups = np.array(["A"] * 2 + ["B"] * 20)
    score = np.array([0, 1] + [1] * 10 + [0] * 10)
    assert subject_ba(y, score, groups) == {"A": 1.0, "B": 0.0}
    with pytest.raises(ValueError, match="equal"):
        balanced_cells(y, groups)


def test_tiny_signal_is_learnable_and_permutation_is_negative_control():
    seed_all(13, 1)
    rng = np.random.default_rng(4)
    labels = np.array([0, 1] * 16, dtype=np.int64)
    time = np.arange(256) / 128
    x = np.stack([np.tile(np.sin(2 * np.pi * (8 if label == 0 else 22) * time), (19, 1)) for label in labels])
    x += rng.normal(0, 0.05, x.shape)
    model = EEGNet(dropout=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(13)
    tensor_x, tensor_y = torch.from_numpy(x.astype(np.float32)), torch.from_numpy(labels)
    losses = [train_epoch(model, optimizer, tensor_x, tensor_y, generator, 16) for _ in range(40)]
    score = infer(model, x)
    assert losses[-1] < losses[0] * 0.6
    assert np.mean((score >= 0.5) == labels) >= 0.95
    # Balanced labels broken independently of the perfectly separable input.
    shuffled = np.array([0, 0, 1, 1] * 8)
    assert np.mean((score >= 0.5) == shuffled) == 0.5


def test_runner_target_perturbation_cannot_change_training(config, tmp_path, monkeypatch):
    from eeg_v2 import stage3
    config = deepcopy(config)
    config["training"].update(max_epochs=2, min_epochs=1, patience=1, threads=1)
    rng = np.random.default_rng(17)
    x = rng.normal(size=(24, 19, 256)).astype(np.float32)
    features = {"groups": np.repeat([f"S{i}" for i in range(6)], 4), "y": np.tile([0, 0, 1, 1], 6)}
    rows = [{"window_id": f"w{i}", "subject": str(s), "label": int(y)} for i, (s, y) in enumerate(zip(features["groups"], features["y"]))]
    manifest = {"dataset_id": "synthetic"}
    monkeypatch.setattr(stage3, "load_stage3", lambda _: (x, features, rows, config, manifest))
    fingerprint_id = digest(stage3.run_fingerprint(config, manifest))
    result = stage3._fit_one("unused", tmp_path / "first", "S0", 42, fingerprint_id)
    assert result["status"] == "ok", result
    x[:4] = rng.normal(100, 50, x[:4].shape)
    features["y"][:4] = 1 - features["y"][:4]
    for i in range(4):
        rows[i]["label"] = int(features["y"][i])
    result = stage3._fit_one("unused", tmp_path / "second", "S0", 42, fingerprint_id)
    assert result["status"] == "ok", result
    a, b = (tmp_path / name / "seed42/S0" for name in ("first", "second"))
    first = torch.load(a / "model.pt", weights_only=True)
    second = torch.load(b / "model.pt", weights_only=True)
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert read_json(a / "selection.json") == read_json(b / "selection.json")
    with np.load(a / "scaler.npz") as p, np.load(b / "scaler.npz") as q:
        np.testing.assert_array_equal(p["mean"], q["mean"])
        np.testing.assert_array_equal(p["scale"], q["scale"])
    stage3._check_completed(a, fingerprint_id)
    (a / "metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        stage3._check_completed(a, fingerprint_id)


def test_failed_training_never_makes_dummy_predictions(config, tmp_path, monkeypatch):
    from eeg_v2 import stage3
    def fail(_):
        raise RuntimeError("intentional input failure")
    monkeypatch.setattr(stage3, "load_stage3", fail)
    result = stage3._fit_one("unused", tmp_path, "S0", 42, "test")
    assert result["status"] == "failed"
    assert not (tmp_path / "seed42/S0/predictions.csv").exists()
    assert (tmp_path / "seed42/S0/failure.json").exists()
