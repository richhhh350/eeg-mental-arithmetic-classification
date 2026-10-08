"""Integration checks for metadata verification and failed-run reporting."""
import numpy as np
import pytest

from eeg_v2.provenance import digest, file_digest, read_csv, read_json, write_csv, write_json
from eeg_v2.stage2 import execute_experiment, load_variant, verify_experiment, window_rows


def prepared_fixture(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    variant = root / "de2"
    variant.mkdir()
    rng = np.random.default_rng(100)
    groups = np.repeat([f"Subject{i:02d}" for i in range(5)], 8)
    y = np.tile([0, 0, 0, 0, 1, 1, 1, 1], 5)
    x = rng.normal(size=(len(y), 4)) + .3 * y[:, None]
    rows = []
    for i in range(5):
        for label in (0, 1):
            rec = f"Subject{i:02d}_{label+1}"
            rows.extend(window_rows(rec, f"Subject{i:02d}", label, [(j*1000,(j+1)*1000) for j in range(4)], "eval", 500))
    train_rows = [{**r, "split": "train"} for r in rows]
    write_csv(variant / "train_windows.csv", train_rows)
    write_csv(variant / "eval_windows.csv", rows)
    np.savez_compressed(variant / "arrays.npz", X_train=x, y_train=y, groups_train=groups, X_eval=x, y_eval=y, groups_eval=groups)
    config = {"candidates": {"lr": [{"C": .01}, {"C": 1}]}, "inner_folds": 3, "seed": 42}
    write_json(root / "config.json", config)
    write_json(root / "stage1_config.json", {})
    fingerprint = {"config": config}
    files = {p.relative_to(root).as_posix(): file_digest(p) for p in root.rglob("*") if p.is_file()}
    write_json(root / "manifest.json", {"dataset_id": digest(fingerprint), "fingerprint": fingerprint, "files": files})
    return root


def test_one_fold_runs_and_verifier_rejects_corrupted_prediction(tmp_path):
    prepared = prepared_fixture(tmp_path)
    output = tmp_path / "result"
    spec = {"id": "test", "kind": "lr", "variant": "de2", "protocol": "S2"}
    execute_experiment(prepared, output, spec, ["Subject00"])
    assert read_json(output / "verification.json")["passed"]
    rows = read_csv(output / "predictions.csv")
    rows[0]["prediction"] = 1 - int(rows[0]["prediction"])
    write_csv(output / "predictions.csv", rows)
    with pytest.raises(AssertionError, match="threshold"):
        verify_experiment(output, prepared)


def test_changed_feature_artifact_is_rejected(tmp_path):
    prepared = prepared_fixture(tmp_path)
    with (prepared / "de2/arrays.npz").open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(ValueError, match="changed"):
        load_variant(prepared, "de2")


def test_failed_training_is_explicit_and_has_no_predictions(tmp_path, monkeypatch):
    import eeg_v2.stage2 as runner
    prepared = prepared_fixture(tmp_path)
    def broken(*args, **kwargs):
        raise RuntimeError("injected training error")
    monkeypatch.setattr(runner, "fit_grouped", broken)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="incomplete"):
        execute_experiment(prepared, output, {"id": "test", "kind": "lr", "variant": "de2", "protocol": "S2"}, ["Subject00"])
    assert not (output / "predictions.csv").exists()
    summary = read_json(output / "summary.json")
    assert summary["status"] == "incomplete"
    assert summary["completed_folds"] == 0
    assert "mean_subject_balanced_accuracy" not in summary


def test_modified_frozen_config_is_rejected(tmp_path):
    prepared = prepared_fixture(tmp_path)
    config = read_json(prepared / "config.json")
    config["seed"] = 13
    write_json(prepared / "config.json", config)
    with pytest.raises(ValueError, match="changed"):
        load_variant(prepared, "de2")


def test_temporal_csv_intervals_round_trip_as_integers(tmp_path):
    prepared = prepared_fixture(tmp_path)
    folder = prepared / "s1"
    folder.mkdir()
    with np.load(prepared / "de2/arrays.npz", allow_pickle=False) as source:
        x, y, groups = source["X_eval"], source["y_eval"], source["groups_eval"]
    template = read_csv(prepared / "de2/eval_windows.csv")
    arrays = {}
    for role, offset in (("train", 0), ("val", 10000), ("test", 20000)):
        rows = []
        for row in template:
            start, stop = int(row["start_sample"]) + offset, int(row["stop_sample"]) + offset
            rows.append({**row, "split": role, "start_sample": start, "stop_sample": stop,
                         "window_id": f"{row['recording_id']}:{start}:{stop}"})
        write_csv(folder / f"{role}_windows.csv", rows)
        arrays.update({f"X_{role}": x, f"y_{role}": y, f"groups_{role}": groups})
    np.savez_compressed(folder / "arrays.npz", **arrays)
    manifest = read_json(prepared / "manifest.json")
    manifest["files"].update({p.relative_to(prepared).as_posix(): file_digest(p) for p in folder.iterdir()})
    write_json(prepared / "manifest.json", manifest)
    output = tmp_path / "temporal"
    execute_experiment(prepared, output, {"id": "s1", "kind": "lr", "variant": "s1", "protocol": "S1"}, ["Subject00"])
    assert read_json(output / "verification.json")["passed"]
