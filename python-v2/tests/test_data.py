"""Meaningful data-contract checks using synthetic EDF bytes and mocked decoding."""

import json
from pathlib import Path

import numpy as np
import pytest

from eeg_v2.data import CHANNELS, inspect_edf, load_recording, quality_summary


def _edf(path: Path, *, labels=None, fs=500, unit="uV", duration=2,
         annotation=b"+0\x14\x14\x00", prefilter="") -> Path:
    labels = list(labels or CHANNELS)
    signal_labels = [f"EEG {label}" for label in labels] + ["EDF Annotations"]
    ns = len(signal_labels)

    def text(value, width):
        return str(value).encode("ascii").ljust(width, b" ")

    fixed = b"".join([
        text(0, 8), text("PRIVATE PATIENT", 80), text("PRIVATE RECORD", 80),
        text("01.01.01", 8), text("01.02.03", 8), text(256 + ns * 256, 8),
        text("EDF+C", 44), text(duration, 8), text(1, 8), text(ns, 4),
    ])
    n_samples = [fs] * len(labels) + [64]
    fields = [
        (signal_labels, 16), ([""] * ns, 80), ([unit] * len(labels) + [""], 8),
        ([-100] * ns, 8), ([100] * ns, 8), ([-32768] * ns, 8), ([32767] * ns, 8),
        ([prefilter] * len(labels) + [""], 80), (n_samples, 8), ([""] * ns, 32),
    ]
    header = fixed + b"".join(text(value, width) for values, width in fields for value in values)
    records = []
    for record in range(duration):
        tal = annotation if record == 0 else f"+{record}\x14\x14\x00".encode()
        records.append(bytes(2 * fs * len(labels)) + tal.ljust(128, b"\x00"))
    path.write_bytes(header + b"".join(records))
    return path


def test_inspection_is_anonymous_and_preserves_provenance(tmp_path):
    path = _edf(tmp_path / "Subject00_1.edf", prefilter="HP:0.5Hz LP:45Hz",
                annotation=b"+0\x14\x14\x00+0.5\x151\x14PRIVATE EVENT\x14\x00")
    meta = inspect_edf(path)
    serialized = json.dumps(meta, allow_nan=False)
    assert "PRIVATE" not in serialized
    assert "01.01.01" not in serialized and "01.02.03" not in serialized
    assert meta["duration_seconds"] == 2 and meta["fs_hz"] == 500
    assert meta["eeg_labels"] == list(CHANNELS)
    assert meta["channels"][-1]["fs_hz"] is None
    assert meta["prefilter"] == ["HP:0.5Hz LP:45Hz"]
    assert len(meta["sha256"]) == 64
    assert meta["annotations"]["event_count"] == 1
    assert meta["annotations"]["event_timings"][0]["onset_seconds"] == 0.5
    assert meta["annotations"]["event_timings"][0]["duration_seconds"] == 1
    assert meta["annotations"]["task_start_confirmed"] is False


def test_hash_changes_with_source_and_truncated_files_raise(tmp_path):
    path = _edf(tmp_path / "source.edf")
    before = inspect_edf(path)["sha256"]
    with path.open("r+b") as stream:
        stream.seek(256 + 20 * 256)
        stream.write(b"\x01")
    assert inspect_edf(path)["sha256"] != before
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="file size"):
        inspect_edf(path)


class _Raw:
    info = {"sfreq": 500.0}
    n_times = 1000

    def __init__(self):
        self.closed = False

    def get_data(self, *, picks, start, stop):
        # Distinct channel values and a time ramp expose reordering/crop bugs.
        return np.array([(CHANNELS.index(label.removeprefix("EEG ")) + 1)
                         + np.arange(start, stop) / 10000 for label in picks]) * 1e-6

    def close(self):
        self.closed = True


def test_load_orders_labels_crops_half_open_and_converts_units(tmp_path, monkeypatch):
    path = _edf(tmp_path / "source.edf", labels=reversed(CHANNELS))
    raw = _Raw()
    calls = []

    def read(*args, **kwargs):
        calls.append(kwargs)
        return raw

    monkeypatch.setattr("mne.io.read_raw_edf", read)
    data, meta = load_recording(path, crop_start=0.2, crop_seconds=0.4)
    assert data.shape == (19, 200)
    np.testing.assert_allclose(data[:, 0], np.arange(1, 20) + 0.01)
    np.testing.assert_allclose(data[:, -1], np.arange(1, 20) + 0.0299)
    assert meta["crop_start_sample"] == 100 and meta["crop_stop_sample_exclusive"] == 300
    assert meta["data_unit"] == "uV" and meta["filter_applied"] is False
    assert calls == [{"preload": False, "verbose": "ERROR"}]
    assert raw.closed


def test_real_mne_decoder_matches_edf_digital_to_physical_calibration(tmp_path):
    path = _edf(tmp_path / "synthetic.edf")
    data, _ = load_recording(path, crop_seconds=1)
    # Digital zero mapped through [-32768, 32767] -> [-100, 100] microvolts.
    expected_uv = 32768 * 200 / 65535 - 100
    np.testing.assert_allclose(data, expected_uv, atol=1e-10, rtol=1e-8)


def test_discontinuous_recording_rejected(tmp_path, monkeypatch):
    path = _edf(tmp_path / "source.edf")
    with path.open("r+b") as stream:
        stream.seek(192)
        stream.write(b"EDF+D")
    monkeypatch.setattr("mne.io.read_raw_edf", lambda *a, **k: pytest.fail("must not decode"))
    with pytest.raises(ValueError, match="discontinuous"):
        load_recording(path, crop_seconds=1)


@pytest.mark.parametrize("kwargs,match", [
    ({"fs": 250}, "500 Hz"), ({"unit": "mV"}, "microvolts"),
    ({"labels": list(CHANNELS[:-1])}, "missing or duplicate"),
    ({"labels": list(CHANNELS[:-1]) + [CHANNELS[0]]}, "missing or duplicate"),
])
def test_invalid_channel_contract_raises_before_decode(tmp_path, kwargs, match, monkeypatch):
    path = _edf(tmp_path / "source.edf", **kwargs)
    monkeypatch.setattr("mne.io.read_raw_edf", lambda *a, **k: pytest.fail("must not decode"))
    with pytest.raises(ValueError, match=match):
        load_recording(path, crop_seconds=1)


@pytest.mark.parametrize("start,seconds", [(-1, 1), (0, 0), (1, 2), (0.001, 1), (0, float("nan"))])
def test_invalid_crops_raise(tmp_path, start, seconds):
    path = _edf(tmp_path / "source.edf")
    with pytest.raises(ValueError):
        load_recording(path, crop_start=start, crop_seconds=seconds)


def test_quality_detects_frequency_flatline_nonfinite_and_possible_clipping():
    fs = 500
    time = np.arange(2000) / fs
    sine = 10 * np.sin(2 * np.pi * 10 * time)
    bad = sine.copy()
    bad[10] = np.nan
    clipped = np.clip(sine, -3, 3)
    summary = quality_summary(np.vstack([sine, np.zeros_like(sine), bad, clipped]), fs)
    json.dumps(summary, allow_nan=False)
    alpha, flat, missing, clipping = summary["channels"]
    assert alpha["bandpower_uV2"]["alpha"] == pytest.approx(50, rel=0.02)
    assert alpha["possible_clipping"] is False
    assert flat["flatline_diff_fraction"] == 1 and flat["ptp_uV"] == 0
    assert flat["possible_clipping"] is False
    assert missing["finite_fraction"] == pytest.approx(1999 / 2000)
    assert missing["psd_status"] == "nonfinite_samples"
    assert all(value is None for value in missing["bandpower_uV2"].values())
    assert clipping["possible_clipping"] is True


def test_quality_all_nonfinite_is_serializable_and_low_nyquist_is_explicit():
    with np.errstate(invalid="ignore"):
        summary = quality_summary(np.full((1, 10), np.nan), 500)
    assert summary["channels"][0]["std_uV"] is None
    json.dumps(summary, allow_nan=False)
    assert quality_summary(np.ones((1, 100)), 50)["channels"][0]["bandpower_uV2"]["gamma"] is None
