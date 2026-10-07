"""EDF import and descriptive quality checks without filtering or model fitting.

Only anonymous EDF metadata are returned. Patient, recording identity, recording
date and annotation text are deliberately excluded from the public summaries.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import welch

CHANNELS = (
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4", "C3", "C4",
    "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz",
)
EXPECTED_FS = 500.0
POWER_BANDS = {
    "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0), "gamma": (30.0, 45.0),
    "analysis": (0.5, 45.0), "line_50hz": (48.0, 52.0),
}


def _label(label: str) -> str:
    label = label.strip()
    return label[4:].strip() if label.startswith("EEG ") else label


def _read_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        fixed = stream.read(256)
        if len(fixed) != 256 or fixed[:8].strip() != b"0":
            raise ValueError(f"{path.name}: expected a complete EDF (not BDF) header")
        try:
            header_bytes = int(fixed[184:192])
            record_count = int(fixed[236:244])
            record_seconds = float(fixed[244:252])
            signal_count = int(fixed[252:256])
        except ValueError as exc:
            raise ValueError(f"{path.name}: invalid numeric EDF header") from exc
        if (record_count <= 0 or not math.isfinite(record_seconds)
                or record_seconds <= 0 or signal_count <= 0
                or header_bytes < 256 + signal_count * 256):
            raise ValueError(f"{path.name}: unsupported or invalid EDF dimensions")
        signal_bytes = stream.read(signal_count * 256)
        if len(signal_bytes) != signal_count * 256:
            raise ValueError(f"{path.name}: incomplete EDF signal header")

    def field(offset: int, width: int, index: int) -> str:
        start = offset * signal_count + width * index
        return signal_bytes[start:start + width].decode("ascii", errors="replace").strip()

    channels = []
    for index in range(signal_count):
        name = field(0, 16, index)
        samples = int(field(216, 8, index))
        if samples <= 0:
            raise ValueError(f"{path.name}: nonpositive samples per record")
        channels.append({
            "label": name, "unit": field(96, 8, index),
            "samples_per_record": samples,
            "fs_hz": None if name == "EDF Annotations" else samples / record_seconds,
            "prefilter": field(136, 80, index),
        })
    record_bytes = 2 * sum(channel["samples_per_record"] for channel in channels)
    if path.stat().st_size != header_bytes + record_count * record_bytes:
        raise ValueError(f"{path.name}: file size does not match EDF record dimensions")
    return {
        "header_bytes": header_bytes, "record_count": record_count,
        "record_seconds": record_seconds, "record_bytes": record_bytes,
        "duration_seconds": record_count * record_seconds, "channels": channels,
        "edf_continuity": fixed[192:197].decode("ascii", errors="replace").strip(),
    }


def _annotations(path: Path, header: dict[str, Any]) -> dict[str, Any]:
    """Read only annotation bytes; hash rather than expose free-text descriptions."""
    annotation_indices = [i for i, c in enumerate(header["channels"])
                          if c["label"] == "EDF Annotations"]
    events: list[dict[str, Any]] = []
    timekeeping: list[float] = []
    with path.open("rb") as stream:
        for record in range(header["record_count"]):
            for annotation_index in annotation_indices:
                byte_offset = 2 * sum(c["samples_per_record"]
                                      for c in header["channels"][:annotation_index])
                stream.seek(header["header_bytes"] + record * header["record_bytes"]
                            + byte_offset)
                payload = stream.read(2 * header["channels"][annotation_index]["samples_per_record"])
                for tal in payload.split(b"\x00"):
                    if not tal:
                        continue
                    parts = tal.split(b"\x14")
                    if len(parts) < 2:
                        raise ValueError(f"{path.name}: malformed EDF annotation")
                    timing = parts[0].split(b"\x15")
                    onset = float(timing[0])
                    duration = float(timing[1]) if len(timing) > 1 else None
                    if (not math.isfinite(onset) or (duration is not None and
                            (not math.isfinite(duration) or duration < 0))):
                        raise ValueError(f"{path.name}: invalid annotation timing")
                    descriptions = [p for p in parts[1:] if p]
                    if not descriptions:
                        timekeeping.append(onset)
                    for description in descriptions:
                        events.append({
                            "onset_seconds": onset, "duration_seconds": duration,
                            "description_sha256": hashlib.sha256(description).hexdigest(),
                        })
    # Timing continuity is checkable; the semantic start of a task is not inferred.
    irregular_steps = sum(not math.isclose(b - a, header["record_seconds"], abs_tol=1e-6)
                          for a, b in zip(timekeeping, timekeeping[1:]))
    return {
        "event_count": len(events), "event_timings": events,
        "timekeeping_record_count": len(timekeeping),
        "first_timekeeping_onset_seconds": timekeeping[0] if timekeeping else None,
        "irregular_timekeeping_steps": irregular_steps,
        "task_start_confirmed": False,
        "description_policy": "free text omitted; SHA-256 only",
    }


def inspect_edf(path: str | Path) -> dict[str, Any]:
    """Return anonymous metadata, annotation timing summary and source SHA-256.

    SHA-256 reads the file's bytes but does not decode physiological waveforms.
    Blank prefilter strings are unknown provenance, not proof of absent filtering.
    """
    path = Path(path)
    header = _read_header(path)
    eeg = [c for c in header["channels"] if _label(c["label"]) in CHANNELS]
    rates = {c["fs_hz"] for c in eeg}
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "filename": path.name, "sha256": digest,
        "duration_seconds": header["duration_seconds"],
        "fs_hz": next(iter(rates)) if len(rates) == 1 else None,
        "record_count": header["record_count"],
        "record_seconds": header["record_seconds"],
        "edf_continuity": header["edf_continuity"],
        "eeg_labels": [_label(c["label"]) for c in eeg],
        "channels": header["channels"],
        "prefilter": sorted({c["prefilter"] for c in eeg}),
        "annotations": _annotations(path, header),
    }


def _sample_index(seconds: float, fs: float, name: str) -> int:
    if not math.isfinite(seconds):
        raise ValueError(f"{name} must be finite")
    samples = seconds * fs
    rounded = round(samples)
    if not math.isclose(samples, rounded, abs_tol=1e-7, rel_tol=0):
        raise ValueError(f"{name} must align with the {fs:g} Hz sample grid")
    return rounded


def load_recording(path: str | Path, crop_start: float = 0.0,
                   crop_seconds: float = 60.0) -> tuple[np.ndarray, dict[str, Any]]:
    """Read named EEG channels in canonical order, returning microvolts.

    This dataset-specific reader requires all 19 channels at 500 Hz with explicit
    microvolt header units. Crops are half-open [start, stop); MNE converts the EDF
    physical values to volts and this function converts once to microvolts.
    No filtering, rereferencing, resampling, ICA, normalization or rejection occurs.
    """
    import mne

    path = Path(path)
    meta = inspect_edf(path)
    eeg = [c for c in meta["channels"] if _label(c["label"]) in CHANNELS]
    labels = [_label(c["label"]) for c in eeg]
    if len(labels) != len(CHANNELS) or set(labels) != set(CHANNELS):
        raise ValueError(f"{path.name}: missing or duplicate required EEG channel labels")
    if any(c["fs_hz"] != EXPECTED_FS for c in eeg):
        raise ValueError(f"{path.name}: expected 500 Hz for every EEG channel")
    if any(c["unit"].replace("µ", "u").replace("μ", "u") != "uV" for c in eeg):
        raise ValueError(f"{path.name}: EEG physical units must be explicitly microvolts")
    if (meta["edf_continuity"] == "EDF+D"
            or meta["annotations"]["irregular_timekeeping_steps"]):
        raise ValueError(f"{path.name}: discontinuous EDF requires explicit segment handling")
    start = _sample_index(float(crop_start), EXPECTED_FS, "crop_start")
    count = _sample_index(float(crop_seconds), EXPECTED_FS, "crop_seconds")
    total = _sample_index(meta["duration_seconds"], EXPECTED_FS, "duration_seconds")
    if start < 0 or count <= 0 or start + count > total:
        raise ValueError(f"{path.name}: crop is outside the recording or has no samples")
    mapping = {_label(c["label"]): c["label"] for c in eeg}
    raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
    try:
        if raw.info["sfreq"] != EXPECTED_FS or raw.n_times != total:
            raise ValueError(f"{path.name}: decoded EDF dimensions disagree with header")
        data = raw.get_data(picks=[mapping[label] for label in CHANNELS],
                            start=start, stop=start + count) * 1e6
    finally:
        raw.close()
    if data.shape != (len(CHANNELS), count):
        raise ValueError(f"{path.name}: unexpected decoded EEG shape {data.shape}")
    meta.update({
        "channel_order": list(CHANNELS), "data_unit": "uV", "filter_applied": False,
        "crop_start_seconds": start / EXPECTED_FS,
        "crop_seconds": count / EXPECTED_FS,
        "crop_start_sample": start, "crop_stop_sample_exclusive": start + count,
        "n_samples": count,
    })
    return np.asarray(data, dtype=np.float64), meta


def _longest_run(mask: np.ndarray) -> int:
    positions = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return int(np.max(positions[1::2] - positions[::2], initial=0))


def _band_integral(freq: np.ndarray, psd: np.ndarray, low: float, high: float) -> float | None:
    if high > freq[-1]:
        return None
    inside = (freq > low) & (freq < high)
    x = np.r_[low, freq[inside], high]
    y = np.interp(x, freq, psd)
    # Explicit trapezoids work across supported NumPy versions.
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2))


def quality_summary(data: np.ndarray, fs: float) -> dict[str, Any]:
    """Describe channel quality; flags are heuristics, never automatic exclusions.

    PSD is withheld for channels containing nonfinite values. Repeated extrema
    approximate possible clipping; without amplifier rail values they cannot
    establish saturation. Flatline tolerance is 1e-6 microvolt.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or not data.shape[0] or data.shape[1] < 2:
        raise ValueError("data must have shape (channels, samples) with at least two samples")
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be finite and positive")
    tolerance = 1e-6
    rows: list[dict[str, Any]] = []
    for index, signal in enumerate(data):
        finite = np.isfinite(signal)
        values = signal[finite]
        adjacent = finite[:-1] & finite[1:]
        differences = signal[1:][adjacent] - signal[:-1][adjacent]
        flat = float(np.mean(np.abs(differences) <= tolerance)) if differences.size else None
        row: dict[str, Any] = {
            "index": index, "label": CHANNELS[index] if len(data) == len(CHANNELS) else f"ch{index + 1}",
            "finite_fraction": float(np.mean(finite)),
            "std_uV": float(np.std(values)) if values.size else None,
            "ptp_uV": float(np.ptp(values)) if values.size else None,
            "flatline_diff_fraction": flat,
            "extrema_fraction": None, "longest_extreme_run_samples": 0,
            "possible_clipping": False,
            "bandpower_uV2": {name: None for name in POWER_BANDS},
            "psd_status": "nonfinite_samples",
        }
        if values.size:
            lower = finite & (np.abs(signal - np.min(values)) <= tolerance)
            upper = finite & (np.abs(signal - np.max(values)) <= tolerance)
            row["extrema_fraction"] = float(np.count_nonzero(lower | upper) / values.size)
            row["longest_extreme_run_samples"] = max(_longest_run(lower), _longest_run(upper))
            row["possible_clipping"] = bool(
                row["ptp_uV"] > tolerance and row["extrema_fraction"] >= 0.01
                and row["longest_extreme_run_samples"] >= 3
            )
        if np.all(finite):
            freq, psd = welch(signal, fs=fs, window="hann",
                              nperseg=min(len(signal), max(2, round(2 * fs))),
                              detrend="constant", scaling="density")
            row["bandpower_uV2"] = {name: _band_integral(freq, psd, *band)
                                    for name, band in POWER_BANDS.items()}
            row["psd_status"] = "ok"
        rows.append(row)
    return {
        "fs_hz": float(fs), "n_channels": data.shape[0], "n_samples": data.shape[1],
        "finite_fraction": float(np.mean(np.isfinite(data))),
        "flatline_tolerance_uV": tolerance,
        "clipping_rule": "extrema fraction >= 1% and >= 3 consecutive extreme samples; heuristic only",
        "psd_method": "Welch Hann, up to 2 s per segment, constant detrend, density integrated by trapezoids",
        "power_band_edges_hz": {name: list(band) for name, band in POWER_BANDS.items()},
        "channels": rows,
    }
