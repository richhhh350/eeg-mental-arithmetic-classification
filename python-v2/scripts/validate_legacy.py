"""Read-only audit of V1 MAT epoch shapes and same-input MATLAB DE parity.

Run from python-v2 after installing the project dependencies. Outputs contain
de-identified recording IDs and numeric evidence, not EDF identity headers.
The raw legacy files are never written. MATLAB is optional; an unavailable
MATLAB run is recorded as unavailable, never counted as a successful check.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import h5py
import mne
import numpy as np
from scipy.io import loadmat, savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eeg_v2.features import band_power, differential_entropy  # noqa: E402


CHANNELS = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4", "C3", "C4",
            "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz"]
BANDS = ((4., 8.), (8., 13.), (13., 30.), (30., 45.))


def audit_legacy(directory: Path) -> dict:
    records = []
    for path in sorted(directory.glob("Sub*_preprocessed.mat")):
        with h5py.File(path, "r") as handle:
            eeg = handle["EEG"]
            scalar = lambda key: float(np.asarray(eeg[key][()]).squeeze())
            pnts, trials, nbchan = (int(scalar(k)) for k in ("pnts", "trials", "nbchan"))
            times = np.asarray(eeg["times"][()]).ravel()
            data = eeg["data"]
            # MATLAB v7.3 storage reverses the axes exposed through h5py.
            expected_hdf5_shape = (trials, pnts, nbchan)
            if data.shape != expected_hdf5_shape:
                raise ValueError(f"Unexpected legacy data shape in {path.name}: {data.shape}")
            # Compare stored epoch boundaries directly instead of inferring
            # duplication from a historical function call or inclusive limits.
            previous_ends = np.asarray(data[:-1, -1, :], dtype=np.float64)
            next_starts = np.asarray(data[1:, 0, :], dtype=np.float64)
            repeated_vectors = np.all(previous_ends == next_starts, axis=1)
            records.append({
                "file": path.name,
                "matlab_data_shape": [nbchan, pnts, trials],
                "hdf5_data_shape": list(data.shape),
                "pnts": pnts, "trials": trials, "srate_hz": scalar("srate"),
                "concatenated_samples": pnts * trials,
                "concatenated_seconds": pnts * trials / scalar("srate"),
                "epoch_times_first_ms": float(times[0]),
                "epoch_times_last_ms": float(times[-1]),
                "epoch_times_step_ms": float(np.median(np.diff(times))),
                "epoch_times_match_sampling_grid": bool(np.allclose(
                    times, np.arange(pnts) * 1000 / scalar("srate"), atol=1e-9)),
                "adjacent_boundaries_checked": int(trials - 1),
                "identical_adjacent_endpoint_vectors": int(repeated_vectors.sum()),
                "minimum_boundary_max_channel_difference": float(
                    np.min(np.max(np.abs(previous_ends - next_starts), axis=1))),
            })
    if not records:
        raise FileNotFoundError(f"No legacy MAT files found in {directory}")
    signatures = Counter((r["pnts"], r["trials"], r["srate_hz"], r["concatenated_samples"])
                         for r in records)
    return {
        "method": "Read only EEG numeric metadata and adjacent epoch endpoints with h5py; no reprocessing.",
        "file_count": len(records),
        "shape_summary": [dict(pnts=k[0], trials=k[1], srate_hz=k[2],
                               concatenated_samples=k[3], file_count=v)
                          for k, v in sorted(signatures.items())],
        "total_boundaries_checked": sum(r["adjacent_boundaries_checked"] for r in records),
        "identical_adjacent_endpoint_vectors": sum(
            r["identical_adjacent_endpoint_vectors"] for r in records),
        "interpretation_limit": "The stored arrays establish epoch sizes and boundary values. They do not independently reconstruct original pre-epoch source timestamps or prove that no earlier preprocessing samples were removed.",
        "records": records,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_same_input(directory: Path) -> tuple[np.ndarray, list[dict]]:
    windows, sources = [], []
    # Preselected subject IDs and times, independent of any model performance.
    for filename in ("Subject00_1.edf", "Subject17_2.edf", "Subject35_1.edf"):
        path = directory / filename
        raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
        if raw.info["sfreq"] != 500:
            raise ValueError(f"Expected 500 Hz: {filename}")
        picks = [f"EEG {name}" for name in CHANNELS]
        signal = raw.get_data(picks=picks, start=0, stop=30000, units="uV")
        for start in (0, 10000, 20000):
            windows.append(signal[:, start:start + 1000])
            sources.append({"recording": filename, "recording_sha256": sha256(path),
                            "start_sample": start, "stop_sample_exclusive": start + 1000,
                            "preprocessing": "none; EDF physical samples converted to microvolts"})
        raw.close()
    return np.asarray(windows, dtype=np.float64), sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/legacy_validation"))
    parser.add_argument("--matlab", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "legacy_mat_audit": audit_legacy(args.legacy_dir)}
    windows, sources = make_same_input(args.data_dir)
    input_path = output / "same_input_windows.mat"
    matlab_output = output / "matlab_reference.mat"
    savemat(input_path, {"windows": windows, "fs": 500., "bands": np.asarray(BANDS)})
    python_de = differential_entropy(windows, fs=500, bands=BANDS).reshape(len(windows), 19, 4)
    python_power = band_power(windows, fs=500, bands=BANDS)
    parity = {"input_shape": list(windows.shape), "input_unit": "microvolt",
              "bands_hz": BANDS, "sources": sources, "status": "unavailable",
              "formula": "V1 one-sided next-power-of-two periodogram; inclusive bands; no demeaning; DE bits; low-power floor -50",
              "de_atol": 1e-10, "power_rtol": 1e-10, "power_atol": 1e-10}
    report["same_input_matlab_parity"] = parity
    if args.matlab and args.matlab.is_file():
        script_dir = Path(__file__).resolve().parent
        quote = lambda path: str(path).replace("'", "''")
        expression = (f"addpath('{quote(script_dir)}'); "
                      f"validate_matlab_de('{quote(input_path)}','{quote(matlab_output)}');")
        try:
            completed = subprocess.run([str(args.matlab), "-batch", expression],
                                       capture_output=True, text=True, errors="replace", timeout=240)
            (output / "matlab_stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (output / "matlab_stderr.txt").write_text(completed.stderr, encoding="utf-8")
            parity["matlab_exit_code"] = completed.returncode
            if completed.returncode == 0 and matlab_output.is_file():
                reference = loadmat(matlab_output)
                de_reference, power_reference = reference["de_matlab"], reference["power_matlab"]
                de_error = float(np.max(np.abs(python_de - de_reference)))
                power_error = float(np.max(np.abs(python_power - power_reference)))
                passed = bool(np.allclose(python_de, de_reference, rtol=0, atol=1e-10)
                              and np.allclose(python_power, power_reference, rtol=1e-10, atol=1e-10))
                parity.update(status="passed" if passed else "failed",
                              matlab_version=str(reference["matlab_version"].ravel()[0]),
                              de_max_abs_error=de_error, power_max_abs_error=power_error,
                              compared_de_values=int(python_de.size))
            else:
                parity["reason"] = "MATLAB failed; inspect matlab_stderr.txt and matlab_stdout.txt."
        except (OSError, subprocess.TimeoutExpired) as error:
            parity["reason"] = f"{type(error).__name__}: {error}"
    else:
        parity["reason"] = "No existing MATLAB executable provided."
    (output / "legacy_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"legacy_files": report["legacy_mat_audit"]["file_count"],
                      "legacy_shapes": report["legacy_mat_audit"]["shape_summary"],
                      "same_input_matlab_parity": parity["status"],
                      "report": str(output / "legacy_validation.json")}, ensure_ascii=False, indent=2))
    return 1 if parity["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
