"""Small JSON/CSV helpers and reproducibility fingerprints."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing an empty result CSV")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def environment():
    packages = ("numpy", "scipy", "mne", "scikit-learn", "matplotlib", "joblib", "pytest", "h5py")
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "compute": "CPU; LR does not use GPU", "blas_threads_during_training": 1,
            "packages": {p: importlib.metadata.version(p) for p in packages}}


def source_fingerprint():
    root = Path(__file__).parent
    return {p.name: file_digest(p) for p in sorted(root.glob("*.py"))}
