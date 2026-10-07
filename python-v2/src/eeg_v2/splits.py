"""Raw-sample interval helpers; all intervals are half-open [start, stop)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import math
import operator


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}.") from exc
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return result


def window_intervals(
    start_sample: int,
    stop_sample: int,
    window_samples: int,
    stride_samples: int,
) -> list[tuple[int, int]]:
    """Enumerate complete windows in original recording sample coordinates.

    An incomplete tail is dropped; a window ending exactly at stop_sample is
    included. Overlap is allowed within this one input block, never outside it.
    """
    start = _integer(start_sample, "start_sample")
    stop = _integer(stop_sample, "stop_sample")
    window = _integer(window_samples, "window_samples", minimum=1)
    stride = _integer(stride_samples, "stride_samples", minimum=1)
    if stop < start:
        raise ValueError("stop_sample must be >= start_sample.")
    return [(left, left + window) for left in range(start, stop - window + 1, stride)]


def chronological_blocks(
    total_samples: int,
    fs: float = 500,
    gap_seconds: float = 4,
    ratios: Sequence[float] = (0.6, 0.2, 0.2),
) -> dict[str, tuple[int, int]]:
    """Split a recording into ordered train/val/test blocks with two gaps.

    First subtract two gaps from total_samples. Floor the remaining sample
    count times each of the train and validation ratios; assign all remaining
    samples to test. Ratios must be positive and sum to one. A gap must map to
    an integer number of samples; no implicit time rounding is performed.

    The default four-second gap is a conservative design choice for the
    current baseline without additional temporal filtering and with planned
    windows up to four seconds. It does NOT guarantee isolation for future
    filters; their support, padding and boundary trimming must be reviewed.
    """
    total = _integer(total_samples, "total_samples", minimum=1)
    fs = float(fs)
    gap_seconds = float(gap_seconds)
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be finite and positive.")
    if not math.isfinite(gap_seconds) or gap_seconds < 0:
        raise ValueError("gap_seconds must be finite and nonnegative.")
    gap_exact = fs * gap_seconds
    if not math.isfinite(gap_exact):
        raise ValueError("Gap sample count must be finite.")
    gap = round(gap_exact)
    if not math.isclose(gap_exact, gap, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("gap_seconds * fs must be an integer sample count.")
    proportions = tuple(float(value) for value in ratios)
    if (
        len(proportions) != 3
        or any(not math.isfinite(value) or value <= 0 for value in proportions)
        or not math.isclose(sum(proportions), 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("ratios must contain three positive values summing to one.")
    available = total - 2 * gap
    if available < 3:
        raise ValueError("Too few samples remain after the two isolation gaps.")
    train_count = math.floor(available * proportions[0])
    val_count = math.floor(available * proportions[1])
    test_count = available - train_count - val_count
    if min(train_count, val_count, test_count) < 1:
        raise ValueError("Each split must contain at least one sample.")
    val_start = train_count + gap
    test_start = val_start + val_count + gap
    return {
        "train": (0, train_count),
        "val": (val_start, val_start + val_count),
        "test": (test_start, total),
    }


def assert_no_interval_overlap(rows: Iterable[Mapping[str, object]]) -> None:
    """Reject cross-split overlap within each recording in one split manifest.

    Rows require recording_id, split, start_sample and stop_sample. Same-split
    windows may overlap, and separate recordings have separate time axes.
    Validate each outer/inner fold separately: rows from distinct folds are not
    expected to be disjoint. Touching half-open endpoints do not overlap.
    """
    grouped: dict[object, list[tuple[int, int, object]]] = defaultdict(list)
    for row in rows:
        try:
            recording = row["recording_id"]
            split = row["split"]
            start = _integer(row["start_sample"], "start_sample")
            stop = _integer(row["stop_sample"], "stop_sample")
        except KeyError as exc:
            raise ValueError(f"Missing interval field: {exc.args[0]}") from exc
        if recording is None or split is None:
            raise ValueError("recording_id and split must not be None.")
        try:
            hash(recording)
            hash(split)
        except TypeError as exc:
            raise ValueError("recording_id and split must be hashable identifiers.") from exc
        if stop <= start:
            raise ValueError("Manifest intervals must satisfy stop_sample > start_sample.")
        grouped[recording].append((start, stop, split))

    for recording, intervals in grouped.items():
        latest_by_split: dict[object, tuple[int, int]] = {}
        for start, stop, split in sorted(intervals, key=lambda item: (item[0], item[1])):
            for other_split, (other_start, other_stop) in latest_by_split.items():
                if other_split != split and other_stop > start:
                    raise AssertionError(
                        f"Recording {recording!r}: {other_split!r} "
                        f"[{other_start}, {other_stop}) overlaps {split!r} [{start}, {stop})."
                    )
            if split not in latest_by_split or stop > latest_by_split[split][1]:
                latest_by_split[split] = (start, stop)
