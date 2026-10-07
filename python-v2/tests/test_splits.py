import numpy as np
import pytest

from eeg_v2.splits import assert_no_interval_overlap, chronological_blocks, window_intervals


def _row(start, stop, split="train", recording="subject00_rest"):
    return {
        "recording_id": recording,
        "split": split,
        "start_sample": start,
        "stop_sample": stop,
    }


def test_complete_windows_include_exact_stop_and_drop_partial_tail():
    assert window_intervals(100, 1100, 500, 500) == [(100, 600), (600, 1100)]
    assert window_intervals(100, 1099, 500, 500) == [(100, 600)]
    assert window_intervals(100, 599, 500, 500) == []
    assert window_intervals(100, 100, 500, 500) == []


def test_overlapping_windows_retain_absolute_recording_coordinates():
    intervals = window_intervals(1500, 3500, 1000, 500)
    assert intervals == [(1500, 2500), (2000, 3000), (2500, 3500)]
    recording = np.arange(5000)
    block = recording[1500:3500]
    for start, stop in intervals:
        np.testing.assert_array_equal(recording[start:stop], block[start - 1500 : stop - 1500])
        assert 1500 <= start < stop <= 3500


def test_default_chronological_blocks_deduct_gaps_before_ratios():
    blocks = chronological_blocks(60 * 500)
    assert blocks == {"train": (0, 15600), "val": (17600, 22800), "test": (24800, 30000)}
    assert blocks["val"][0] - blocks["train"][1] == 2000
    assert blocks["test"][0] - blocks["val"][1] == 2000


def test_rounding_remainder_goes_to_test():
    assert chronological_blocks(101, fs=1, gap_seconds=0) == {
        "train": (0, 60), "val": (60, 80), "test": (80, 101)
    }


def test_windows_from_separate_chronological_blocks_never_cross_boundaries():
    blocks = chronological_blocks(30000)
    rows = []
    for split, (start, stop) in blocks.items():
        intervals = window_intervals(start, stop, 2000, 1000)
        assert intervals
        for left, right in intervals:
            assert start <= left < right <= stop
            rows.append(_row(left, right, split=split))
    assert_no_interval_overlap(rows)


def test_overlap_checker_accepts_same_split_overlap_touching_and_other_recordings():
    rows = [
        _row(0, 1000), _row(500, 1500), _row(1500, 2000, split="val"),
        _row(0, 1000, split="test", recording="subject01_rest"),
    ]
    assert_no_interval_overlap(reversed(rows))
    assert_no_interval_overlap([])


def test_overlap_checker_rejects_single_shared_sample():
    with pytest.raises(AssertionError, match="overlaps"):
        assert_no_interval_overlap([_row(0, 1000), _row(999, 1500, split="test")])


def test_nested_same_split_interval_does_not_hide_cross_split_overlap():
    rows = [_row(0, 5000), _row(100, 200), _row(4000, 4500, split="test")]
    with pytest.raises(AssertionError, match="overlaps"):
        assert_no_interval_overlap(rows)


@pytest.mark.parametrize(
    "arguments", [(-1, 10, 2, 1), (10, 9, 2, 1), (0, 10, 0, 1), (0, 10, 2, 0), (0, 10, 2.5, 1)]
)
def test_invalid_window_parameters_are_rejected(arguments):
    with pytest.raises(ValueError):
        window_intervals(*arguments)


@pytest.mark.parametrize("total", [0, 4000, 4001, 4004])
def test_insufficient_chronological_samples_are_rejected(total):
    with pytest.raises(ValueError):
        chronological_blocks(total)


@pytest.mark.parametrize("ratios", [(0.6, 0.2), (0.6, 0.2, 0.3), (1, 0, 0), (0.6, np.nan, 0.4)])
def test_invalid_ratios_are_rejected(ratios):
    with pytest.raises(ValueError, match="ratios"):
        chronological_blocks(30000, ratios=ratios)


def test_fractional_sample_gap_is_rejected_instead_of_silently_rounded():
    with pytest.raises(ValueError, match="integer sample"):
        chronological_blocks(30000, gap_seconds=0.001)


@pytest.mark.parametrize("fs, gap", [(0, 4), (np.nan, 4), (500, -1), (500, np.inf)])
def test_invalid_gap_or_sampling_rate_is_rejected(fs, gap):
    with pytest.raises(ValueError):
        chronological_blocks(30000, fs=fs, gap_seconds=gap)


@pytest.mark.parametrize("start, stop", [(10, 10), (10, 9), (-1, 10)])
def test_invalid_manifest_intervals_are_rejected(start, stop):
    with pytest.raises(ValueError):
        assert_no_interval_overlap([_row(start, stop)])


def test_missing_manifest_field_is_rejected():
    row = _row(0, 1000)
    del row["recording_id"]
    with pytest.raises(ValueError, match="recording_id"):
        assert_no_interval_overlap([row])
