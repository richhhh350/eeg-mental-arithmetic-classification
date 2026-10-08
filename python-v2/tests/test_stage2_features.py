import numpy as np
import pytest

from eeg_v2.features import differential_entropy
from eeg_v2.stage2_features import extract_features


def all_channels(signal):
    return np.broadcast_to(signal, (1, 19, len(signal))).copy()


def test_de_retains_exact_stage1_values_and_column_order():
    windows = np.random.default_rng(42).normal(size=(3, 19, 1000))
    actual, names = extract_features(windows)
    np.testing.assert_array_equal(actual, differential_entropy(windows))
    assert names[:4] == ["Fp1:de:4-8Hz", "Fp1:de:8-13Hz", "Fp1:de:13-30Hz", "Fp2:de:4-8Hz"]
    assert names[-1] == "Pz:de:13-30Hz"


@pytest.mark.parametrize("bands", [((8, 13), (13, 30)), ((4, 8), (8, 13), (13, 30), (30, 45))])
def test_band_ablation_and_gamma_keep_exact_de_values(bands):
    windows = np.random.default_rng(13).normal(size=(2, 19, 1000))
    actual, names = extract_features(windows, bands=bands)
    np.testing.assert_array_equal(actual, differential_entropy(windows, bands=bands))
    assert len(names) == actual.shape[1] == 19 * len(bands)


def test_relative_power_known_mixture_and_amplitude_invariance():
    time = np.arange(256) / 128
    mixture = (np.sin(2 * np.pi * 6 * time) + 2 * np.sin(2 * np.pi * 10 * time)
               + 3 * np.sin(2 * np.pi * 20 * time))
    windows = all_channels(mixture) * np.arange(1, 20)[None, :, None]
    actual, names = extract_features(windows, fs=128, kind="relative_power")
    np.testing.assert_allclose(actual.reshape(19, 3), np.tile(np.array([1, 4, 9]) / 14, (19, 1)), rtol=1e-12)
    scaled, _ = extract_features(1e-8 * windows, fs=128, kind="relative_power")
    np.testing.assert_allclose(scaled, actual, rtol=1e-12)
    assert names[0] == "Fp1:relative_power:4-8Hz"


def test_relative_power_denominator_does_not_double_count_shared_edge():
    signal = np.sin(2 * np.pi * 8 * np.arange(256) / 128)
    actual, _ = extract_features(all_channels(signal), fs=128, kind="relative_power")
    # Both inclusive theta and alpha contain 8 Hz, but the denominator includes
    # that bin only once. Normalizing by summed band powers would give 0.5.
    np.testing.assert_allclose(actual.reshape(19, 3), np.tile([1, 1, 0], (19, 1)), atol=1e-25, rtol=1e-12)


def test_relative_power_zero_denominator_returns_zero():
    actual, _ = extract_features(np.zeros((2, 19, 1000)), kind="relative_power")
    np.testing.assert_array_equal(actual, np.zeros((2, 57)))


def test_relative_power_rejects_bands_outside_denominator_range():
    for bands in (((0.5, 4.0),), ((30.0, 45.0),)):
        with pytest.raises(ValueError, match="4-30 Hz denominator"):
            extract_features(np.zeros((1, 19, 1000)), kind="relative_power", bands=bands)


def test_hjorth_matches_scalar_reference_sample_differences():
    signal = np.array([0.0, 1.0, 4.0, 2.0, -1.0, 3.0])
    windows = all_channels(signal)
    actual, names = extract_features(windows, kind="hjorth")
    # Explicit scalar population variances are independent of the vectorized
    # implementation and guard against fs-scaled derivatives or ddof=1.
    def variance(x):
        mean = sum(x) / len(x)
        return sum((v - mean) ** 2 for v in x) / len(x)
    first = [signal[i + 1] - signal[i] for i in range(len(signal) - 1)]
    second = [first[i + 1] - first[i] for i in range(len(first) - 1)]
    v0, v1, v2 = variance(signal), variance(first), variance(second)
    expected = [np.log2(v0 + 1e-15), np.sqrt(v1 / v0), np.sqrt(v2 / v1) / np.sqrt(v1 / v0)]
    np.testing.assert_allclose(actual.reshape(19, 3), np.tile(expected, (19, 1)), rtol=1e-14)
    assert names[:3] == ["Fp1:hjorth:log2_activity", "Fp1:hjorth:mobility", "Fp1:hjorth:complexity"]
    different_fs, _ = extract_features(windows, fs=128, kind="hjorth")
    np.testing.assert_array_equal(actual, different_fs)


@pytest.mark.parametrize("signal", [np.zeros(1000), np.ones(1000) * 3, np.arange(1000)])
def test_hjorth_flat_signal_or_constant_derivative_stays_finite(signal):
    actual, _ = extract_features(all_channels(signal), kind="hjorth")
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual.reshape(19, 3)[:, 1:], np.zeros((19, 2)))
    np.testing.assert_allclose(actual[0, 0], np.log2(np.var(signal) + 1e-15))


def test_concatenation_matches_individual_feature_blocks():
    windows = np.random.default_rng(2026).normal(size=(2, 19, 500))
    de, de_names = extract_features(windows, kind="de")
    hjorth, hjorth_names = extract_features(windows, kind="hjorth")
    combined, names = extract_features(windows, kind="de_hjorth")
    np.testing.assert_array_equal(combined, np.concatenate((de, hjorth), axis=1))
    assert names == de_names + hjorth_names
    assert combined.shape == (2, 114)


@pytest.mark.parametrize("kind,width", [("de", 57), ("relative_power", 57), ("hjorth", 57), ("de_hjorth", 114)])
def test_features_are_per_window_and_preserve_empty_batch(kind, width):
    windows = np.random.default_rng(10).normal(size=(2, 19, 500))
    original = windows.copy()
    actual, names = extract_features(windows, kind=kind)
    perturbed = windows.copy()
    perturbed[1] *= 100
    changed, _ = extract_features(perturbed, kind=kind)
    np.testing.assert_array_equal(actual[0], changed[0])
    np.testing.assert_array_equal(windows, original)
    empty, empty_names = extract_features(windows[:0], kind=kind)
    assert empty.shape == (0, width)
    assert names == empty_names


@pytest.mark.parametrize("kind", ["de", "relative_power", "hjorth", "de_hjorth"])
def test_nonfinite_values_rejected_for_every_feature_family(kind):
    windows = np.zeros((1, 19, 1000))
    windows[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        extract_features(windows, kind=kind)


@pytest.mark.parametrize("kind", ["hjorth", "de_hjorth"])
def test_hjorth_requires_nonempty_second_difference(kind):
    with pytest.raises(ValueError, match="three samples"):
        extract_features(np.zeros((1, 19, 2)), kind=kind)


def test_invalid_kind_and_duplicate_bands_rejected():
    windows = np.zeros((1, 19, 1000))
    with pytest.raises(ValueError, match="Unknown feature"):
        extract_features(windows, kind="secret_feature")
    with pytest.raises(ValueError, match="Duplicate bands"):
        extract_features(windows, bands=((4, 8), (4, 8)))
