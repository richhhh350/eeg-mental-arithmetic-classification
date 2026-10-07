import numpy as np
import pytest
from scipy.signal import periodogram

from eeg_v2.features import band_power, differential_entropy, log_energy_entropy


def _all_channels(signal):
    return np.broadcast_to(signal, (1, 19, len(signal))).copy()


@pytest.mark.parametrize("n_samples", [256, 300, 500])
def test_full_band_parseval_includes_dc_and_zero_padding(n_samples):
    windows = np.random.default_rng(42).normal(size=(3, 19, n_samples)) + 2.0
    powers = band_power(windows, fs=128, bands=((0, 64),))
    np.testing.assert_allclose(powers[..., 0], np.mean(windows**2, axis=-1), rtol=1e-12)


def test_matches_independent_scipy_periodogram_without_detrending():
    windows = np.random.default_rng(13).normal(size=(2, 19, 500)) + 5.0
    bands = ((0, 4), (4, 8), (8, 13), (13, 30))
    frequencies, density = periodogram(
        windows, fs=500, window="boxcar", nfft=512, detrend=False, scaling="density"
    )
    expected = np.stack(
        [
            density[..., (frequencies >= low) & (frequencies <= high)].sum(axis=-1)
            * (500 / 512)
            for low, high in bands
        ],
        axis=-1,
    )
    np.testing.assert_allclose(band_power(windows, bands=bands), expected, rtol=1e-12)


def test_known_sinusoids_and_channel_major_feature_order():
    fs = 128
    time = np.arange(256) / fs
    amplitude = np.arange(1, 20, dtype=float)
    mixture = (
        np.sin(2 * np.pi * 6 * time)
        + 2 * np.sin(2 * np.pi * 10 * time)
        + 3 * np.sin(2 * np.pi * 20 * time)
    )
    windows = amplitude[None, :, None] * mixture[None, None, :]
    expected_power = amplitude[:, None] ** 2 * np.array([0.5, 2.0, 4.5])[None, :]
    powers = band_power(windows, fs=fs)
    np.testing.assert_allclose(powers[0], expected_power, rtol=1e-12)
    expected_de = 0.5 * np.log2(2 * np.pi * np.e * expected_power)
    features = differential_entropy(windows, fs=fs)
    assert features.shape == (1, 57)
    np.testing.assert_allclose(features[0], expected_de.ravel(order="C"), rtol=1e-12)


@pytest.mark.parametrize("frequency, expected_bands", [(8, [0.5, 0.5, 0]), (13, [0, 0.5, 0.5])])
def test_shared_band_boundaries_are_inclusive(frequency, expected_bands):
    signal = np.sin(2 * np.pi * frequency * np.arange(256) / 128)
    actual = band_power(_all_channels(signal), fs=128)
    np.testing.assert_allclose(actual[0, 0], expected_bands, atol=1e-25, rtol=1e-12)


@pytest.mark.parametrize("signal", [np.full(256, 3.0), 3.0 * (-1.0) ** np.arange(256)])
def test_dc_and_nyquist_are_not_doubled(signal):
    powers = band_power(_all_channels(signal), fs=128, bands=((0, 64),))
    np.testing.assert_allclose(powers, 9.0, rtol=1e-12)


def test_zero_and_tiny_signals_use_v1_floors_without_nonfinite_values():
    windows = np.zeros((2, 19, 500))
    windows[1] = 1e-10 * np.sin(2 * np.pi * 10 * np.arange(500) / 500)
    np.testing.assert_array_equal(differential_entropy(windows), np.full((2, 57), -50.0))
    np.testing.assert_array_equal(log_energy_entropy(windows), np.full((2, 57), -15.0))


def test_lee_is_affine_in_de_away_from_clipping_for_fixed_window_length():
    windows = np.random.default_rng(2026).normal(size=(4, 19, 500))
    de = differential_entropy(windows)
    lee = log_energy_entropy(windows)
    expected = 2 * np.log10(2) * de + np.log10(500 / (2 * np.pi * np.e))
    np.testing.assert_allclose(lee, expected, atol=1e-12)


def test_empty_window_batch_preserves_feature_shape():
    windows = np.empty((0, 19, 500))
    assert differential_entropy(windows).shape == (0, 57)
    assert band_power(windows).shape == (0, 19, 3)


@pytest.mark.parametrize(
    "windows", [np.zeros((19, 500)), np.zeros((2, 500, 19)), np.zeros((2, 19, 1))]
)
def test_invalid_window_axes_or_length_are_rejected(windows):
    with pytest.raises(ValueError):
        differential_entropy(windows)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_nonfinite_signal_is_rejected(invalid_value):
    windows = np.zeros((1, 19, 500))
    windows[0, 0, 0] = invalid_value
    with pytest.raises(ValueError, match="finite"):
        differential_entropy(windows)


@pytest.mark.parametrize("fs", [0, -500, np.inf, np.nan])
def test_invalid_sampling_rate_is_rejected(fs):
    with pytest.raises(ValueError, match="fs"):
        band_power(np.zeros((1, 19, 500)), fs=fs)


@pytest.mark.parametrize("bands", [(), ((8, 4),), ((-1, 4),), ((4, 251),), ((np.nan, 8),)])
def test_invalid_bands_are_rejected(bands):
    with pytest.raises(ValueError):
        band_power(np.zeros((1, 19, 500)), bands=bands)


def test_complex_signal_is_rejected():
    with pytest.raises(ValueError, match="real-valued"):
        band_power(np.zeros((1, 19, 500), dtype=complex))
