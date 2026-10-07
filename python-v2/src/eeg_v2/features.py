"""Frequency features compatible with the MATLAB V1 periodogram formulas.

Inputs keep their physical units: no demeaning, re-referencing, filtering, or
scaling is performed here. For V1-compatible values, the caller must supply
microvolts. Adjacent frequency bands deliberately share an inclusive boundary,
as in the original MATLAB script.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


DEFAULT_BANDS = ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0))
POWER_FLOOR = 1e-15
N_CHANNELS = 19


def _validated_inputs(
    windows: np.ndarray, fs: float, bands: Sequence[tuple[float, float]]
) -> tuple[np.ndarray, float, np.ndarray]:
    if np.iscomplexobj(windows):
        raise ValueError("EEG windows must be real-valued.")
    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != N_CHANNELS:
        raise ValueError("windows must have shape (n_windows, 19, n_samples).")
    if values.shape[2] < 2:
        raise ValueError("Each window must contain at least two samples.")
    if not np.isfinite(values).all():
        raise ValueError("EEG windows must contain only finite values.")
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be finite and positive.")
    band_array = np.asarray(bands, dtype=np.float64)
    if band_array.ndim != 2 or band_array.shape[1] != 2 or len(band_array) == 0:
        raise ValueError("bands must be a nonempty sequence of (low, high) pairs.")
    if (
        not np.isfinite(band_array).all()
        or np.any(band_array[:, 0] < 0)
        or np.any(band_array[:, 0] > band_array[:, 1])
        or np.any(band_array[:, 1] > fs / 2)
    ):
        raise ValueError("Each band must satisfy 0 <= low <= high <= fs / 2.")
    return values, fs, band_array


def band_power(
    windows: np.ndarray,
    fs: float = 500,
    bands: Sequence[tuple[float, float]] = DEFAULT_BANDS,
) -> np.ndarray:
    """Return integrated band powers with shape ``(windows, 19, bands)``.

    Use a rectangular, non-detrended periodogram with NFFT equal to the next
    power of two at least as large as the original window length N. The
    one-sided PSD is ``abs(FFT)**2 / (N * fs)``; only bins other than DC and
    Nyquist are doubled. Band power is ``sum(PSD[inclusive_band]) * fs/NFFT``.
    Zero padding changes the frequency grid, not the normalization by N.
    """
    values, fs, band_array = _validated_inputs(windows, fs, bands)
    n_samples = values.shape[-1]
    nfft = 1 << (n_samples - 1).bit_length()
    spectrum = np.fft.rfft(values, n=nfft, axis=-1)
    psd = np.abs(spectrum) ** 2 / (n_samples * fs)
    psd[..., 1:-1] *= 2.0
    frequencies = np.arange(nfft // 2 + 1, dtype=np.float64) * (fs / nfft)
    powers = np.stack(
        [
            psd[..., (frequencies >= low) & (frequencies <= high)].sum(axis=-1)
            * (fs / nfft)
            for low, high in band_array
        ],
        axis=-1,
    )
    return powers


def differential_entropy(
    windows: np.ndarray,
    fs: float = 500,
    bands: Sequence[tuple[float, float]] = DEFAULT_BANDS,
) -> np.ndarray:
    """Return V1 DE features in channel-major, band-minor order.

    For the default three bands, output shape is ``(n_windows, 57)``. Powers
    strictly above 1e-15 use ``0.5 * log2(2*pi*e*power)``; all other powers,
    including equality, return -50 exactly, matching the V1 MATLAB branch.
    """
    powers = band_power(windows, fs=fs, bands=bands)
    features = np.full_like(powers, -50.0)
    valid = powers > POWER_FLOOR
    features[valid] = 0.5 * np.log2(2.0 * np.pi * np.e * powers[valid])
    return features.reshape(powers.shape[0], N_CHANNELS * powers.shape[-1])


def log_energy_entropy(
    windows: np.ndarray,
    fs: float = 500,
    bands: Sequence[tuple[float, float]] = DEFAULT_BANDS,
) -> np.ndarray:
    """Return the V1 LEE compatibility feature, not an independent feature set.

    LEE is ``log10(N * band_power)`` above the energy floor 1e-15, otherwise
    -15. With fixed N and away from the respective clipping thresholds, LEE
    is an affine transformation of DE. Output ordering matches DE.
    """
    powers = band_power(windows, fs=fs, bands=bands)
    energies = powers * np.asarray(windows).shape[-1]
    features = np.full_like(energies, -15.0)
    valid = energies > POWER_FLOOR
    features[valid] = np.log10(energies[valid])
    return features.reshape(powers.shape[0], N_CHANNELS * powers.shape[-1])
