"""Frozen stage-two feature ablations, computed independently per window.

Inputs are already preprocessed microvolt windows in the stage-one channel
order. This module does not learn parameters or change the input signals. DE
delegates to the original implementation, including inclusive band edges.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .data import CHANNELS
from .features import DEFAULT_BANDS, _validated_inputs, band_power, differential_entropy


def _band_names(kind: str, bands: np.ndarray) -> list[str]:
    return [
        f"{channel}:{kind}:{low:g}-{high:g}Hz"
        for channel in CHANNELS
        for low, high in bands
    ]


def _hjorth(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    if values.shape[-1] < 3:
        raise ValueError("Hjorth features require at least three samples per window.")
    variance = np.var(values, axis=-1, ddof=0)
    difference_variance = np.var(np.diff(values, axis=-1), axis=-1, ddof=0)
    second_difference_variance = np.var(np.diff(values, n=2, axis=-1), axis=-1, ddof=0)
    mobility = np.sqrt(np.divide(
        difference_variance, variance,
        out=np.zeros_like(variance), where=variance > 0,
    ))
    difference_mobility = np.sqrt(np.divide(
        second_difference_variance, difference_variance,
        out=np.zeros_like(variance), where=difference_variance > 0,
    ))
    complexity = np.divide(
        difference_mobility, mobility,
        out=np.zeros_like(variance), where=mobility > 0,
    )
    # Derivatives are sample differences, not derivatives in units of seconds.
    features = np.stack((np.log2(variance + 1e-15), mobility, complexity), axis=-1)
    names = [
        f"{channel}:hjorth:{measure}"
        for channel in CHANNELS
        for measure in ("log2_activity", "mobility", "complexity")
    ]
    return features.reshape(values.shape[0], len(names)), names


def extract_features(
    windows: np.ndarray,
    fs: float = 500,
    kind: str = "de",
    bands: Sequence[tuple[float, float]] = DEFAULT_BANDS,
) -> tuple[np.ndarray, list[str]]:
    """Return a finite feature matrix and its ordered, unique column names.

    ``de`` and ``relative_power`` flatten channel-major, band-minor. Relative
    power divides each band by a *single* inclusive 4--30 Hz integral; because
    adjacent bands share edge bins, their ratios need not sum to one. A zero
    denominator produces zero, without an amplitude-dependent floor.

    ``hjorth`` uses channel-major (log2 activity, mobility, complexity).
    ``de_hjorth`` concatenates the complete DE block then the complete Hjorth
    block. Activity is log2(var + 1e-15); all variances have ddof=0, and any
    zero denominator in mobility or complexity produces zero. Hjorth needs
    three or more samples; the other features accept two or more.
    """
    if kind not in ("de", "relative_power", "hjorth", "de_hjorth"):
        raise ValueError(f"Unknown feature kind: {kind!r}")
    values, fs, band_array = _validated_inputs(windows, fs, bands)
    if kind == "hjorth":
        features, names = _hjorth(values)
    elif kind == "relative_power":
        if np.any(band_array[:, 0] < 4.0) or np.any(band_array[:, 1] > 30.0):
            raise ValueError("relative_power bands must lie within the 4-30 Hz denominator range.")
        powers = band_power(values, fs=fs, bands=band_array)
        denominator = band_power(values, fs=fs, bands=((4.0, 30.0),))
        ratios = np.divide(powers, denominator, out=np.zeros_like(powers), where=denominator > 0)
        names = _band_names(kind, band_array)
        features = ratios.reshape(values.shape[0], len(names))
    else:
        features = differential_entropy(values, fs=fs, bands=band_array)
        names = _band_names("de", band_array)
        if kind == "de_hjorth":
            hjorth, hjorth_names = _hjorth(values)
            features = np.concatenate((features, hjorth), axis=1)
            names.extend(hjorth_names)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate bands would produce duplicate feature names.")
    if not np.isfinite(features).all():
        raise ValueError("Feature calculation produced nonfinite values.")
    return features, names
