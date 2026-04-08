"""
Causal regime instability detection for the Model 2 BTC accumulation strategy.

Computes a regime instability score in [0, 1] at each timestep by comparing
the recent short-window distribution of MVRV z-score to a preceding baseline
window. A high score indicates a potential regime transition.

All computations here are inherently causal: pandas .rolling() at position t
uses only data through t. The caller (precompute_features) applies an
additional .shift(1) so that the signal reaching the multiplier on day t
reflects data through t-1.
"""

import numpy as np
import pandas as pd


def compute_regime_instability(
    series: pd.Series,
    short_window: int = 30,
    long_window: int = 120,
) -> pd.Series:
    """
    Causal regime instability score based on distributional shift detection.

    At each position t (before the 1-day lag applied in precompute_features),
    compares:
      Recent:   values in [t - short_window, t]
      Baseline: values in [t - short_window - long_window, t - short_window - 1]

    The standardized mean shift (how many baseline standard deviations the
    recent mean has moved) is mapped through a sigmoid to [0, 1].

    This detects regime transitions earlier than the MA-200 (which is a lagging
    indicator), addressing the gap identified in the EDA.

    Score interpretation:
      ~0.3  — insufficient history (early-period default)
      ~0.5  — no significant distributional shift
      ~0.73 — shift of ~2 baseline std (elevated instability)
      ~0.88 — shift of ~3 baseline std (high instability)

    Args:
        series: Time series (e.g., MVRV z-score). DatetimeIndex required.
        short_window: Recent window in days (default: 30).
        long_window:  Baseline window in days (default: 120).

    Returns:
        Series of instability scores in [0, 1], same index as input.
        Early rows (insufficient history) default to 0.3.
    """
    if series.empty:
        return series.copy()

    # Recent window: rolling mean over [t - short_window, t]
    recent_mean = series.rolling(
        short_window, min_periods=max(short_window // 3, 5)
    ).mean()

    # Baseline window: shift the series forward by short_window so that
    # rolling(long_window) at position t covers [t - short_window - long_window,
    # t - short_window]. These two windows are non-overlapping by construction.
    baseline_series = series.shift(short_window)
    baseline_mean = baseline_series.rolling(
        long_window, min_periods=long_window // 3
    ).mean()
    baseline_std = baseline_series.rolling(
        long_window, min_periods=long_window // 3
    ).std()

    # Standardized mean shift (absolute — we care about magnitude, not direction)
    stable_std = baseline_std.clip(lower=1e-8)
    mean_shift = (recent_mean - baseline_mean).abs() / stable_std

    # Sigmoid normalization
    # sigmoid(1.5 * (x - 1)):
    #   x = 0.5 → 0.39   (minimal instability)
    #   x = 1.0 → 0.50   (moderate)
    #   x = 2.0 → 0.69   (elevated)
    #   x = 3.0 → 0.83   (high)
    instability = 1.0 / (1.0 + np.exp(-1.5 * (mean_shift - 1.0)))

    # Default early period (insufficient data) to 0.3 (slightly below neutral)
    return instability.fillna(0.3)
