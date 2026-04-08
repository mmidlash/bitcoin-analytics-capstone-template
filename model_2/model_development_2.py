"""
Model 2: Regime-Aware RF + Change Detection BTC Accumulation Strategy.

Extends Example 1 (MVRV + MA-200 + Polymarket) with three additions:

1. Random Forest buy-opportunity classifier (rf_buy_prob)
   Predicts whether today is a relatively cheap accumulation day using a
   data-driven combination of on-chain, price momentum, and Polymarket signals.
   Falls back to 0.5 (neutral / no signal) when rf_model.pkl is absent.

2. Regime instability dampener (regime_instability)
   Detects distributional shifts in the MVRV z-score that precede regime
   transitions. When instability is elevated, all signals are dampened to
   prevent overcommitting to a strategy that may no longer apply.
   Addresses the EDA finding that the MA-200 is a lagging regime indicator.

3. EDA-validated Polymarket activity signal (polymarket_activity_zscore)
   Uses daily trade-count z-score (expanding window — no lookahead) from
   high-liquidity BTC prediction markets. Distinct from Example 1's market-
   creation sentiment; this signal was validated at 30d (p=0.029) and
   60d (p=0.011) in the EDA.

Signal weights:
  value_signal (MVRV):        50%
  ma_signal (MA-200):         15%
  rf_signal (RF classifier):  25%
  poly_signal (trade-count):  10%

All modulated by regime_instability, acceleration, confidence, and volatility.

NO LOOKAHEAD: every feature entering the multiplier is lagged by 1 day via
.shift(1) applied in precompute_features(). The RF is a static artifact
(rf_model.pkl) trained offline in training_pipeline.py using future prices
ONLY during label construction — never at inference time.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from template.prelude_template import load_polymarket_data
from template.model_development_template import (
    _compute_stable_signal,
    allocate_sequential_stable,
    _clean_array,
)
from model_2.change_detection import compute_regime_instability
from model_2.polymarket_features import load_polymarket_btc_activity

# =============================================================================
# Constants
# =============================================================================

PRICE_COL = "PriceUSD_coinmetrics"
MVRV_COL = "CapMVRVCur"
HASHRATE_COL = "HashRate"
TXCOUNT_COL = "TxCnt"

MIN_W = 1e-6
MA_WINDOW = 200
MVRV_GRADIENT_WINDOW = 30
MVRV_ROLLING_WINDOW = 365
MVRV_ACCEL_WINDOW = 14
MVRV_VOLATILITY_WINDOW = 90
MVRV_VOLATILITY_DAMPENING = 0.2

# MVRV zone thresholds (same as Example 1)
MVRV_ZONE_DEEP_VALUE = -2.0
MVRV_ZONE_VALUE = -1.0
MVRV_ZONE_CAUTION = 1.5
MVRV_ZONE_DANGER = 2.5

# Signal weights in compute_dynamic_multiplier
W_VALUE = 0.50   # MVRV value signal
W_MA = 0.15      # MA-200 signal
W_RF = 0.25      # RF buy-opportunity probability
W_POLY = 0.10    # Polymarket activity z-score

# Regime instability dampening factor (max reduction when instability = 1.0)
REGIME_DAMPENING = 0.30

# Multiplier scaling — tighter upper clip (vs Example 1's 100) to prevent
# extreme weight concentration when multiple signals simultaneously agree.
DYNAMIC_STRENGTH = 5.0
ADJUSTMENT_CLIP_LOW = -5.0
ADJUSTMENT_CLIP_HIGH = 12.0

# Features used as RF inputs (NOT including rf_buy_prob — that is the output)
_RF_FEATURE_COLS: list[str] = [
    "price_vs_ma",
    "mvrv_zscore",
    "mvrv_gradient",
    "mvrv_acceleration",
    "mvrv_zone",
    "mvrv_volatility",
    "signal_confidence",
    "regime_instability",
    "polymarket_activity_zscore",
    "btc_return_7d",
    "btc_return_30d",
    "hash_rate_zscore",
    "tx_count_zscore",
]

FEATS: list[str] = _RF_FEATURE_COLS + ["rf_buy_prob"]

RF_MODEL_PATH = Path(__file__).parent / "rf_model.pkl"

# =============================================================================
# RF Model — module-level singleton, loaded once at import time
# =============================================================================

_RF_MODEL = None


def _load_rf_model() -> None:
    """Load RF model from disk into the module-level singleton.

    Called once at import. If the model file is absent (e.g., before training),
    _RF_MODEL stays None and rf_buy_prob defaults to 0.5 throughout.
    """
    global _RF_MODEL
    if RF_MODEL_PATH.exists():
        try:
            _RF_MODEL = joblib.load(RF_MODEL_PATH)
            logging.info(f"RF model loaded from {RF_MODEL_PATH}")
        except Exception as exc:
            logging.warning(
                f"Could not load RF model ({exc}). "
                "rf_buy_prob will be 0.5 (neutral) for all dates."
            )
            _RF_MODEL = None
    else:
        logging.info(
            f"RF model not found at {RF_MODEL_PATH}. "
            "rf_buy_prob will be 0.5 (neutral). "
            "Run `python -m model_2.training_pipeline` to train."
        )
        _RF_MODEL = None


_load_rf_model()


def _compute_rf_buy_prob(features: pd.DataFrame) -> pd.Series:
    """
    Apply the RF model to the lagged features DataFrame.

    Returns P(good buying day) for each row. When the RF model is not
    available, returns 0.5 everywhere (no signal — RF contributes nothing
    to the multiplier).

    Args:
        features: DataFrame with columns matching _RF_FEATURE_COLS,
                  already lagged by 1 day.

    Returns:
        Series of probabilities in [0, 1] indexed by features.index.
    """
    if _RF_MODEL is None:
        return pd.Series(0.5, index=features.index, name="rf_buy_prob")

    # Use only columns the RF was trained on, fill any missing with 0.0
    rf_cols = [c for c in _RF_FEATURE_COLS if c in features.columns]
    X = features[rf_cols].fillna(0.0).values.astype(float)

    try:
        probs = _RF_MODEL.predict_proba(X)[:, 1]
    except Exception as exc:
        logging.warning(f"RF prediction failed ({exc}). Defaulting to 0.5.")
        probs = np.full(len(features), 0.5)

    return pd.Series(probs, index=features.index, name="rf_buy_prob")


# =============================================================================
# Helper Functions (signal construction)
# =============================================================================


def _zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score with min_periods = window // 2."""
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return ((series - mean) / std).fillna(0.0)


def _classify_mvrv_zone(mvrv_zscore: np.ndarray) -> np.ndarray:
    """Discrete zone classification: {-2, -1, 0, 1, 2}."""
    return np.select(
        [
            mvrv_zscore < MVRV_ZONE_DEEP_VALUE,
            mvrv_zscore < MVRV_ZONE_VALUE,
            mvrv_zscore < MVRV_ZONE_CAUTION,
            mvrv_zscore < MVRV_ZONE_DANGER,
        ],
        [-2, -1, 0, 1],
        default=2,
    )


def _compute_mvrv_volatility(mvrv_zscore: pd.Series, window: int) -> pd.Series:
    """Rolling volatility percentile of MVRV z-score in [0, 1]."""
    vol = mvrv_zscore.rolling(window, min_periods=window // 4).std()
    vol_pct = vol.rolling(window * 4, min_periods=window).apply(
        lambda x: (x.iloc[-1] > x[:-1]).sum() / max(len(x) - 1, 1)
        if len(x) > 1
        else 0.5,
        raw=False,
    )
    return vol_pct.fillna(0.5)


def _compute_signal_confidence(
    mvrv_zscore: np.ndarray,
    mvrv_gradient: np.ndarray,
    price_vs_ma: np.ndarray,
) -> np.ndarray:
    """Signal agreement score in [0, 1]. Higher = more signals agree."""
    z_signal = -mvrv_zscore / 4.0
    ma_signal = -price_vs_ma

    gradient_alignment = np.where(
        z_signal < 0,
        np.where(mvrv_gradient > 0, 1.0, 0.5),
        np.where(mvrv_gradient < 0, 1.0, 0.5),
    )

    signals = np.stack([z_signal, ma_signal], axis=0)
    signal_std = signals.std(axis=0)
    agreement = 1.0 - np.clip(signal_std / 1.0, 0.0, 1.0)

    confidence = agreement * 0.7 + gradient_alignment * 0.3
    return np.clip(confidence, 0.0, 1.0)


def _compute_asymmetric_extreme_boost(mvrv_zscore: np.ndarray) -> np.ndarray:
    """Asymmetric boost for extreme MVRV values (identical to Example 1)."""
    boost = np.zeros_like(mvrv_zscore)

    deep = mvrv_zscore < MVRV_ZONE_DEEP_VALUE
    boost = np.where(
        deep,
        0.8 * (mvrv_zscore - MVRV_ZONE_DEEP_VALUE) ** 2 + 0.5,
        boost,
    )

    value = (mvrv_zscore >= MVRV_ZONE_DEEP_VALUE) & (mvrv_zscore < MVRV_ZONE_VALUE)
    boost = np.where(value, -0.5 * mvrv_zscore, boost)

    caution = (mvrv_zscore >= MVRV_ZONE_CAUTION) & (mvrv_zscore < MVRV_ZONE_DANGER)
    boost = np.where(caution, -0.3 * (mvrv_zscore - MVRV_ZONE_CAUTION), boost)

    danger = mvrv_zscore >= MVRV_ZONE_DANGER
    boost = np.where(
        danger,
        -0.5 * (mvrv_zscore - MVRV_ZONE_DANGER) ** 2 - 0.3,
        boost,
    )
    return boost


def _compute_acceleration_modifier(
    mvrv_acceleration: np.ndarray,
    mvrv_gradient: np.ndarray,
) -> np.ndarray:
    """Momentum modifier in [0.5, 1.5] (identical to Example 1)."""
    same_direction = (mvrv_acceleration * mvrv_gradient) > 0
    modifier = np.where(
        same_direction,
        1.0 + 0.3 * np.abs(mvrv_acceleration),
        1.0 - 0.2 * np.abs(mvrv_acceleration),
    )
    return np.clip(modifier, 0.5, 1.5)


def _compute_adaptive_trend_modifier(
    mvrv_gradient: np.ndarray,
    mvrv_zscore: np.ndarray,
) -> np.ndarray:
    """Adaptive MA trend modifier (identical to Example 1)."""
    threshold = np.where(
        mvrv_zscore < -1, 0.1, np.where(mvrv_zscore > 1.5, 0.4, 0.2)
    )
    modifier = np.where(
        mvrv_gradient > threshold,
        1.0 + 0.5 * np.minimum(mvrv_gradient, 1.0),
        np.where(
            mvrv_gradient < -threshold,
            0.3 + 0.2 * (1 + mvrv_gradient),
            1.0,
        ),
    )
    return np.clip(modifier, 0.3, 1.5)


# =============================================================================
# Feature Engineering
# =============================================================================


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all model features and return a single aligned DataFrame.

    Features computed (all lagged 1 day via .shift(1) before return):
      price_vs_ma            — distance from 200-day MA in [-1, 1]
      mvrv_zscore            — MVRV z-score (365-day window), clipped [-4, 4]
      mvrv_gradient          — smoothed 30-day MVRV trend in [-1, 1]
      mvrv_acceleration      — 14-day second derivative in [-1, 1]
      mvrv_zone              — discrete zone in {-2, -1, 0, 1, 2}
      mvrv_volatility        — MVRV volatility percentile in [0, 1]
      signal_confidence      — MVRV/MA agreement score in [0, 1]
      regime_instability     — distributional shift score in [0, 1]
      polymarket_activity_zscore — expanding-window trade-count z-score
      btc_return_7d          — 7-day log return
      btc_return_30d         — 30-day log return
      hash_rate_zscore       — 90-day rolling z-score of HashRate
      tx_count_zscore        — 30-day rolling z-score of TxCnt
      rf_buy_prob            — RF P(good buying day), or 0.5 if no model

    The RF (rf_buy_prob) is computed AFTER all other features are lagged.
    It receives features[t-1] at row t and outputs P(buy opportunity at t).
    This is the same lag contract as training: features lagged, labels forward.

    Args:
        df: DataFrame from load_data() with time index.

    Returns:
        DataFrame with price column and all features, indexed by date.
    """
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found. Available: {list(df.columns)}")

    # ── Price and MA ──────────────────────────────────────────────────────────
    price = df[PRICE_COL].loc["2010-07-18":].copy()
    ma = price.rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        price_vs_ma = ((price / ma) - 1.0).clip(-1.0, 1.0).fillna(0.0)

    # ── MVRV features ─────────────────────────────────────────────────────────
    if MVRV_COL in df.columns:
        mvrv = df[MVRV_COL].loc[price.index]
        mvrv_z = _zscore(mvrv, MVRV_ROLLING_WINDOW).clip(-4.0, 4.0)

        gradient_raw = mvrv_z.diff(MVRV_GRADIENT_WINDOW)
        gradient_smooth = gradient_raw.ewm(span=MVRV_GRADIENT_WINDOW, adjust=False).mean()
        mvrv_gradient = np.tanh(gradient_smooth * 2.0).fillna(0.0)

        accel_raw = mvrv_gradient.diff(MVRV_ACCEL_WINDOW)
        mvrv_acceleration = np.tanh(
            accel_raw.ewm(span=MVRV_ACCEL_WINDOW, adjust=False).mean() * 3.0
        ).fillna(0.0)

        mvrv_zone = pd.Series(
            _classify_mvrv_zone(mvrv_z.values), index=mvrv_z.index
        )
        mvrv_volatility = _compute_mvrv_volatility(mvrv_z, MVRV_VOLATILITY_WINDOW)

        # Regime instability — computed on full series, then lagged below
        regime_instability = compute_regime_instability(mvrv_z)
    else:
        mvrv_z = pd.Series(0.0, index=price.index)
        mvrv_gradient = pd.Series(0.0, index=price.index)
        mvrv_acceleration = pd.Series(0.0, index=price.index)
        mvrv_zone = pd.Series(0, index=price.index)
        mvrv_volatility = pd.Series(0.5, index=price.index)
        regime_instability = pd.Series(0.3, index=price.index)

    # ── Price momentum features ───────────────────────────────────────────────
    # Log returns: natural for return series, symmetric around zero
    btc_return_7d = np.log(price / price.shift(7)).fillna(0.0)
    btc_return_30d = np.log(price / price.shift(30)).fillna(0.0)

    # ── On-chain features ─────────────────────────────────────────────────────
    if HASHRATE_COL in df.columns:
        hash_rate_zscore = _zscore(
            df[HASHRATE_COL].loc[price.index].ffill(), window=90
        ).fillna(0.0)
    else:
        hash_rate_zscore = pd.Series(0.0, index=price.index)

    if TXCOUNT_COL in df.columns:
        tx_count_zscore = _zscore(
            df[TXCOUNT_COL].loc[price.index].ffill(), window=30
        ).fillna(0.0)
    else:
        tx_count_zscore = pd.Series(0.0, index=price.index)

    # ── Polymarket activity ───────────────────────────────────────────────────
    try:
        poly_data = load_polymarket_data()
        poly_feats = load_polymarket_btc_activity(poly_data, price.index)
        polymarket_activity_zscore = poly_feats["polymarket_activity_zscore"]
    except Exception as exc:
        logging.warning(f"Polymarket features unavailable ({exc}). Using 0.0.")
        polymarket_activity_zscore = pd.Series(0.0, index=price.index)

    # ── Assemble raw (unlagged) feature DataFrame ─────────────────────────────
    features = pd.DataFrame(
        {
            PRICE_COL: price,
            "price_ma": ma,
            "price_vs_ma": price_vs_ma,
            "mvrv_zscore": mvrv_z,
            "mvrv_gradient": mvrv_gradient,
            "mvrv_acceleration": mvrv_acceleration,
            "mvrv_zone": mvrv_zone,
            "mvrv_volatility": mvrv_volatility,
            "regime_instability": regime_instability,
            "polymarket_activity_zscore": polymarket_activity_zscore,
            "btc_return_7d": btc_return_7d,
            "btc_return_30d": btc_return_30d,
            "hash_rate_zscore": hash_rate_zscore,
            "tx_count_zscore": tx_count_zscore,
        },
        index=price.index,
    )

    # ── Apply 1-day lag to all signal columns ─────────────────────────────────
    # This is the sole lookahead guard for all features.
    # After this shift, features.loc[t] contains information from t-1.
    signal_cols = [
        "price_vs_ma",
        "mvrv_zscore",
        "mvrv_gradient",
        "mvrv_acceleration",
        "mvrv_zone",
        "mvrv_volatility",
        "regime_instability",
        "polymarket_activity_zscore",
        "btc_return_7d",
        "btc_return_30d",
        "hash_rate_zscore",
        "tx_count_zscore",
    ]
    features[signal_cols] = features[signal_cols].shift(1)

    # ── Fill defaults for early-period NaNs ───────────────────────────────────
    features["mvrv_zone"] = features["mvrv_zone"].fillna(0)
    features["mvrv_volatility"] = features["mvrv_volatility"].fillna(0.5)
    features["regime_instability"] = features["regime_instability"].fillna(0.3)
    features["polymarket_activity_zscore"] = features["polymarket_activity_zscore"].fillna(0.0)
    features = features.fillna(0.0)

    # ── Signal confidence (computed on already-lagged values — no lookahead) ──
    features["signal_confidence"] = _compute_signal_confidence(
        features["mvrv_zscore"].values,
        features["mvrv_gradient"].values,
        features["price_vs_ma"].values,
    )

    # ── RF buy probability (computed on lagged features — no lookahead) ───────
    # The RF sees features[t-1] at row t (because all signals were shifted above).
    # rf_buy_prob is NOT in _RF_FEATURE_COLS — it is the RF output, not an input.
    features["rf_buy_prob"] = _compute_rf_buy_prob(features)

    return features


# =============================================================================
# Dynamic Multiplier
# =============================================================================


def compute_dynamic_multiplier(
    price_vs_ma: np.ndarray,
    mvrv_zscore: np.ndarray,
    mvrv_gradient: np.ndarray,
    mvrv_acceleration: np.ndarray,
    mvrv_volatility: np.ndarray,
    signal_confidence: np.ndarray,
    rf_buy_prob: np.ndarray,
    regime_instability: np.ndarray,
    polymarket_activity_zscore: np.ndarray,
) -> np.ndarray:
    """
    Compute per-day weight multipliers from all signal inputs.

    Combined signal formula:
      combined = (value_signal * W_VALUE
                + ma_signal    * W_MA
                + rf_signal    * W_RF
                + poly_signal  * W_POLY)
               * stability_multiplier   # regime-instability dampener

    Then standard modifiers: acceleration, confidence boost, volatility dampening.
    Finally: exp(clip(combined * DYNAMIC_STRENGTH, -5, 12)).

    Args:
        All arrays must be the same length, already lagged 1 day.

    Returns:
        Array of positive multipliers (centered around 1.0).
    """
    # 1. MVRV value signal with asymmetric extreme boost
    extreme_boost = _compute_asymmetric_extreme_boost(mvrv_zscore)
    value_signal = -mvrv_zscore + extreme_boost

    # 2. MA signal with adaptive trend modulation
    trend_modifier = _compute_adaptive_trend_modifier(mvrv_gradient, mvrv_zscore)
    ma_signal = -price_vs_ma * trend_modifier

    # 3. RF signal: buy probability recentered to [-1, +1]
    #    rf_buy_prob = 0.5 → rf_signal = 0.0 (no contribution)
    #    rf_buy_prob = 1.0 → rf_signal = +1.0 (strong buy signal)
    #    rf_buy_prob = 0.0 → rf_signal = -1.0 (strong avoid signal)
    rf_signal = (rf_buy_prob - 0.5) * 2.0

    # 4. Polymarket activity signal
    #    tanh maps the z-score smoothly to [-1, +1]:
    #      z = 0.0 → 0.00 (no signal)
    #      z = 1.5 → 0.64 (activity shock level)
    #      z = 3.0 → 0.91 (extreme activity)
    poly_signal = np.tanh(polymarket_activity_zscore * 0.5)

    # Weighted combination
    combined = (
        value_signal * W_VALUE
        + ma_signal * W_MA
        + rf_signal * W_RF
        + poly_signal * W_POLY
    )

    # 5. Regime instability dampener
    #    When instability = 1.0, all signals are reduced by REGIME_DAMPENING (30%).
    #    When instability = 0.5, reduction is 15%.
    #    Prevents overcommitting during detected regime transitions.
    stability = 1.0 - regime_instability * REGIME_DAMPENING
    combined = combined * stability

    # 6. Acceleration modifier (momentum detection)
    accel_modifier = _compute_acceleration_modifier(mvrv_acceleration, mvrv_gradient)
    accel_subtle = 0.85 + 0.30 * (accel_modifier - 0.5) / 0.5
    combined = combined * np.clip(accel_subtle, 0.85, 1.15)

    # 7. Confidence boost (only when signals strongly agree)
    confidence_boost = np.where(
        signal_confidence > 0.7,
        1.0 + 0.15 * (signal_confidence - 0.7) / 0.3,
        1.0,
    )
    combined = combined * confidence_boost

    # 8. Volatility dampening (only in extreme volatility — top 20%)
    vol_damp = np.where(
        mvrv_volatility > 0.8,
        1.0 - MVRV_VOLATILITY_DAMPENING * (mvrv_volatility - 0.8) / 0.2,
        1.0,
    )
    combined = combined * vol_damp

    # 9. Scale and clip — tighter upper bound (12 vs Example 1's 100) to prevent
    #    extreme concentration when all signals agree simultaneously.
    adjustment = np.clip(combined * DYNAMIC_STRENGTH, ADJUSTMENT_CLIP_LOW, ADJUSTMENT_CLIP_HIGH)
    multiplier = np.exp(adjustment)
    return np.where(np.isfinite(multiplier), multiplier, 1.0)


# =============================================================================
# Weight Computation API  (matches Example 1 interface exactly)
# =============================================================================


def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """
    Compute investment weights for a date window using precomputed features.

    Args:
        features_df:    DataFrame from precompute_features().
        start_date:     Window start (inclusive).
        end_date:       Window end (inclusive).
        n_past:         Number of past/current days (locked). Default: all days.
        locked_weights: Optional pre-computed locked weights.

    Returns:
        Series of weights indexed by date, summing to 1.0.
    """
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n

    # Extract features, replacing any residual NaN with neutral defaults
    def _get(col: str, neutral: float = 0.0) -> np.ndarray:
        if col in df.columns:
            arr = _clean_array(df[col].values)
            return arr
        return np.full(n, neutral)

    price_vs_ma = _get("price_vs_ma")
    mvrv_zscore = _get("mvrv_zscore")
    mvrv_gradient = _get("mvrv_gradient")
    mvrv_acceleration = _get("mvrv_acceleration")
    mvrv_volatility = np.where(_get("mvrv_volatility") == 0, 0.5, _get("mvrv_volatility"))
    signal_confidence = np.where(_get("signal_confidence") == 0, 0.5, _get("signal_confidence"))
    rf_buy_prob = np.where(_get("rf_buy_prob") == 0, 0.5, _get("rf_buy_prob"))
    regime_instability = _get("regime_instability", neutral=0.3)
    polymarket_activity_zscore = _get("polymarket_activity_zscore")

    dyn = compute_dynamic_multiplier(
        price_vs_ma=price_vs_ma,
        mvrv_zscore=mvrv_zscore,
        mvrv_gradient=mvrv_gradient,
        mvrv_acceleration=mvrv_acceleration,
        mvrv_volatility=mvrv_volatility,
        signal_confidence=signal_confidence,
        rf_buy_prob=rf_buy_prob,
        regime_instability=regime_instability,
        polymarket_activity_zscore=polymarket_activity_zscore,
    )

    raw = base * dyn
    n_past_eff = n if n_past is None else n_past
    weights = allocate_sequential_stable(raw, n_past_eff, locked_weights)
    return pd.Series(weights, index=df.index)


def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """
    Compute weights for a date range with lock-on-compute stability.

    Past/current dates use signal-based weights (locked once computed).
    Future dates within the window use uniform weights.

    Args:
        features_df:    DataFrame from precompute_features().
        start_date:     Investment window start.
        end_date:       Investment window end.
        current_date:   Today's date (past/future boundary).
        locked_weights: Optional locked weights from database.

    Returns:
        Series of weights summing to 1.0, indexed by daily dates.
    """
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Extend features with neutral placeholders for future dates
    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame(
            {col: 0.0 for col in features_df.columns},
            index=missing,
        )
        placeholder["mvrv_zone"] = 0
        placeholder["mvrv_volatility"] = 0.5
        placeholder["signal_confidence"] = 0.5
        placeholder["rf_buy_prob"] = 0.5
        placeholder["regime_instability"] = 0.3
        features_df = pd.concat([features_df, placeholder]).sort_index()

    past_end = min(current_date, end_date)
    n_past = (
        len(pd.date_range(start=start_date, end=past_end, freq="D"))
        if start_date <= past_end
        else 0
    )

    weights = compute_weights_fast(
        features_df, start_date, end_date, n_past, locked_weights
    )
    return weights.reindex(full_range, fill_value=0.0)
