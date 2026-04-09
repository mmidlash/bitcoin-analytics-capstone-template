"""
Model 3: Cycle-Aware Gradient Boosting with Exchange Flow Dynamics (CPGB).

Extends Model 2's Regime-Aware RF with three novel contributions:

1. Bitcoin Halving Cycle Phase Encoding  (onchain_features.py)
   Bitcoin's 4-year supply-reduction cycle creates repeating price dynamics.
   Phase is encoded as sin/cos of the fraction of the 4-year (1461-day)
   cycle elapsed since the most recent halving. Only halvings that have
   already occurred by date t are used — the exact block date is not
   knowable in advance, so future halvings are masked in the computation.

   Why this matters: the MVRV z-score captures absolute valuation but is
   agnostic to where in the halving cycle we sit. A given MVRV level near a
   halving bottom carries different forward-return odds than the same MVRV
   level 30 months post-halving approaching a bull-run peak.

2. Exchange Flow Signal  (onchain_features.py)
   Net direction of Bitcoin movement on/off exchanges (FlowInExNtv vs
   FlowOutExNtv) and the velocity of change in exchange supply fraction
   (SplyExNtv / SplyCur). Together these capture whether large holders
   are accumulating or distributing at the margin — information orthogonal
   to realized-value metrics like MVRV.

   Rule-based composite (no ML required):
     net_flow_zscore < 0  → net outflows → accumulation → buy signal
     supply_velocity < 0  → supply shock → fewer coins to sell → buy signal

3. Gradient Boosting Classifier  (vs Random Forest in Model 2)
   GBM builds trees sequentially, each correcting its predecessors'
   residuals. This gives it an advantage over RF for structured tabular
   data where features interact non-linearly across market regimes. In
   particular, GBM can learn interactions like:
     "MVRV in value zone AND early cycle phase AND negative exchange flow
      → very high probability of 30-day price gain"
   which RF's independent trees are less likely to isolate.

Signal combination:
  value_signal  (MVRV core):      40%
  ma_signal     (MA-200):         10%
  gbm_signal    (GBM classifier): 30%
  exchange_flow (rule-based):     15%
  poly_signal   (Polymarket):      5%

All modulated by regime_instability (same technique as Model 2).

NO LOOKAHEAD: All features are .shift(1) lagged in precompute_features().
training_pipeline.py is the ONLY module that accesses future prices (for
label construction). The only crossing artifact is gbm_walkforward_probs.parquet
— a static series of OOS probabilities with no price information.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from template.prelude_template import load_polymarket_data
from template.model_development_template import (
    _compute_stable_signal,  # noqa: F401 — re-exported for backtest engine
    allocate_sequential_stable,
    _clean_array,
)
from model_2.change_detection import compute_regime_instability
from model_2.polymarket_features import load_polymarket_btc_activity
from model_3.onchain_features import (
    compute_halving_cycle_features,
    compute_exchange_flow_features,
)

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

# MVRV zone thresholds
MVRV_ZONE_DEEP_VALUE = -2.0
MVRV_ZONE_VALUE = -1.0
MVRV_ZONE_CAUTION = 1.5
MVRV_ZONE_DANGER = 2.5

# ── Signal weights (sum to 1.0) ───────────────────────────────────────────────
W_VALUE = 0.40    # MVRV realized-value signal  (dominant, proven signal)
W_MA = 0.10       # 200-day MA trend signal      (reduced vs Model 2; flow covers more)
W_GBM = 0.30      # GBM buy-opportunity signal   (ML layer with cycle awareness)
W_EXCHANGE = 0.15 # Exchange flow composite      (novel; supply/demand at margin)
W_POLY = 0.05     # Polymarket activity signal   (limited history; modest weight)

# Regime instability dampening (30 % max reduction at instability = 1.0)
REGIME_DAMPENING = 0.30

DYNAMIC_STRENGTH = 4.0
ADJUSTMENT_CLIP_LOW = -5.0
ADJUSTMENT_CLIP_HIGH = 10.0  # Tighter upper bound than Model 2 (12→10)

# Features fed into the GBM (NOT including gbm_buy_prob — that is the output)
_GBM_FEATURE_COLS: list[str] = [
    # --- from Model 2 ---
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
    # --- new in Model 3 ---
    "btc_return_14d",          # 14-day momentum (fills gap between 7d & 30d)
    "btc_return_60d",          # 60-day momentum (medium-term trend context)
    "halving_cycle_sin",       # position in 4-year halving cycle (cyclic encoding)
    "halving_cycle_cos",       # position in 4-year halving cycle (cyclic encoding)
    "halving_days_since",      # raw days since last halving (monotone within cycle)
    "net_flow_zscore",         # exchange inflow/outflow balance
    "supply_velocity_zscore",  # rate of change in exchange supply fraction
]

FEATS: list[str] = _GBM_FEATURE_COLS + ["gbm_buy_prob"]

GBM_WALKFORWARD_PATH = Path(__file__).parent / "gbm_walkforward_probs.parquet"

# =============================================================================
# Walk-forward probabilities — loaded once at import time
# =============================================================================

# Each value in this series was produced by a GBM trained only on data
# *before* that date, making every historical gbm_buy_prob strictly OOS.
_WALKFORWARD_PROBS: pd.Series | None = None


def _load_walkforward_probs() -> None:
    """Load the walk-forward probability series from disk."""
    global _WALKFORWARD_PROBS
    if GBM_WALKFORWARD_PATH.exists():
        try:
            df = pd.read_parquet(GBM_WALKFORWARD_PATH)
            _WALKFORWARD_PROBS = df["gbm_buy_prob_walkforward"]
            logging.info(
                f"Walk-forward probs loaded from {GBM_WALKFORWARD_PATH} "
                f"({len(_WALKFORWARD_PROBS):,} dates, "
                f"{_WALKFORWARD_PROBS.index.min().date()} to "
                f"{_WALKFORWARD_PROBS.index.max().date()})"
            )
        except Exception as exc:
            logging.warning(
                f"Could not load walk-forward probs ({exc}). "
                "gbm_buy_prob will fall back to 0.5 (neutral)."
            )
            _WALKFORWARD_PROBS = None
    else:
        logging.info(
            f"Walk-forward probs not found at {GBM_WALKFORWARD_PATH}. "
            "Run `python -m model_3.training_pipeline` to generate."
        )
        _WALKFORWARD_PROBS = None


_load_walkforward_probs()


def _compute_gbm_buy_prob(features: pd.DataFrame) -> pd.Series:
    """
    Compute P(30-day forward return > 0) for each row.

    Uses walk-forward probabilities from gbm_walkforward_probs.parquet.
    Each value was produced by a GBM trained strictly before that date,
    so every historical prediction is genuinely out-of-sample.
    Falls back to 0.5 (neutral) for any dates not covered.

    Args:
        features: DataFrame with columns matching _GBM_FEATURE_COLS,
                  already lagged by 1 day.

    Returns:
        Series of probabilities in [0, 1] indexed by features.index.
    """
    result = pd.Series(0.5, index=features.index, name="gbm_buy_prob")

    if _WALKFORWARD_PROBS is not None:
        overlap = features.index.intersection(_WALKFORWARD_PROBS.index)
        if len(overlap) > 0:
            result.loc[overlap] = _WALKFORWARD_PROBS.reindex(overlap).values
            logging.debug(
                f"Walk-forward probs applied for {len(overlap):,} of "
                f"{len(features):,} dates"
            )

    return result


# =============================================================================
# Helper Functions (shared with Model 2 pattern)
# =============================================================================


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return ((series - mean) / std).fillna(0.0)


def _classify_mvrv_zone(mvrv_zscore: np.ndarray) -> np.ndarray:
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
    z_signal = -mvrv_zscore / 4.0
    ma_signal = -price_vs_ma
    gradient_alignment = np.where(
        z_signal < 0,
        np.where(mvrv_gradient > 0, 1.0, 0.5),
        np.where(mvrv_gradient < 0, 1.0, 0.5),
    )
    signals = np.stack([z_signal, ma_signal], axis=0)
    agreement = 1.0 - np.clip(signals.std(axis=0) / 1.0, 0.0, 1.0)
    return np.clip(agreement * 0.7 + gradient_alignment * 0.3, 0.0, 1.0)


# =============================================================================
# Feature Engineering
# =============================================================================


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all model features and return a single aligned DataFrame.

    Features computed (all lagged 1 day via .shift(1) before return):

    Inherited from Model 2:
      price_vs_ma              — distance from 200-day MA in [-1, 1]
      mvrv_zscore              — MVRV z-score (365-day window), clipped [-4, 4]
      mvrv_gradient            — smoothed 30-day trend in [-1, 1]
      mvrv_acceleration        — 14-day second derivative in [-1, 1]
      mvrv_zone                — discrete zone {-2, -1, 0, 1, 2}
      mvrv_volatility          — MVRV volatility percentile in [0, 1]
      signal_confidence        — MVRV/MA agreement score in [0, 1]
      regime_instability       — distributional shift score in [0, 1]
      polymarket_activity_zscore — trade-count expanding z-score
      btc_return_7d            — 7-day log return
      btc_return_30d           — 30-day log return
      hash_rate_zscore         — 90-day rolling z-score of HashRate
      tx_count_zscore          — 30-day rolling z-score of TxCnt

    New in Model 3:
      btc_return_14d           — 14-day log return
      btc_return_60d           — 60-day log return
      halving_cycle_sin        — sin(2π x cycle phase) in [-1, 1]
      halving_cycle_cos        — cos(2π x cycle phase) in [-1, 1]
      halving_days_since       — days since most recent halving
      net_flow_zscore          — exchange inflow/outflow balance z-score
      supply_velocity_zscore   — z-scored 30d rate of change in exchange supply%
      exchange_flow_signal     — composite rule-based signal in [-1, 1]
      gbm_buy_prob             — GBM P(30d return > 0), 0.5 if no model

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
        mvrv_gradient = np.tanh(
            gradient_raw.ewm(span=MVRV_GRADIENT_WINDOW, adjust=False).mean() * 2.0
        ).fillna(0.0)

        accel_raw = mvrv_gradient.diff(MVRV_ACCEL_WINDOW)
        mvrv_acceleration = np.tanh(
            accel_raw.ewm(span=MVRV_ACCEL_WINDOW, adjust=False).mean() * 3.0
        ).fillna(0.0)

        mvrv_zone = pd.Series(
            _classify_mvrv_zone(mvrv_z.values), index=mvrv_z.index
        )
        mvrv_volatility = _compute_mvrv_volatility(mvrv_z, MVRV_VOLATILITY_WINDOW)
        regime_instability = compute_regime_instability(mvrv_z)
    else:
        mvrv_z = pd.Series(0.0, index=price.index)
        mvrv_gradient = pd.Series(0.0, index=price.index)
        mvrv_acceleration = pd.Series(0.0, index=price.index)
        mvrv_zone = pd.Series(0, index=price.index)
        mvrv_volatility = pd.Series(0.5, index=price.index)
        regime_instability = pd.Series(0.3, index=price.index)

    # ── Multi-timeframe price momentum ────────────────────────────────────────
    # Log returns are symmetric around zero and better behaved than % returns
    btc_return_7d = np.log(price / price.shift(7)).fillna(0.0)
    btc_return_14d = np.log(price / price.shift(14)).fillna(0.0)
    btc_return_30d = np.log(price / price.shift(30)).fillna(0.0)
    btc_return_60d = np.log(price / price.shift(60)).fillna(0.0)

    # ── On-chain network features ─────────────────────────────────────────────
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

    # ── Polymarket activity (inherited from Model 2) ──────────────────────────
    try:
        poly_data = load_polymarket_data()
        poly_feats = load_polymarket_btc_activity(poly_data, price.index)
        polymarket_activity_zscore = poly_feats["polymarket_activity_zscore"]
    except Exception as exc:
        logging.warning(f"Polymarket features unavailable ({exc}). Using 0.0.")
        polymarket_activity_zscore = pd.Series(0.0, index=price.index)

    # ── NEW: Bitcoin halving cycle phase ─────────────────────────────────────
    halving_feats = compute_halving_cycle_features(price.index)

    # ── NEW: Exchange flow and supply dynamics ────────────────────────────────
    flow_feats = compute_exchange_flow_features(df, price.index)

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
            "btc_return_14d": btc_return_14d,
            "btc_return_30d": btc_return_30d,
            "btc_return_60d": btc_return_60d,
            "hash_rate_zscore": hash_rate_zscore,
            "tx_count_zscore": tx_count_zscore,
        },
        index=price.index,
    )

    # Merge in new feature groups
    features = features.join(halving_feats, how="left")
    features = features.join(flow_feats, how="left")

    # ── Apply 1-day lag to all signal columns (SINGLE lookahead guard) ────────
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
        "btc_return_14d",
        "btc_return_30d",
        "btc_return_60d",
        "hash_rate_zscore",
        "tx_count_zscore",
        "halving_cycle_sin",
        "halving_cycle_cos",
        "halving_days_since",
        "net_flow_zscore",
        "supply_velocity_zscore",
        "exchange_flow_signal",
    ]
    features[signal_cols] = features[signal_cols].shift(1)

    # ── Fill defaults for early-period NaNs ───────────────────────────────────
    features["mvrv_zone"] = features["mvrv_zone"].fillna(0)
    features["mvrv_volatility"] = features["mvrv_volatility"].fillna(0.5)
    features["regime_instability"] = features["regime_instability"].fillna(0.3)
    features["polymarket_activity_zscore"] = features["polymarket_activity_zscore"].fillna(0.0)
    features["exchange_flow_signal"] = features["exchange_flow_signal"].fillna(0.0)
    features = features.fillna(0.0)

    # ── Signal confidence (computed on already-lagged values) ─────────────────
    features["signal_confidence"] = _compute_signal_confidence(
        features["mvrv_zscore"].values,
        features["mvrv_gradient"].values,
        features["price_vs_ma"].values,
    )

    # ── GBM buy probability (computed on lagged features — no lookahead) ──────
    features["gbm_buy_prob"] = _compute_gbm_buy_prob(features)

    return features


# =============================================================================
# Signal Construction
# =============================================================================


def _compute_asymmetric_extreme_boost(mvrv_zscore: np.ndarray) -> np.ndarray:
    """Asymmetric boost for extreme MVRV values (identical to Models 1 & 2)."""
    boost = np.zeros_like(mvrv_zscore)

    deep = mvrv_zscore < MVRV_ZONE_DEEP_VALUE
    boost = np.where(deep, 0.8 * (mvrv_zscore - MVRV_ZONE_DEEP_VALUE) ** 2 + 0.5, boost)

    value = (mvrv_zscore >= MVRV_ZONE_DEEP_VALUE) & (mvrv_zscore < MVRV_ZONE_VALUE)
    boost = np.where(value, -0.5 * mvrv_zscore, boost)

    caution = (mvrv_zscore >= MVRV_ZONE_CAUTION) & (mvrv_zscore < MVRV_ZONE_DANGER)
    boost = np.where(caution, -0.3 * (mvrv_zscore - MVRV_ZONE_CAUTION), boost)

    danger = mvrv_zscore >= MVRV_ZONE_DANGER
    boost = np.where(
        danger, -0.5 * (mvrv_zscore - MVRV_ZONE_DANGER) ** 2 - 0.3, boost
    )
    return boost


def _compute_adaptive_trend_modifier(
    mvrv_gradient: np.ndarray, mvrv_zscore: np.ndarray
) -> np.ndarray:
    """Adaptive MA trend modifier (identical to Models 1 & 2)."""
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


def _compute_acceleration_modifier(
    mvrv_acceleration: np.ndarray, mvrv_gradient: np.ndarray
) -> np.ndarray:
    """Momentum modifier in [0.5, 1.5] (identical to Model 2)."""
    same_direction = (mvrv_acceleration * mvrv_gradient) > 0
    modifier = np.where(
        same_direction,
        1.0 + 0.3 * np.abs(mvrv_acceleration),
        1.0 - 0.2 * np.abs(mvrv_acceleration),
    )
    return np.clip(modifier, 0.5, 1.5)


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
    gbm_buy_prob: np.ndarray,
    regime_instability: np.ndarray,
    polymarket_activity_zscore: np.ndarray,
    exchange_flow_signal: np.ndarray,
) -> np.ndarray:
    """
    Compute per-day weight multipliers from all signal inputs.

    Combined signal formula:
      combined = (value_signal  * W_VALUE
                + ma_signal     * W_MA
                + gbm_signal    * W_GBM
                + flow_signal   * W_EXCHANGE
                + poly_signal   * W_POLY)
               * stability_multiplier    # regime dampener
               * accel_modifier          # momentum
               * confidence_boost        # signal agreement
               * vol_dampener            # high-volatility caution

    Then: exp(clip(combined x DYNAMIC_STRENGTH, LOW, HIGH)).

    All input arrays must be the same length and already lagged 1 day.

    Returns:
        Array of positive multipliers (centered around 1.0).
    """
    # 1. MVRV value signal with asymmetric extreme boost
    extreme_boost = _compute_asymmetric_extreme_boost(mvrv_zscore)
    value_signal = -mvrv_zscore + extreme_boost

    # 2. MA-200 signal with adaptive trend modulation
    trend_modifier = _compute_adaptive_trend_modifier(mvrv_gradient, mvrv_zscore)
    ma_signal = -price_vs_ma * trend_modifier

    # 3. GBM signal: recentered buy probability → [-1, +1]
    #    0.5 → 0.0 (neutral), 1.0 → +1.0 (strong buy), 0.0 → -1.0 (avoid)
    gbm_signal = (gbm_buy_prob - 0.5) * 2.0

    # 4. Exchange flow signal: already in [-1, 1] (outflows = positive)
    flow_signal = exchange_flow_signal  # already computed & bounded in onchain_features

    # 5. Polymarket activity: tanh-smoothed z-score
    poly_signal = np.tanh(polymarket_activity_zscore * 0.5)

    # Weighted combination
    combined = (
        value_signal  * W_VALUE
        + ma_signal   * W_MA
        + gbm_signal  * W_GBM
        + flow_signal * W_EXCHANGE
        + poly_signal * W_POLY
    )

    # 6. Regime instability dampener (identical to Model 2)
    stability = 1.0 - regime_instability * REGIME_DAMPENING
    combined = combined * stability

    # 7. Acceleration modifier (momentum confirmation)
    accel_modifier = _compute_acceleration_modifier(mvrv_acceleration, mvrv_gradient)
    accel_subtle = 0.85 + 0.30 * (accel_modifier - 0.5) / 0.5
    combined = combined * np.clip(accel_subtle, 0.85, 1.15)

    # 8. Confidence boost (amplify when multiple signals agree)
    confidence_boost = np.where(
        signal_confidence > 0.7,
        1.0 + 0.15 * (signal_confidence - 0.7) / 0.3,
        1.0,
    )
    combined = combined * confidence_boost

    # 9. Volatility dampening (top 20% MVRV volatility → reduce signal)
    vol_damp = np.where(
        mvrv_volatility > 0.8,
        1.0 - MVRV_VOLATILITY_DAMPENING * (mvrv_volatility - 0.8) / 0.2,
        1.0,
    )
    combined = combined * vol_damp

    # 10. Scale, clip, exponentiate
    adjustment = np.clip(combined * DYNAMIC_STRENGTH, ADJUSTMENT_CLIP_LOW, ADJUSTMENT_CLIP_HIGH)
    multiplier = np.exp(adjustment)
    return np.where(np.isfinite(multiplier), multiplier, 1.0)


# =============================================================================
# Weight Computation API
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

    def _get(col: str, neutral: float = 0.0) -> np.ndarray:
        if col in df.columns:
            return _clean_array(df[col].values)
        return np.full(n, neutral)

    price_vs_ma = _get("price_vs_ma")
    mvrv_zscore = _get("mvrv_zscore")
    mvrv_gradient = _get("mvrv_gradient")
    mvrv_acceleration = _get("mvrv_acceleration")
    # Replace zeros with neutral defaults for percentage-type features
    mvrv_volatility = np.where(_get("mvrv_volatility") == 0.0, 0.5, _get("mvrv_volatility"))
    signal_confidence = np.where(_get("signal_confidence") == 0.0, 0.5, _get("signal_confidence"))
    gbm_buy_prob = np.where(_get("gbm_buy_prob") == 0.0, 0.5, _get("gbm_buy_prob"))
    regime_instability = _get("regime_instability", neutral=0.3)
    polymarket_activity_zscore = _get("polymarket_activity_zscore")
    exchange_flow_signal = _get("exchange_flow_signal")

    dyn = compute_dynamic_multiplier(
        price_vs_ma=price_vs_ma,
        mvrv_zscore=mvrv_zscore,
        mvrv_gradient=mvrv_gradient,
        mvrv_acceleration=mvrv_acceleration,
        mvrv_volatility=mvrv_volatility,
        signal_confidence=signal_confidence,
        gbm_buy_prob=gbm_buy_prob,
        regime_instability=regime_instability,
        polymarket_activity_zscore=polymarket_activity_zscore,
        exchange_flow_signal=exchange_flow_signal,
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
            {col: 0.0 for col in features_df.columns}, index=missing
        )
        placeholder["mvrv_zone"] = 0
        placeholder["mvrv_volatility"] = 0.5
        placeholder["signal_confidence"] = 0.5
        placeholder["gbm_buy_prob"] = 0.5
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
