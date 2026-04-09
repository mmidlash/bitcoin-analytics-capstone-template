"""
On-chain feature engineering for Model 3.

Computes two groups of novel features not present in Models 1 or 2:

1. Bitcoin Halving Cycle Phase
   Bitcoin's 4-year supply-halving cycle creates recurring price dynamics.
   Halving dates are hardcoded as historically observed facts (the exact block
   date is not knowable until mined). Only halvings that have already occurred
   by date t are used — future halvings are masked. Phase is encoded as sin/cos
   to preserve the circular nature of the cycle (i.e., day 1 after a halving
   is "close to" the day before the next one).

2. Exchange Flow & Supply Dynamics
   Captures supply/demand pressure from large holders and institutions.
   - Net flow direction: outflows > inflows signals accumulation (bullish)
   - Exchange supply velocity: declining on-exchange supply = supply shock

All computations here are causal (use only past data through row t).
The caller (precompute_features in model_development_3.py) applies an
additional .shift(1) before these features reach the weight multiplier.
"""

import logging

import numpy as np
import pandas as pd

# =============================================================================
# Bitcoin Halving Dates — exact dates recorded after each halving occurred.
# Only halvings already past on a given date t are used (future halvings are masked).
# =============================================================================

BITCOIN_HALVINGS: list[pd.Timestamp] = [
    pd.Timestamp("2012-11-28"),
    pd.Timestamp("2016-07-09"),
    pd.Timestamp("2020-05-11"),
    pd.Timestamp("2024-04-20"),
]

# Approximate BTC halving cycle length in days.
# 1461 = 4 × 365.25 matching the conventional
# understanding of a "4-year Bitcoin cycle."
HALVING_CYCLE_DAYS: float = 1461.0


# =============================================================================
# Halving Cycle Features
# =============================================================================


def compute_halving_cycle_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Compute Bitcoin halving cycle phase features for a DatetimeIndex.

    For each date finds the most recent halving and computes the fraction
    of one full cycle that has elapsed, then encodes it as sin/cos.

    Sin/cos encoding vs raw phase:
    - Preserves the *circular* nature: phase 0.99 is close to 0.0.
    - The pair (sin, cos) uniquely identifies any phase in [0, 1).
    - GBM can learn patterns like "phase 0.3-0.5 post-halving = bull run peak."

    Returns:
        DataFrame with columns:
          halving_cycle_sin  : sin(2π * phase) — peak at phase ~0.25
          halving_cycle_cos  : cos(2π * phase) — 1.0 at phase=0 (just after halving)
          halving_days_since : integer days elapsed since the most recent halving
                               (useful as a standalone diagnostic / monotone signal)
    """
    # Convert to integer days since Unix epoch for vectorized arithmetic
    epoch = pd.Timestamp("1970-01-01")
    date_days = np.array([(d - epoch).days for d in index], dtype=float)
    halving_days = np.array([(h - epoch).days for h in BITCOIN_HALVINGS], dtype=float)

    # Shape: (n_dates, n_halvings) — days from each halving to each date
    days_matrix = date_days[:, None] - halving_days[None, :]

    # Mask future halvings (negative = halving hasn't happened yet)
    days_since = np.where(days_matrix >= 0.0, days_matrix, np.inf)

    # Minimum positive value = days since the most recent halving
    days_since_last = np.min(days_since, axis=1)

    # Pre-first-halving dates: set phase to 0 (treat as "cycle start")
    days_since_last = np.where(np.isinf(days_since_last), 0.0, days_since_last)

    phase = (days_since_last % HALVING_CYCLE_DAYS) / HALVING_CYCLE_DAYS

    logging.debug(
        f"Halving cycle: phase range [{phase.min():.3f}, {phase.max():.3f}], "
        f"days_since_last range [{days_since_last.min():.0f}, {days_since_last.max():.0f}]"
    )

    return pd.DataFrame(
        {
            "halving_cycle_sin": np.sin(2.0 * np.pi * phase),
            "halving_cycle_cos": np.cos(2.0 * np.pi * phase),
            "halving_days_since": days_since_last,
        },
        index=index,
    )


# =============================================================================
# Exchange Flow & Supply Features
# =============================================================================

# Column names in the CoinMetrics dataset
_COL_FLOW_IN = "FlowInExNtv"
_COL_FLOW_OUT = "FlowOutExNtv"
_COL_SPLY_EX = "SplyExNtv"
_COL_SPLY_TOTAL = "SplyCur"

# Smoothing / normalization windows (in days)
_NET_FLOW_SMOOTH = 7      # MA to reduce daily noise before z-scoring
_NET_FLOW_ZSCORE_WIN = 90  # Rolling window for z-score normalization
_SUPPLY_DELTA_WIN = 30     # Window for measuring exchange supply change
_SUPPLY_ZSCORE_WIN = 180   # Longer window to capture full cycle context


def compute_exchange_flow_features(
    df: pd.DataFrame,
    price_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Compute exchange flow and supply dynamics features.

    Signal interpretation:
      net_flow_zscore > 0  : net inflows to exchanges → selling pressure → bearish
      net_flow_zscore < 0  : net outflows from exchanges → accumulation → bullish
      supply_velocity_zscore > 0 : on-exchange supply growing → more coins to sell
      supply_velocity_zscore < 0 : on-exchange supply shrinking → supply shock bullish

    These capture information orthogonal to MVRV:
      MVRV     → long-run realized-value valuation (where is market in macro cycle?)
      ExFlows  → short-run supply/demand pressure  (are large holders selling now?)

    Returns:
        DataFrame with columns:
          net_flow_zscore          : z-scored 7d-smoothed net exchange flow
          supply_velocity_zscore   : z-scored 30d rate-of-change in exchange supply %
          exchange_flow_signal     : composite rule-based signal in [-1, 1]:
                                     positive = bullish (outflows / supply declining)
    """
    neutral = pd.DataFrame(
        {
            "net_flow_zscore": 0.0,
            "supply_velocity_zscore": 0.0,
            "exchange_flow_signal": 0.0,
        },
        index=price_index,
    )

    # Check required columns are present
    missing = [
        c
        for c in [_COL_FLOW_IN, _COL_FLOW_OUT, _COL_SPLY_EX, _COL_SPLY_TOTAL]
        if c not in df.columns
    ]
    if missing:
        logging.warning(
            f"Exchange flow features unavailable — missing columns: {missing}. "
            "Using neutral (0.0) for all exchange flow features."
        )
        return neutral

    try:
        # Align to price index
        flow_in = df[_COL_FLOW_IN].reindex(price_index).ffill().fillna(0.0)
        flow_out = df[_COL_FLOW_OUT].reindex(price_index).ffill().fillna(0.0)
        sply_ex = df[_COL_SPLY_EX].reindex(price_index).ffill()
        sply_total = df[_COL_SPLY_TOTAL].reindex(price_index).ffill()

        # ── Net flow (negative = outflows dominate = bullish) ─────────────────
        net_flow_raw = flow_in - flow_out
        # 7-day smoothing to reduce daily noise
        net_flow_smooth = net_flow_raw.rolling(_NET_FLOW_SMOOTH, min_periods=3).mean()
        # 90-day rolling z-score (causal: only uses past data)
        nf_mean = net_flow_smooth.rolling(_NET_FLOW_ZSCORE_WIN, min_periods=30).mean()
        nf_std = net_flow_smooth.rolling(_NET_FLOW_ZSCORE_WIN, min_periods=30).std().clip(lower=1e-8)
        net_flow_zscore = ((net_flow_smooth - nf_mean) / nf_std).fillna(0.0).clip(-4.0, 4.0)

        # ── Exchange supply velocity (negative = supply shrinking = bullish) ──
        # Supply fraction: % of total BTC sitting on exchanges
        with np.errstate(divide="ignore", invalid="ignore"):
            supply_pct = (sply_ex / sply_total.clip(lower=1e-8)).fillna(0.5)

        # 30-day rate of change in exchange supply fraction
        supply_delta = supply_pct - supply_pct.shift(_SUPPLY_DELTA_WIN)
        # 180-day rolling z-score for context-aware normalization
        sv_mean = supply_delta.rolling(_SUPPLY_ZSCORE_WIN, min_periods=60).mean()
        sv_std = supply_delta.rolling(_SUPPLY_ZSCORE_WIN, min_periods=60).std().clip(lower=1e-10)
        supply_velocity_zscore = ((supply_delta - sv_mean) / sv_std).fillna(0.0).clip(-4.0, 4.0)

        # ── Composite exchange flow signal ────────────────────────────────────
        # Invert signs: negative flow/supply velocity = bullish → positive signal
        flow_signal = np.tanh(-net_flow_zscore * 0.5)        # outflows → positive
        supply_signal = np.tanh(-supply_velocity_zscore * 0.5)  # shrinking supply → positive
        exchange_flow_signal = (0.6 * flow_signal + 0.4 * supply_signal).clip(-1.0, 1.0)

        n_active = int((net_flow_zscore != 0.0).sum())
        logging.info(
            f"Exchange flow features: {n_active} active days, "
            f"net_flow_zscore range [{net_flow_zscore.min():.2f}, {net_flow_zscore.max():.2f}], "
            f"signal range [{exchange_flow_signal.min():.2f}, {exchange_flow_signal.max():.2f}]"
        )

        return pd.DataFrame(
            {
                "net_flow_zscore": net_flow_zscore,
                "supply_velocity_zscore": supply_velocity_zscore,
                "exchange_flow_signal": exchange_flow_signal,
            },
            index=price_index,
        )

    except Exception as exc:
        logging.warning(
            f"Failed to compute exchange flow features ({exc}). "
            "Defaulting to 0.0 for all exchange flow features."
        )
        return neutral
