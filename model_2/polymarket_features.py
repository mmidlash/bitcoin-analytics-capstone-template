"""
Polymarket BTC activity feature computation for Model 2.

Extracts a daily trade-count z-score from high-liquidity Polymarket BTC
prediction markets. Uses an expanding-window normalization to ensure the
z-score is strictly causal (no lookahead bias).

EDA finding (validated): Activity shocks (z > 1.5) predict +5.4% at 30 days
(p = 0.029) and +8.5% at 60 days (p = 0.011).

Critical fix vs the EDA notebook: The EDA computed the z-score using global
mean and std (future data included). This module uses an expanding window
so the normalization at day t uses only data through day t.
"""

import logging

import numpy as np
import pandas as pd


# Minimum USD volume to include a market
_BTC_MARKET_MIN_VOLUME = 1_000_000

# Keywords for determining market orientation
_BULLISH_KEYWORDS = frozenset(["above", "reach", "hit", "over", "exceed"])
_BEARISH_KEYWORDS = frozenset(["below", "under"])


def _identify_btc_markets(markets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to high-liquidity BTC price-target prediction markets.

    Returns DataFrame with columns [market_id, orientation]
    where orientation is 1 (bullish) or -1 (bearish).
    Markets with ambiguous orientation are excluded.
    """
    df = markets_df.copy()
    df["question_lc"] = df["question"].str.lower().fillna("")

    is_btc = (
        df["question_lc"].str.contains(r"bitcoin|\bbtc\b", regex=True)
        & df["question_lc"].str.contains(
            r"hit|reach|above|below|under|exceed", regex=True
        )
        & (df["volume"] > _BTC_MARKET_MIN_VOLUME)
    )
    btc = df[is_btc].copy()

    def _orientation(q: str) -> int:
        if any(kw in q for kw in _BULLISH_KEYWORDS):
            return 1
        if any(kw in q for kw in _BEARISH_KEYWORDS):
            return -1
        return 0

    btc["orientation"] = btc["question_lc"].apply(_orientation)
    btc = btc[btc["orientation"] != 0]

    logging.info(f"Polymarket: identified {len(btc)} high-liquidity BTC markets")
    return btc[["market_id", "orientation"]].copy()


def _compute_activity_zscore(
    trades_df: pd.DataFrame,
    btc_market_ids: list,
    price_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Compute daily trade-count z-score using an expanding window.

    Expanding window ensures normalization at day t uses only data through t.
    Missing dates (no Polymarket data) receive z-score of 0.0 (neutral).
    """
    btc_trades = trades_df[trades_df["market_id"].isin(btc_market_ids)].copy()

    if btc_trades.empty:
        logging.warning("No BTC trades found; activity z-score will be 0.0 everywhere.")
        return pd.Series(0.0, index=price_index, name="polymarket_activity_zscore")

    # Normalize to date
    btc_trades["date"] = pd.to_datetime(btc_trades["timestamp"]).dt.normalize()

    daily_count = (
        btc_trades.groupby("date")
        .size()
        .rename("daily_trade_count")
        .astype(float)
    )

    # Expanding window z-score — strictly causal.
    # min_periods=7: require at least one week of data before standardizing.
    exp_mean = daily_count.expanding(min_periods=7).mean()
    exp_std = daily_count.expanding(min_periods=7).std().fillna(1.0).clip(lower=1e-8)
    zscore = (daily_count - exp_mean) / exp_std

    # Align to the full price index; dates without Polymarket data → 0.0
    zscore = zscore.reindex(price_index, fill_value=0.0)
    zscore.name = "polymarket_activity_zscore"

    n_active = int((zscore != 0.0).sum())
    logging.info(
        f"Polymarket activity: {n_active} active days "
        f"({n_active / len(price_index):.1%} of price history)"
    )
    return zscore


def load_polymarket_btc_activity(
    poly_data: dict,
    price_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build the Polymarket activity feature DataFrame for use in precompute_features.

    Returns a DataFrame indexed by the same dates as the BTC price series with:
      polymarket_activity_zscore : float
          Expanding-window z-score of daily trade count in high-liquidity BTC markets.
          0.0 on dates before Polymarket coverage begins (no signal, neutral).
          EDA-validated: z > 1.5 predicts +5.4% at 30 days (p=0.029).

    Args:
        poly_data:    Dictionary returned by load_polymarket_data().
        price_index:  DatetimeIndex from the BTC price series.

    Returns:
        DataFrame with a single column 'polymarket_activity_zscore'.
    """
    neutral = pd.DataFrame(
        {"polymarket_activity_zscore": 0.0}, index=price_index
    )

    if "markets" not in poly_data or "trades" not in poly_data:
        logging.warning(
            "Polymarket 'markets' or 'trades' not available. "
            "Activity z-score will be 0.0 for all dates."
        )
        return neutral

    try:
        btc_markets = _identify_btc_markets(poly_data["markets"])
        if btc_markets.empty:
            logging.warning("No qualifying BTC markets found. Activity z-score = 0.0.")
            return neutral

        btc_ids = btc_markets["market_id"].tolist()
        zscore = _compute_activity_zscore(poly_data["trades"], btc_ids, price_index)
        return zscore.to_frame()

    except Exception as exc:
        logging.warning(
            f"Failed to compute Polymarket activity features ({exc}). "
            "Defaulting to 0.0."
        )
        return neutral
