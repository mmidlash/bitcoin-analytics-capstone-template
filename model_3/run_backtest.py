"""
Model 3 Backtest Runner.

Run the full strategy evaluation pipeline:
  1. Load BTC data
  2. Generate walk-forward GBM probs (if gbm_walkforward_probs.parquet missing)
  3. Precompute features — walk-forward probs embedded in gbm_buy_prob
  4. Run rolling-window SPD backtest (2018-01-01 to 2025-12-31)
  5. Validate: weights sum to 1.0, win rate ≥ 50%
  6. Save charts and metrics to model_3/output/

Walk-forward design: every historical gbm_buy_prob value is produced by a
GBM trained only on data before that date, making the full backtest strictly
out-of-sample for the ML component.

Usage:
    python -m model_3.run_backtest

Outputs (in model_3/output/):
    performance_comparison.svg
    excess_percentile_distribution.svg
    win_loss_comparison.svg
    cumulative_performance.svg
    metrics_summary.svg
    metrics.json
"""

import logging
from pathlib import Path

import pandas as pd

from template.prelude_template import load_data
from template.backtest_template import run_full_analysis
from model_3 import model_development_3
from model_3.model_development_3 import (
    precompute_features,
    compute_window_weights,
    GBM_WALKFORWARD_PATH,
)

_FEATURES_DF: pd.DataFrame | None = None


def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    """
    Adapter connecting the backtest engine to Model 3's weight computation.

    For backtesting, current_date = end_date so all dates are treated as
    'past' and receive signal-based weights.
    """
    global _FEATURES_DF

    if _FEATURES_DF is None:
        raise ValueError(
            "Features not precomputed. This should not happen in normal operation."
        )
    if df_window.empty:
        return pd.Series(dtype=float)

    start_date = df_window.index.min()
    end_date = df_window.index.max()
    current_date = end_date

    return compute_window_weights(_FEATURES_DF, start_date, end_date, current_date)


def main() -> None:
    global _FEATURES_DF

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=" * 60)
    logging.info("Model 3: Cycle-Aware GBM + Exchange Flow Strategy")
    logging.info("=" * 60)

    # ── Step 1: Load data (once — shared by all setup steps) ──────────────────
    logging.info("Loading BTC data...")
    btc_df = load_data()

    # ── Step 2: Generate walk-forward probs if not already done ───────────────
    # Walk-forward training produces gbm_walkforward_probs.parquet where every
    # historical prediction is strictly out-of-sample for the GBM that made it.
    if not GBM_WALKFORWARD_PATH.exists():
        logging.info(
            "gbm_walkforward_probs.parquet not found. "
            "Running walk-forward training (this takes a few minutes)..."
        )
        from model_3.training_pipeline import walk_forward_train  # noqa: PLC0415
        walk_forward_train(btc_df=btc_df)
        logging.info("Walk-forward training complete. Reloading probs...")
        model_development_3._load_walkforward_probs()
    else:
        logging.info(f"Walk-forward probs found at {GBM_WALKFORWARD_PATH}")

    # ── Step 3: Precompute features ───────────────────────────────────────────
    # Walk-forward probs are now loaded; precompute_features will embed them
    # in gbm_buy_prob for all covered historical dates.
    logging.info(
        "Precomputing features (MVRV + halving cycle + exchange flow + GBM)..."
    )
    _FEATURES_DF = precompute_features(btc_df)
    logging.info(
        f"Features shape: {_FEATURES_DF.shape}, "
        f"columns: {list(_FEATURES_DF.columns)}"
    )

    gbm_active = (_FEATURES_DF["gbm_buy_prob"] != 0.5).any()
    logging.info(
        f"GBM signal active: {gbm_active} "
        f"(gbm_buy_prob range: [{_FEATURES_DF['gbm_buy_prob'].min():.3f}, "
        f"{_FEATURES_DF['gbm_buy_prob'].max():.3f}])"
    )

    ex_flow_active = (_FEATURES_DF["exchange_flow_signal"] != 0.0).sum()
    logging.info(
        f"Exchange flow signal: {ex_flow_active} active days "
        f"(range: [{_FEATURES_DF['exchange_flow_signal'].min():.3f}, "
        f"{_FEATURES_DF['exchange_flow_signal'].max():.3f}])"
    )

    halving_range = (
        f"[{_FEATURES_DF['halving_cycle_sin'].min():.3f}, "
        f"{_FEATURES_DF['halving_cycle_sin'].max():.3f}]"
    )
    logging.info(f"Halving cycle sin range: {halving_range}")

    poly_active = (_FEATURES_DF["polymarket_activity_zscore"] != 0.0).sum()
    logging.info(f"Polymarket activity: {poly_active} active days")

    # ── Step 4: Run backtest ──────────────────────────────────────────────────
    output_dir = Path(__file__).parent / "output"

    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="Model 3 (Cycle-Aware GBM + Exchange Flow)",
    )


if __name__ == "__main__":
    main()
