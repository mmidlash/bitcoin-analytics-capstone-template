"""
Model 2 Backtest Runner.

Run the full strategy evaluation pipeline:
  1. Train the RF model (if rf_model.pkl does not yet exist)
  2. Precompute features (RF loaded automatically if pkl exists)
  3. Run rolling-window SPD backtest (2018-01-01 to 2025-12-31)
  4. Validate: no lookahead, weights sum to 1.0, win rate ≥ 50%
  5. Save charts and metrics to model_2/output/

Usage:
    python -m model_2.run_backtest

Outputs (in model_2/output/):
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
from model_2.model_development_2 import (
    precompute_features,
    compute_window_weights,
    RF_MODEL_PATH,
)

_FEATURES_DF: pd.DataFrame | None = None


def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    """
    Adapter connecting the backtest engine to Model 2's weight computation.

    The engine passes a features slice (or the full df) indexed by date.
    We extract the date range and delegate to compute_window_weights(),
    which uses the globally precomputed _FEATURES_DF.

    For backtesting, current_date = end_date so all dates are treated as
    "past" and receive signal-based weights (none are treated as future).
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
    current_date = end_date  # All historical dates are "past" in a backtest

    return compute_window_weights(_FEATURES_DF, start_date, end_date, current_date)


def main() -> None:
    global _FEATURES_DF

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=" * 60)
    logging.info("Model 2: Regime-Aware RF + Change Detection Strategy")
    logging.info("=" * 60)

    # ── Step 1: Train RF if not already trained ───────────────────────────────
    if not RF_MODEL_PATH.exists():
        logging.info(
            "rf_model.pkl not found. Running training pipeline first..."
        )
        # Import here (not at top) to keep training pipeline isolated from inference
        from model_2.training_pipeline import train_and_save  # noqa: PLC0415
        btc_df_for_training = load_data()
        train_and_save(btc_df=btc_df_for_training)
        logging.info("Training complete. Reloading model for backtest...")

        # Reload the RF singleton now that the pkl exists
        from model_2 import model_development_2  # noqa: PLC0415
        model_development_2._load_rf_model()
    else:
        logging.info(f"RF model found at {RF_MODEL_PATH}")

    # ── Step 2: Load data ─────────────────────────────────────────────────────
    logging.info("Loading BTC data...")
    btc_df = load_data()

    # ── Step 3: Precompute features ───────────────────────────────────────────
    logging.info("Precomputing features (MVRV + regime instability + Polymarket + RF)...")
    _FEATURES_DF = precompute_features(btc_df)
    logging.info(f"Features shape: {_FEATURES_DF.shape}, columns: {list(_FEATURES_DF.columns)}")

    rf_present = (_FEATURES_DF["rf_buy_prob"] != 0.5).any()
    logging.info(
        f"RF signal active: {rf_present} "
        f"(rf_buy_prob range: [{_FEATURES_DF['rf_buy_prob'].min():.3f}, "
        f"{_FEATURES_DF['rf_buy_prob'].max():.3f}])"
    )

    poly_active = (_FEATURES_DF["polymarket_activity_zscore"] != 0.0).sum()
    logging.info(
        f"Polymarket activity: {poly_active} active days "
        f"({poly_active / len(_FEATURES_DF):.1%} of history)"
    )

    # ── Step 4: Run backtest ──────────────────────────────────────────────────
    output_dir = Path(__file__).parent / "output"

    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="Model 2 (Regime-Aware RF)",
    )


if __name__ == "__main__":
    main()
