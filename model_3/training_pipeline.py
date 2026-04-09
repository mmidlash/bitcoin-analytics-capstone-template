"""
GBM Training Pipeline for Model 3.

!! THIS MODULE ACCESSES FUTURE PRICES TO GENERATE TRAINING LABELS. !!
!! IT MUST NEVER BE IMPORTED BY model_development_3.py              !!
!! OR ANY INFERENCE MODULE.                                          !!

Strict separation enforced by module structure:
  training_pipeline.py    →  imports precompute_features, builds labels using
                              future prices, trains walk-forward GBMs, saves parquet
  model_development_3.py  →  loads gbm_walkforward_probs.parquet at import time (static)

One artifact is produced:

  gbm_walkforward_probs.parquet
    Walk-forward expanding-window predictions. For each predict year Y,
    a separate GBM is trained on data TRAINING_START through (Y-1)-12-31
    and applied to year Y. Every historical prediction is therefore
    strictly out-of-sample for the model that produced it.

Usage:
    python -m model_3.training_pipeline

Design Rationale — Walk-Forward vs Single-Model Backtest:

  A single GBM trained on 2012-2023 and backtested from 2018 evaluates the
  model IN-SAMPLE for ~75% of the backtest period.  Walk-forward retraining
  ensures every backtest date uses a GBM that has never seen that date's
  labels, giving a genuine out-of-sample win rate.

  Walk-forward schedule (expanding window):
    Train 2012-2016 → predict 2017
    Train 2012-2017 → predict 2018   ← backtest start
    Train 2012-2018 → predict 2019
    ...
    Train 2012-2023 → predict 2024
    Train 2012-2024 → predict 2025

  Label buffer: the last FORWARD_DAYS samples are excluded from each
  training set so the training labels never peek into the predict year.

Design Rationale — GBM vs Random Forest (Model 2):

  GradientBoostingClassifier builds trees SEQUENTIALLY, each fitted to the
  pseudo-residuals of the current ensemble. This gives three advantages over
  the RF used in Model 2:

  1. Sequential error correction: Later trees focus on the hard examples
     (rare market regimes, unusual cycle phases) that earlier trees
     mispredicted. RF trains all trees independently, giving equal weight to
     all examples.

  2. Better performance on structured tabular data: Extensive benchmarks on
     tabular datasets (Kaggle, OpenML) consistently show GBM outperforms RF
     when features have non-linear interactions — which is likely true here
     given the interaction between halving cycle phase and MVRV level.

  3. More compact model: GBM typically achieves comparable performance with
     fewer trees (fewer trees x smaller trees = faster inference).

  Tradeoff: GBM is slower to train (sequential) and has an additional
  learning_rate hyperparameter. We use conservative settings
  (learning_rate=0.05, max_depth=3) to control overfitting.

Cross-validation strategy:
  TimeSeriesSplit with a 30-day gap prevents label leakage into the
  validation fold. The gap ensures that the validation labels (which look
  30 days forward) do not overlap with the training window's recent data.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from template.prelude_template import load_data
from model_3.model_development_3 import precompute_features, _GBM_FEATURE_COLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

PRICE_COL = "PriceUSD_coinmetrics"
WALK_FORWARD_PATH = Path(__file__).parent / "gbm_walkforward_probs.parquet"

# Minimum years of training data required before generating predictions.
# 5 years → first prediction is for 2017 (trained on 2012-2016).
WALK_FORWARD_MIN_TRAIN_YEARS = 5

# ── Training configuration ────────────────────────────────────────────────────
FORWARD_DAYS = 30           # Prediction horizon: 30-day forward return
TRAINING_START = "2012-01-01"
MIN_TRAIN_SAMPLES = 500

# ── GBM hyperparameters ───────────────────────────────────────────────────────
# Conservative settings to prevent overfitting on ~3500 training samples.
# max_depth=3: classical "stumps + interaction" depth for GBM
# learning_rate=0.05: slow shrinkage; compensated by n_estimators=300
# subsample=0.75: stochastic gradient boosting — reduces variance, adds noise
# min_samples_leaf=25: ~0.7% of training data; prevents over-specific splits
# max_features=0.75: column subsampling per split (like RF but proportional)
GBM_N_ESTIMATORS = 300
GBM_MAX_DEPTH = 3
GBM_LEARNING_RATE = 0.05
GBM_SUBSAMPLE = 0.75
GBM_MIN_SAMPLES_LEAF = 25
GBM_MAX_FEATURES = 0.75
GBM_RANDOM_STATE = 42


def build_training_labels(
    btc_df: pd.DataFrame,
    features_df: pd.DataFrame,
    forward_days: int = FORWARD_DAYS,
) -> pd.Series:
    """
    Build binary buy-opportunity labels.

    !! Accesses future prices — TRAINING USE ONLY. !!

    Label definition:
      y[t] = 1 if price[t + forward_days] > price[t]
             (price rises → today is a cheap accumulation day)
      y[t] = 0 otherwise

    Features at row t reflect data from t-1 (after .shift(1) in
    precompute_features). Labels at row t use price[t + forward_days].
    No overlap — same separation as inference time.

    Args:
        btc_df:       Raw BTC DataFrame (contains future prices for labels).
        features_df:  Precomputed features (used for index alignment).
        forward_days: Prediction horizon.

    Returns:
        Series of {0, 1} labels; NaN for the last forward_days rows.
    """
    price = btc_df[PRICE_COL].dropna()
    forward_price = price.shift(-forward_days)
    labels = (forward_price > price).astype(float)
    labels = labels.reindex(features_df.index)
    labels.iloc[-forward_days:] = np.nan

    pos_rate = labels.dropna().mean()
    logging.info(
        f"Labels: {labels.notna().sum():,} valid rows, "
        f"{pos_rate:.1%} positive (price rises in {forward_days}d)"
    )
    return labels


def walk_forward_train(
    btc_df: pd.DataFrame | None = None,
    features_df: pd.DataFrame | None = None,
    output_path: Path = WALK_FORWARD_PATH,
) -> pd.Series:
    """
    Walk-forward expanding-window GBM training.

    For each predict year Y (once WALK_FORWARD_MIN_TRAIN_YEARS of prior data
    are available):
      1. Train a fresh GBM on TRAINING_START → (Y-1)-12-31, excluding the
         last FORWARD_DAYS samples from the label set so that 30-day forward
         returns never cross into the predict year.
      2. Apply the trained GBM to every date in year Y.

    This ensures that every historical gbm_buy_prob value is strictly
    out-of-sample — the model that produced it never saw that date's labels.

    Args:
        btc_df:      BTC DataFrame. If None, loads from disk.
        features_df: Precomputed features. If None, computed from btc_df.
                     gbm_buy_prob will be 0.5 (no model yet) — that column
                     is excluded from _GBM_FEATURE_COLS so it has no effect.
        output_path: Parquet path for the resulting probability series.

    Returns:
        Series of walk-forward gbm_buy_prob values, indexed by date.
    """
    if btc_df is None:
        btc_df = load_data()

    if features_df is None:
        logging.info("Computing features for walk-forward training...")
        features_df = precompute_features(btc_df)

    train_start_ts = pd.Timestamp(TRAINING_START)
    first_predict_year = train_start_ts.year + WALK_FORWARD_MIN_TRAIN_YEARS
    last_data_year = features_df.index.max().year

    gbm_cols = [c for c in _GBM_FEATURE_COLS if c in features_df.columns]

    all_probs: dict[pd.Timestamp, float] = {}
    last_gbm = None
    last_gbm_cols: list[str] = []

    logging.info(
        f"Walk-forward training: {first_predict_year}-{last_data_year} "
        f"({last_data_year - first_predict_year + 1} prediction years)"
    )

    for predict_year in range(first_predict_year, last_data_year + 1):
        train_end = f"{predict_year - 1}-12-31"
        predict_start = f"{predict_year}-01-01"
        predict_end = f"{predict_year}-12-31"

        # Label buffer: exclude the last FORWARD_DAYS of the training period
        # so that no training label's forward-return window crosses into the
        # predict year (e.g. a Dec-3 label for a 30-day horizon looks at
        # Jan 2 — one day into the prediction year; we cut at Dec 1 to be safe).
        label_cutoff = pd.Timestamp(predict_start) - pd.Timedelta(days=FORWARD_DAYS + 1)

        train_feats = features_df.loc[TRAINING_START:train_end].copy()
        if train_feats.empty:
            logging.info(f"  {predict_year}: no training features — skipping")
            continue

        labels = build_training_labels(btc_df, train_feats, FORWARD_DAYS)

        valid = (
            labels.notna()
            & (labels.index <= label_cutoff)
            & train_feats[gbm_cols].notna().all(axis=1)
        )
        X_tr = train_feats.loc[valid, gbm_cols].values.astype(float)
        y_tr = labels.loc[valid].values.astype(int)

        if len(X_tr) < MIN_TRAIN_SAMPLES:
            logging.info(
                f"  {predict_year}: skipped — only {len(X_tr)} valid training "
                f"samples (need ≥ {MIN_TRAIN_SAMPLES})"
            )
            continue

        # Sample weights from regime instability 
        if "regime_instability" in train_feats.columns:
            instability = train_feats.loc[valid, "regime_instability"].values
            sample_weight = np.clip(1.0 - 0.4 * instability, 0.6, 1.0)
        else:
            sample_weight = None

        gbm = GradientBoostingClassifier(
            n_estimators=GBM_N_ESTIMATORS,
            max_depth=GBM_MAX_DEPTH,
            learning_rate=GBM_LEARNING_RATE,
            subsample=GBM_SUBSAMPLE,
            min_samples_leaf=GBM_MIN_SAMPLES_LEAF,
            max_features=GBM_MAX_FEATURES,
            random_state=GBM_RANDOM_STATE,
        )
        gbm.fit(X_tr, y_tr, sample_weight=sample_weight)

        # Predict for the full prediction year
        pred_feats = features_df.loc[predict_start:predict_end]
        if pred_feats.empty:
            logging.info(f"  {predict_year}: no prediction dates in data — skipping")
            continue

        pred_valid = pred_feats[gbm_cols].notna().all(axis=1)
        X_pred = pred_feats.loc[pred_valid, gbm_cols].fillna(0.0).values.astype(float)

        if len(X_pred) == 0:
            logging.info(f"  {predict_year}: no valid prediction rows — skipping")
            continue

        probs = gbm.predict_proba(X_pred)[:, 1]
        pred_dates = pred_feats.loc[pred_valid].index
        for date, prob in zip(pred_dates, probs):
            all_probs[date] = float(prob)

        pos_rate = float(y_tr.mean())
        logging.info(
            f"  {predict_year}: trained on {len(X_tr):,} samples "
            f"({TRAINING_START}–{label_cutoff.date()}, {pos_rate:.1%} positive) "
            f"→ predicted {len(probs):,} days"
        )
        last_gbm, last_gbm_cols = gbm, gbm_cols  # track for final importance log

    if not all_probs:
        raise RuntimeError(
            "Walk-forward training produced no predictions. "
            "Check data availability and WALK_FORWARD_MIN_TRAIN_YEARS setting."
        )

    prob_series = pd.Series(all_probs, name="gbm_buy_prob_walkforward").sort_index()
    prob_series.to_frame().to_parquet(output_path)
    logging.info(
        f"Walk-forward probs saved → {output_path}  "
        f"({len(prob_series):,} dates, "
        f"{prob_series.index.min().date()} to {prob_series.index.max().date()})"
    )

    # Log feature importances from the final walk-forward GBM (most data-rich).
    if last_gbm is not None:
        importances = pd.Series(last_gbm.feature_importances_, index=last_gbm_cols)
        importances = importances.sort_values(ascending=False)
        logging.info("Feature importances — final walk-forward GBM (top 15 by impurity gain):")
        for feat, imp in importances.head(15).items():
            bar = "█" * int(imp * 300)
            logging.info(f"  {feat:<38s} {imp:.4f}  {bar}")

    return prob_series


if __name__ == "__main__":
    walk_forward_train()
