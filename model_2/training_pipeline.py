"""
RF Training Pipeline for Model 2.

!! THIS MODULE ACCESSES FUTURE PRICES TO GENERATE TRAINING LABELS. !!
!! IT MUST NEVER BE IMPORTED BY model_development_2.py              !!
!! OR ANY INFERENCE MODULE.                                          !!

The strict separation is enforced by module structure:
  training_pipeline.py  →  imports precompute_features, builds labels using
                            future prices, trains RF, saves rf_model.pkl
  model_development_2.py →  loads rf_model.pkl at import time (static artifact)

The ONLY artifact crossing from training to inference is rf_model.pkl.
No future price data ever reaches the inference path.

Usage:
    python -m model_2.training_pipeline

Flow:
  1. Load BTC data
  2. Call precompute_features() — rf_buy_prob will be 0.5 everywhere because
     no model exists yet. This is intentional and correct.
  3. Build forward-return labels: y[t] = 1 if price[t+30] > price[t]
     (price rises → today is a relatively cheap accumulation day)
  4. Align lagged features with labels (X uses features[t-1], y uses price[t+30])
  5. Temporal CV with gap to prevent label leakage into validation
  6. Train final RF on full training set, evaluate on OOS holdout (2024+)
  7. Save rf_model.pkl

RF design rationale:
  max_depth=4, min_samples_leaf=40: shallow trees prevent overfitting
    given ~216 Polymarket days (only ~5% of training data has non-zero
    Polymarket features). The RF will primarily learn MVRV/momentum signal.
  class_weight='balanced': BTC trends up so ~55% of 30d windows see
    positive returns. Balanced weighting prevents the majority class dominating.
  sample_weight from regime_instability: down-weights ambiguous samples
    near detected regime transitions where forward returns are noisy.
"""

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

# ─── Imports from this project ───────────────────────────────────────────────
# model_development_2 import triggers _load_rf_model() at module load.
# Since rf_model.pkl does not exist yet, _RF_MODEL = None and rf_buy_prob = 0.5.
from template.prelude_template import load_data
from model_2.model_development_2 import precompute_features, _RF_FEATURE_COLS
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

PRICE_COL = "PriceUSD_coinmetrics"
RF_MODEL_PATH = Path(__file__).parent / "rf_model.pkl"

# ── Training configuration ────────────────────────────────────────────────────
FORWARD_DAYS = 30           # Prediction horizon: 30-day forward return
TRAINING_START = "2012-01-01"  # Start of training (need sufficient MVRV history)
TRAINING_END = "2023-12-31"    # Training cutoff — 2024+ held out for OOS eval
N_CV_SPLITS = 5
CV_GAP = FORWARD_DAYS       # Gap between train and val folds (prevents label leakage)
MIN_TRAIN_SAMPLES = 500

# ── RF hyperparameters ────────────────────────────────────────────────────────
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 4            # Shallow: limits overfitting on 14 features
RF_MIN_SAMPLES_LEAF = 40    # Large: accommodates small Polymarket sample
RF_MAX_FEATURES = "sqrt"
RF_RANDOM_STATE = 42


def build_training_labels(
    btc_df: pd.DataFrame,
    features_df: pd.DataFrame,
    forward_days: int = FORWARD_DAYS,
) -> pd.Series:
    """
    Build binary buy-opportunity labels.

    !! Accesses future prices — training use ONLY. !!

    Label definition:
      y[t] = 1 if price[t + forward_days] > price[t]
             (price rises → today's price is cheap relative to the future
              → today gives more sats per dollar than forward_days from now)
      y[t] = 0 otherwise

    Features at row t are from t-1 (after shift(1) in precompute_features).
    Labels at row t use price[t+forward_days]. No overlap — this is the
    same separation enforced at inference time.

    Args:
        btc_df:       Raw BTC DataFrame (contains future prices for label build).
        features_df:  Precomputed features (used for index alignment only).
        forward_days: Prediction horizon in days.

    Returns:
        Series of int labels {0, 1}, NaN for the last forward_days rows.
    """
    price = btc_df[PRICE_COL].dropna()
    forward_price = price.shift(-forward_days)
    forward_return = (forward_price - price) / price

    # y = 1: price rises from t to t+forward_days (today is cheap)
    labels = (forward_return > 0.0).astype(float)

    # Align to the features index (may start later than btc_df due to warm-up)
    labels = labels.reindex(features_df.index)

    # NaN out the last forward_days rows — no future price available
    labels.iloc[-forward_days:] = np.nan

    pos_rate = labels.dropna().mean()
    logging.info(
        f"Labels: {labels.notna().sum():,} valid rows, "
        f"{pos_rate:.1%} positive (price rises in {forward_days}d), "
        f"{forward_days} tail rows excluded (no future data)"
    )
    return labels


def train_and_save(
    btc_df: pd.DataFrame | None = None,
    output_path: Path = RF_MODEL_PATH,
) -> None:
    """
    Full training pipeline: features → labels → RF → rf_model.pkl.

    Args:
        btc_df:      BTC DataFrame. If None, loads from disk.
        output_path: Path for the saved RF model.
    """
    if btc_df is None:
        btc_df = load_data()

    # ── Step 1: Compute features ─────────────────────────────────────────────
    # rf_buy_prob will be 0.5 everywhere (no model yet). This is correct:
    # the RF is trained on _RF_FEATURE_COLS which does NOT include rf_buy_prob.
    logging.info("Computing features (rf_buy_prob = 0.5 — no model yet, this is correct)...")
    features_df = precompute_features(btc_df)

    # ── Step 2: Build labels ─────────────────────────────────────────────────
    logging.info(f"Building {FORWARD_DAYS}-day forward-return labels...")
    labels = build_training_labels(btc_df, features_df, FORWARD_DAYS)

    # ── Step 3: Restrict to training period and drop NaNs ────────────────────
    train_feats = features_df.loc[TRAINING_START:TRAINING_END].copy()
    train_labels = labels.loc[TRAINING_START:TRAINING_END].copy()

    # Drop rows with NaN in labels OR in any RF feature
    rf_cols = [c for c in _RF_FEATURE_COLS if c in train_feats.columns]
    valid = train_labels.notna() & train_feats[rf_cols].notna().all(axis=1)

    X = train_feats.loc[valid, rf_cols].values.astype(float)
    y = train_labels.loc[valid].values.astype(int)

    logging.info(
        f"Training set: {len(X):,} samples from {TRAINING_START} to {TRAINING_END}"
    )
    logging.info(
        f"  Positive rate: {y.mean():.1%} | Features: {len(rf_cols)}"
    )
    logging.info(
        f"  Missing RF features (filled as 0.0): "
        f"{[c for c in _RF_FEATURE_COLS if c not in train_feats.columns]}"
    )

    if len(X) < MIN_TRAIN_SAMPLES:
        raise ValueError(
            f"Insufficient training data: {len(X)} samples < {MIN_TRAIN_SAMPLES} required. "
            f"Check TRAINING_START ('{TRAINING_START}') and data availability."
        )

    # ── Step 4: Sample weights from regime instability ───────────────────────
    # Days near detected regime transitions receive lower weight (more ambiguous).
    if "regime_instability" in train_feats.columns:
        instability = train_feats.loc[valid, "regime_instability"].values
        # Weight = 1.0 when instability = 0 (stable), 0.6 when instability = 1.0
        sample_weight = np.clip(1.0 - 0.4 * instability, 0.6, 1.0)
        logging.info(
            f"Sample weights: mean={sample_weight.mean():.3f}, "
            f"min={sample_weight.min():.3f}, max={sample_weight.max():.3f}"
        )
    else:
        sample_weight = None
        logging.info("regime_instability not available; using uniform sample weights.")

    # ── Step 5: Temporal cross-validation ────────────────────────────────────
    # TimeSeriesSplit with gap=FORWARD_DAYS prevents labels from leaking into
    # the validation fold (the gap ensures val-set labels don't use train-period
    # future prices).
    logging.info(
        f"Temporal CV: {N_CV_SPLITS} folds, gap={CV_GAP} days, "
        f"min_train={MIN_TRAIN_SAMPLES} samples..."
    )
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS, gap=CV_GAP)
    cv_aucs: list[float] = []

    for fold_num, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        if len(train_idx) < MIN_TRAIN_SAMPLES:
            logging.info(f"  Fold {fold_num}: skipped (only {len(train_idx)} train samples)")
            continue

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        sw_tr = sample_weight[train_idx] if sample_weight is not None else None

        if len(np.unique(y_val)) < 2:
            logging.info(f"  Fold {fold_num}: skipped (single class in validation)")
            continue

        rf_fold = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            max_features=RF_MAX_FEATURES,
            class_weight="balanced",
            random_state=RF_RANDOM_STATE,
            n_jobs=-1,
        )
        rf_fold.fit(X_tr, y_tr, sample_weight=sw_tr)

        val_proba = rf_fold.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_proba)
        cv_aucs.append(auc)
        logging.info(
            f"  Fold {fold_num}: train={len(train_idx):,}, val={len(val_idx):,}, "
            f"AUC={auc:.4f}"
        )

    if cv_aucs:
        mean_auc = float(np.mean(cv_aucs))
        std_auc = float(np.std(cv_aucs))
        logging.info(
            f"CV summary: mean AUC={mean_auc:.4f} ± {std_auc:.4f} "
            f"over {len(cv_aucs)} folds"
        )
        if mean_auc < 0.51:
            warnings.warn(
                f"Mean CV AUC ({mean_auc:.4f}) is near random (0.5). "
                "The RF may add limited signal. The model will still run; "
                "rf_buy_prob will be close to 0.5 and contribute minimally "
                "to the multiplier — this degrades gracefully to the "
                "MVRV + MA + Polymarket baseline.",
                UserWarning,
                stacklevel=2,
            )
    else:
        logging.warning("No valid CV folds completed. Check training data size.")

    # ── Step 6: Final model trained on full training data ────────────────────
    logging.info("Training final RF on full training set...")
    rf_final = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=RF_MAX_FEATURES,
        class_weight="balanced",
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )
    rf_final.fit(X, y, sample_weight=sample_weight)

    # Feature importance summary
    importances = pd.Series(rf_final.feature_importances_, index=rf_cols)
    importances = importances.sort_values(ascending=False)
    logging.info("Feature importances (all features):")
    for feat, imp in importances.items():
        bar = "█" * int(imp * 200)
        logging.info(f"  {feat:<35s} {imp:.4f}  {bar}")

    # ── Step 7: OOS evaluation on 2024+ ─────────────────────────────────────
    oos_start = "2024-01-01"
    oos_feats = features_df.loc[oos_start:].copy()
    oos_labels = labels.reindex(oos_feats.index)
    oos_valid = oos_labels.notna() & oos_feats[rf_cols].notna().all(axis=1)

    n_oos = oos_valid.sum()
    if n_oos >= 30 and len(np.unique(oos_labels.loc[oos_valid].values)) >= 2:
        X_oos = oos_feats.loc[oos_valid, rf_cols].values.astype(float)
        y_oos = oos_labels.loc[oos_valid].values.astype(int)
        oos_proba = rf_final.predict_proba(X_oos)[:, 1]
        oos_auc = roc_auc_score(y_oos, oos_proba)
        logging.info(
            f"OOS evaluation (2024+): {n_oos:,} samples, "
            f"AUC={oos_auc:.4f}"
        )
    else:
        logging.info(
            f"OOS evaluation skipped: {n_oos} valid samples "
            f"(need ≥ 30 with both label classes)"
        )

    # ── Step 8: Save ─────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_final, output_path)
    logging.info(f"RF model saved → {output_path}")
    logging.info(
        "Run `python -m model_2.run_backtest` to evaluate the full strategy."
    )


if __name__ == "__main__":
    train_and_save()
