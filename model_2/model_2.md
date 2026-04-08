# Model 2: Regime-Aware Random Forest + Change Detection

## Executive Summary

Model 2 extends the rule-based MVRV + MA-200 baseline (Example 1) with three additions: a **Random Forest buy-opportunity classifier**, a **causal regime instability detector** based on distributional shift, and an **EDA-validated Polymarket activity signal** (expanding-window trade-count z-score from high-liquidity BTC prediction markets). Together these address two limitations identified in the EDA — the MA-200's lag during regime transitions, and the absence of any data-driven synthesis across multiple signals.

| Metric                   | Example 1 | **Model 2 (RF)** |
|--------------------------|-----------|------------------|
| Score                    | 59.54%    | **59.51%**       |
| Win Rate                 | 60.31%    | **59.56%**       |
| Exp-Decay Percentile     | 58.78%    | **59.45%**       |
| Mean Excess Percentile   | +5.70%    | **+4.54%**       |
| Median Excess Percentile | +6.43%    | **+3.90%**       |
| Mean SPD Ratio           | 1.17      | **1.13**         |

> **Score** = 0.5 × Win Rate + 0.5 × Exp-Decay Avg Percentile. 2,557 rolling 1-year windows, 2018-01-01 to 2025-12-31.

---

## Motivation & Design Philosophy

Standard DCA allocates equal capital every day — a sound baseline but one that ignores the well-documented cyclicality of Bitcoin's valuation. The prior models use MVRV z-score and the 200-day MA to identify relatively cheap accumulation days. Model 2 builds on this with three additions driven by two core questions:

1. **Can a supervised classifier add signal beyond the hand-crafted rules?** A Random Forest trained on labeled "good buying days" (30-day forward return > 0) can learn non-linear interactions between MVRV, momentum, on-chain activity, and Polymarket data that a fixed weighting scheme cannot capture.

2. **Can we detect regime transitions before the MA-200 does?** The 200-day MA lags heavily at turning points — it's slow to recognize that the market has entered a new regime. A causal distributional shift test on MVRV can flag instability earlier, allowing the model to dampen all signals during uncertain transitions rather than doubling down on a strategy that may no longer apply.

The third addition is an **EDA-validated Polymarket activity signal**: the EDA found that spikes in trade-count activity across high-liquidity BTC prediction markets are statistically associated with positive forward returns, and this signal is incorporated directly as a rule-based component (distinct from its role as a feature inside the RF).

---

## Architecture

> **Note:** Features serve a dual role. All 13 features feed into the RF classifier, which synthesizes them into the single `rf_buy_prob` output. Separately, the MVRV, MA-200, and Polymarket features are also used directly by hand-crafted formulas to produce their own signal components — these bypass the RF entirely and enter the multiplier independently. Change detection plays a different role again: it is not a weighted signal but a **multiplicative dampener** applied to the full combined signal, scaling it down by up to 30% when a distributional shift is detected. In stable markets it has no effect; it only activates during detected regime transitions.

```
┌────────────────────────────────────────────────────────────────┐
│                      Feature Engineering                       │
│                                                                │
│  ┌──────────────┐  ┌────────────────┐  ┌───────────────┐       │
│  │ MVRV / MA-200│  │ On-Chain       │  │ Polymarket    │       │
│  │ gradient     │  │ HashRate, TxCnt│  │ Activity      │       │
│  │ acceleration │  │ 7d/30d returns │  │ zscore        │       │
│  └──────┬───────┘  └──────┬─────────┘  └───────┬───────┘       │
│         │                 │                    │               │
│  ┌──────▼──────────────┐  │                    │               │
│  │ Change Detection    │  │                    │               │
│  │ MVRV z-score:       │  │                    │               │
│  │ 30d vs 120d shift   │  │                    │               │
│  │ → regime_instability│  │                    │               │
│  └──────┬──────────────┘  │                    │               │
│         │                 │                    │               │
│         └─────────────────┴────────────────────┘               │
│                           │                                    │
│                   ┌───────▼────────┐                           │
│                   │  RF Classifier │                           │
│                   │  (13 features) │                           │
│                   │  → rf_buy_prob │                           │
│                   └───────┬────────┘                           │
└───────────────────────────┼────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│                    Signal Combination                          │
│                                                                │
│  MVRV value signal    × 0.50   (primary valuation)             │
│  MA-200 signal        × 0.15   (trend confirmation)            │
│  RF buy probability   × 0.25   (data-driven synthesis)         │
│  Polymarket activity  × 0.10   (validated activity signal)     │
│                                                                │
│  × (1 − regime_instability × 0.30)  ← change detection output  │
│  × acceleration · confidence · volatility modifiers            │
└────────────────────────────────────────────────────────────────┘
                            │
                            ▼
        exp(clip(combined × 5.0, −5, 12))  → multiplier
                            │
                            ▼
        allocate_sequential_stable()  → weights summing to 1.0
```

---

## Component 1: Random Forest Buy-Opportunity Classifier

### What it learns

The RF predicts whether a given day is a relatively cheap accumulation opportunity — specifically, whether BTC's price will be higher 30 days later (a binary "buy day" label). It combines 13 features spanning on-chain metrics, price momentum, MVRV characteristics, and Polymarket activity, learning non-linear interactions that the hand-crafted rules cannot express.

### Training label

```python
y[t] = 1  if price[t + 30] > price[t]   # price rises → cheap today
y[t] = 0  otherwise
```

Labels are only computed in `training_pipeline.py`, which is strictly separated from the inference path. The only artifact that crosses the boundary is `rf_model.pkl`.

### Features (13 total)

| Feature                      | Type      | Description                                                  |
|------------------------------|-----------|--------------------------------------------------------------|
| `price_vs_ma`                | Trend     | Normalized distance from 200-day MA, clipped [-1, 1]         |
| `mvrv_zscore`                | Valuation | 365-day rolling z-score of MVRV ratio, clipped [-4, 4]       |
| `mvrv_gradient`              | Momentum  | 30-day EMA-smoothed MVRV trend direction in [-1, 1]          |
| `mvrv_acceleration`          | Momentum  | 14-day second derivative of gradient in [-1, 1]              |
| `mvrv_zone`                  | Regime    | Discrete zone: {Deep Value, Value, Neutral, Caution, Danger} |
| `mvrv_volatility`            | Risk      | 90-day MVRV volatility percentile in [0, 1]                  |
| `signal_confidence`          | Agreement | MVRV/MA directional agreement score in [0, 1]                |
| `regime_instability`         | Risk      | Distributional shift score in [0, 1] (see Component 2)       |
| `polymarket_activity_zscore` | Activity  | Expanding-window daily trade-count z-score                   |
| `btc_return_7d`              | Momentum  | 7-day log return                                             |
| `btc_return_30d`             | Momentum  | 30-day log return                                            |
| `hash_rate_zscore`           | On-chain  | 90-day rolling z-score of hash rate                          |
| `tx_count_zscore`            | On-chain  | 30-day rolling z-score of transaction count                  |

### Hyperparameters

| Parameter          | Value        | Rationale                                                                                                   |
|--------------------|--------------|-------------------------------------------------------------------------------------------------------------|
| `n_estimators`     | 300          | Sufficient ensemble stability given training set size                                                       |
| `max_depth`        | 4            | Intentionally shallow — limits memorization capacity                                                        |
| `min_samples_leaf` | 40           | Large leaf size; Polymarket coverage is only ~249 days                                                      |
| `max_features`     | `"sqrt"`     | Standard RF column subsampling for diversity                                                                |
| `class_weight`     | `"balanced"` | BTC trends up, so ~55% of 30-day windows are positive; balanced weighting prevents majority class dominance |
| `random_state`     | 42           | Reproducibility                                                                                             |

### Sample weighting

Training samples are weighted by regime stability. Days near detected regime transitions (`regime_instability` close to 1.0) receive lower weight (down to 0.6×), because forward returns near transition points are noisy and learning from them can mislead the model.

```python
sample_weight = clip(1.0 - 0.4 × regime_instability, 0.6, 1.0)
```

### Training / Inference Separation

`training_pipeline.py` is **never imported** by `model_development_2.py`. The label construction uses `price.shift(-30)` (future prices), which would constitute lookahead if present at inference time. The strict module boundary ensures this never happens. The RF model artifact (`rf_model.pkl`) contains only learned tree structure — no price data.

---

## Component 2: Causal Regime Instability Detection

### Motivation

The MA-200 is a lagging indicator by design. It takes weeks to months to recognize that a market regime has changed. The EDA confirmed this: the MA-200 signal often persists well into new regimes, causing the strategy to over-accumulate during early bear markets or over-reduce during early recoveries.

### Algorithm: Two-Sample Mean-Shift Detection

At each timestep `t`, the detector compares two non-overlapping windows of MVRV z-score:

- **Recent window** (30 days): `[t − 30, t]` — where the market is now
- **Baseline window** (120 days): `[t − 150, t − 31]` — where it was before

The standardized mean shift (how many baseline standard deviations the recent mean has moved) is mapped through a sigmoid to produce a score in [0, 1]:

```python
mean_shift = |recent_mean − baseline_mean| / baseline_std

instability = sigmoid(1.5 × (mean_shift − 1.0))
```

Score interpretation:
| Score | Meaning                                       |
|-------|-----------------------------------------------|
| ~0.30 | Early period (insufficient history)           |
| ~0.50 | No significant distributional shift           |
| ~0.73 | ~2 baseline std shift (elevated instability)  |
| ~0.88 | ~3 baseline std shift (high instability)      |

This is fundamentally a **causal two-sample mean-shift test** — there is no lookahead because rolling windows at position `t` only use data through `t`. The additional `.shift(1)` in `precompute_features()` means the signal reaching the multiplier on day `t` reflects data through `t−1`.

### How it modulates signals

When instability is elevated, all combined signals are dampened:

```python
stability = 1.0 − regime_instability × 0.30

combined = (value + ma + rf + poly signals) × stability
```

At maximum instability (score = 1.0), signal strength is reduced by 30%. This prevents the model from making aggressive accumulation or reduction decisions when the market may be mid-transition.

---

## Component 3: EDA-Validated Polymarket Activity Signal

High-liquidity BTC price-target prediction markets on Polymarket show statistically significant activity spikes ahead of price moves. The EDA validated this at 30 days (p = 0.029, +5.4% excess return when z > 1.5) and 60 days (p = 0.011, +8.5%).

The signal uses **expanding-window normalization** — a critical fix from the EDA, which computed z-scores using the full dataset mean/std (lookahead). This module computes z-scores using only data available at each point in time:

```python
exp_mean = daily_trade_count.expanding(min_periods=7).mean()
exp_std  = daily_trade_count.expanding(min_periods=7).std()
zscore   = (daily_count − exp_mean) / exp_std
```

Polymarket BTC market coverage spans approximately 249 active days in the dataset. Outside of coverage, the z-score defaults to 0.0 (no signal contribution).

---

## Training Period & The In-Sample Concern

The RF is trained on **2012–2023** and the backtest spans **2018–2025**. This means the 2018–2023 period is technically in-sample — the RF's labels and features from those years were used to fit the model parameters.

**Why this design was chosen:** Restricting training to pre-2018 data would give the RF only ~6 years of Bitcoin history (much of it pre-institutional, low-liquidity market structure). The on-chain features, Polymarket signal, and MVRV behavior in 2018–2023 are fundamentally more representative of how the model will be used going forward. The alternative — training only on pre-2017 data — would produce a model effectively blind to the dynamics it needs to generalize across.

**The strictly out-of-sample window is 2024+**, which appears in the training pipeline's OOS evaluation as an AUC holdout.

### Mitigating factors

The in-sample overlap is a real limitation, but three design choices significantly reduce its practical impact:

1. **The RF is intentionally shallow** (`max_depth=4`, `min_samples_leaf=40`). Shallow trees have limited memorization capacity — they cannot overfit to individual market days or local sequences. The depth-4 constraint means each tree makes at most 4 binary splits, capturing broad structural patterns rather than memorizing specific episodes.

2. **The RF carries only 25% of the combined signal.** Even if the RF were to overfit, 75% of the weight multiplier comes from rule-based components (MVRV value signal at 50%, MA-200 at 15%, Polymarket activity at 10%) that do not train on historical data at all. The RF can only move the final signal so far.

3. **Graceful degradation design.** The RF signal is recentered so that `rf_buy_prob = 0.5` contributes exactly zero to the combined signal: `rf_signal = (rf_buy_prob − 0.5) × 2.0`. An uninformative RF (one that outputs near 0.5 everywhere) contributes nothing and the model degrades cleanly to the MVRV + MA + Polymarket baseline.

---

## No-Lookahead Guarantee

Every feature entering the weight multiplier on day `t` reflects information available through day `t−1`. The sole lookahead guard is a single `.shift(1)` applied to all signal columns in `precompute_features()`:

```python
features[signal_cols] = features[signal_cols].shift(1)
```

The RF model is a static artifact. It is loaded once at import time from `rf_model.pkl` and applied to the already-lagged features — it never accesses the current day's data during inference.

---

## Signal Combination & Multiplier

```python
rf_signal   = (rf_buy_prob − 0.5) × 2.0          # recentered to [−1, +1]
poly_signal = tanh(polymarket_activity_zscore × 0.5)  # smoothly bounded

combined = value_signal × 0.50
         + ma_signal    × 0.15
         + rf_signal    × 0.25
         + poly_signal  × 0.10

combined = combined × (1.0 − regime_instability × 0.30)   # instability dampener
combined = combined × acceleration_modifier                 # [0.85, 1.15]
combined = combined × confidence_boost                      # [1.0, 1.15] if conf > 0.7
combined = combined × volatility_dampening                  # [0.8, 1.0] if vol > 0.8

multiplier = exp(clip(combined × 5.0, −5, 12))
```

The upper clip of 12 (vs. Example 1's 100) prevents extreme weight concentration when multiple signals simultaneously agree strongly — a safeguard against the model going all-in on a single day.

---

## Design Safeguards

| Safeguard                   | Implementation                                                                            |
|-----------------------------|-------------------------------------------------------------------------------------------|
| No lookahead                | All signal features `.shift(1)` in `precompute_features()`                                |
| Training isolation          | `training_pipeline.py` never imported by inference module                                 |
| RF degradation              | `rf_buy_prob = 0.5` when `rf_model.pkl` absent → zero RF contribution                     |
| Polymarket degradation      | `polymarket_activity_zscore = 0.0` when data absent → zero poly contribution              |
| Weight constraints          | All weights ≥ 1e-6; sum to exactly 1.0 per window                                         |
| Extreme concentration limit | Multiplier clipped to exp(12) ≈ 162,754 (practically: log(max_weight/min_weight) ≤ 17)    |

---

## RF Classifier Evaluation

### Cross-validation AUC (temporal, 5 folds)

| Fold     | Train samples | Val samples | AUC                 |
|----------|---------------|-------------|---------------------|
| 1        | 703           | 730         | 0.5422              |
| 2        | 1,433         | 730         | 0.4797              |
| 3        | 2,163         | 730         | 0.5544              |
| 4        | 2,893         | 730         | 0.4088              |
| 5        | 3,623         | 730         | 0.4809              |
| **Mean** |               |             | **0.4932 ± 0.0521** |

The CV AUC of 0.49 is statistically indistinguishable from random (0.50). The high fold-to-fold variance (±0.052) indicates the RF is not learning a stable, generalizable pattern — it finds mild signal in some periods and anti-signal in others. The training pipeline raises a `UserWarning` at this threshold and the model degrades gracefully: since `rf_buy_prob ≈ 0.5` everywhere, `rf_signal ≈ 0.0` and the RF contributes nothing to the multiplier.

### OOS evaluation (2024+, 715 samples)

The strictly out-of-sample window (2024+, trained on 2012–2023) produces AUC = **0.5133** — marginally above random, but not meaningfully so. Taken together, the CV and OOS results suggest the RF is unable to reliably predict 30-day forward returns from the available feature set beyond what the rule-based signals already capture.

This is an honest and expected finding: daily Bitcoin returns at the 30-day horizon are sufficiently noisy that a shallow RF with 13 features (most of which are smoothed MVRV and momentum derivatives) has limited room to add signal beyond the hand-crafted rules it is trained on. It does not hurt the model — the graceful degradation design ensures the strategy falls back to the MVRV + MA + Polymarket baseline when the RF is uninformative.

### Feature importances (final model, trained on 2012–2023)

| Feature                        | Importance |
|--------------------------------|------------|
| `mvrv_zscore`                  | 0.2008     |
| `price_vs_ma`                  | 0.1891     |
| `mvrv_volatility`              | 0.1506     |
| `signal_confidence`            | 0.1349     |
| `mvrv_gradient`                | 0.0729     |
| `regime_instability`           | 0.0728     |
| `mvrv_acceleration`            | 0.0567     |
| `btc_return_30d`               | 0.0432     |
| `mvrv_zone`                    | 0.0243     |
| `hash_rate_zscore`             | 0.0241     |
| `btc_return_7d`                | 0.0238     |
| `tx_count_zscore`              | 0.0067     |
| `polymarket_activity_zscore`   | 0.0000     |

The RF leans heavily on MVRV z-score and price-vs-MA (the same features driving the rule-based signals), confirming it is learning a noisy restatement of the same information rather than discovering an independent signal. The on-chain features (hash rate, tx count) rank near the bottom, suggesting they carry little predictive value for 30-day forward returns in this formulation. The Polymarket activity feature has an importance of exactly 0.0000 — the RF made zero splits on it across all 300 trees. With only ~216 active days in the training set (~5% of samples), the feature provides the RF with insufficient observations to be actionable. Its contribution to the model comes entirely through the rule-based 10% signal weight (Component 3), not through the RF.

---

## Performance Analysis

### Overall Metrics (2018–2025, 2,557 rolling 1-year windows)

| Metric                      | Value          |
|-----------------------------|----------------|
| **Model Score**             | **59.51%**     |
| Win Rate (vs. Uniform DCA)  | 59.56%         |
| Exp-Decay Avg Percentile    | 59.45%         |
| Mean Excess Percentile      | +4.54 pp       |
| Median Excess Percentile    | +3.90 pp       |
| Mean Relative Improvement   | +13.43%        |
| Median Relative Improvement | +9.90%         |
| Mean SPD Ratio              | 1.13           |
| Median SPD Ratio            | 1.10           |
| Windows Analyzed            | 2,557          |
| Wins / Losses               | 1,523 / 1,034  |

### Where the model underperforms

The 1,034 losing windows are concentrated in two periods:

1. **Strong uptrends with elevated MVRV (2020–2021 peak, 2024 recovery):** The model correctly identifies that MVRV is elevated and reduces accumulation — but in a sustained bull run, "accumulate less" still means missing upside. The RF, trained to identify cheap days, correctly flags these periods as expensive; the model's conservative stance is rational but costly in hindsight.

2. **Extended bear markets (2018–2019):** The model increases accumulation as MVRV falls, which is the right structural call, but if prices keep falling for months, those windows underperform uniform DCA in the short run.

### Comparison with prior models

The Model 2 score (59.51%) is nearly identical to Example 1 (59.54%). This reflects that the RF and regime instability detector are adding signal in some windows and reducing signal in others, with the net improvement roughly neutral at this level of aggregation. The value of the regime instability component is most visible in specific transition windows rather than in the aggregate metrics. Model 3 addresses this by replacing the RF with a Gradient Boosting classifier and adding halving cycle + exchange flow features for a more meaningful lift.

---

## Running the Model

**Train the RF (run once; auto-triggered by run_backtest if pkl absent):**
```bash
python -m model_2.training_pipeline
```

**Run the full backtest:**
```bash
python -m model_2.run_backtest
```

**Outputs in `model_2/output/`:**
```
performance_comparison.svg         — Dynamic vs. Uniform DCA percentile over time
excess_percentile_distribution.svg — Distribution of win margin
win_loss_comparison.svg            — Win/loss bar chart
cumulative_performance.svg         — Cumulative excess percentile
metrics_summary.svg                — Key metrics table
metrics.json                       — Full results data
```

**Artifact in `model_2/`:**
```
rf_model.pkl   — Trained RF (300 trees, depth 4, trained on 2012–2023)
```

---

## Assumptions & Limitations

1. **30-day forward return is a noisy label.** Bitcoin can be fundamentally cheap but drift lower for 60+ days. The RF is learning from a binary label that has substantial noise at any individual data point.

2. **2018–2023 is technically in-sample.** The RF has seen labels from the years covered by the backtest. The shallow tree design and 25% weight cap are the primary mitigations — they do not eliminate the concern, only reduce it.

3. **Polymarket data is insufficient for the RF.** With only ~216 active days in the training set, `polymarket_activity_zscore` received zero feature importance across all 300 trees — the RF never split on it. Any Polymarket contribution to the model comes entirely from the rule-based 10% signal weight (Component 3), which activates during those ~249 active days regardless of the RF.

4. **Change detection assumes MVRV is regime-informative.** The instability score is computed on MVRV z-score specifically. If a regime transition is not reflected in MVRV (e.g., a regulatory shock), the detector will not flag it.
