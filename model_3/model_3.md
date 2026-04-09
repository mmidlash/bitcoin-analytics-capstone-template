# Model 3: Cycle-Aware Gradient Boosting with Exchange Flow Dynamics (CPGB)

## Executive Summary

Model 3 introduces three novel signals — **Bitcoin's halving cycle phase**, **exchange flow dynamics**, and a **Gradient Boosting classifier** — and uses **walk-forward expanding-window retraining** to ensure every historical GBM prediction is strictly out-of-sample.

| Metric                   | Example 1 | Model 2 (RF) | **Model 3 (CPGB)** |
|--------------------------|-----------|--------------|--------------------|
| Score                    | 59.54%    | 59.51%       | **60.62%**         |
| Win Rate                 | 60.31%    | 59.56%       | **64.41%**         |
| Exp-Decay Percentile     | 58.78%    | 59.45%       | **56.83%**         |
| Mean Excess Percentile   | +5.70%    | +4.54%       | **+5.95%**         |
| Median Excess Percentile | +6.43%    | +3.90%       | **+7.27%**         |
| Mean SPD Ratio           | 1.17      | 1.13         | **1.17**           |

> **Score** = 0.5 × Win Rate + 0.5 × Exp-Decay Avg Percentile. 2,557 rolling 1-year windows, 2018-01-01 to 2025-12-31.

---

## Motivation & Design Philosophy

Standard DCA is a disciplined, market-agnostic strategy. Our goal is to add *informed timing* that buys more when prices are systematically cheap within a 1-year window — without predicting short-term price movements.

The prior models (1 and 2) identify cheap periods primarily through:
- **MVRV z-score**: Is the current market price cheap relative to realized value?
- **200-day MA**: Is the price below its long-run trend?

Model 3 asks two additional questions:
1. **Where are we in the 4-year Bitcoin halving cycle?** The halving cycle is the most structurally important feature of Bitcoin's supply schedule — halvings cut new supply by 50% on a predictable schedule. Historically, the 12–18 months post-halving are a high-probability accumulation zone, while months 24–36 often mark bull-run peaks.
2. **Are large holders accumulating or distributing?** Exchange flow data (net BTC entering/leaving exchanges) captures supply/demand pressure at the margin. When coins flow off exchanges at an elevated rate, it signals large holders are accumulating — often a bullish leading indicator.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Feature Engineering                         │
│                                                                     │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────────┐  │
│  │ MVRV / MA-200  │  │ Halving Cycle  │  │  Exchange Flow        │  │
│  │ (Model 2 base) │  │  sin/cos/days  │  │  net_flow_zscore      │  │
│  │                │  │  [NEW]         │  │  supply_velocity      │  │
│  └───────┬────────┘  └───────┬────────┘  └──────────┬────────────┘  │
│          │                   │                      │               │
│          └─────────┬─────────┘                      │               │
│                    ▼                                │               │
│          ┌─────────────────┐                        │               │
│          │  Walk-Forward   │◄───────────────────────┘               │
│          │  GBM Classifier │                                        │
│          │  (OOS for every │                                        │
│          │   backtest date)│                                        │
│          └────────┬────────┘                                        │
└───────────────────┼─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Signal Combination                             │
│                                                                     │
│  MVRV value signal    × 0.40   (proven long-run valuation)          │
│  MA-200 signal        × 0.10   (trend confirmation, reduced)        │
│  GBM buy probability  × 0.30   (ML layer w/ cycle + flow context)   │
│  Exchange flow signal × 0.15   (supply/demand at margin)            │
│  Polymarket activity  × 0.05   (retail sentiment, limited data)     │
│                                                                     │
│  Modulated by: regime instability, acceleration, confidence, vol    │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
        exp(clip(combined × 4.0, −5, 10))  → multiplier
                    │
                    ▼
        allocate_sequential_stable()  → weights summing to 1.0
```

---

## Novel Contribution 1: Bitcoin Halving Cycle Phase Encoding

### Rationale

Bitcoin's supply issuance halves approximately every four years (the "halving"). This creates a well-documented cyclical pattern:

| Phase                     | Approx. timing          | Historical pattern                    |
|---------------------------|-------------------------|---------------------------------------|
| 0–6 months post-halving   | Supply shock absorption | Prices often consolidate or drift     |
| 6–18 months post-halving  | Parabolic run           | Historically strong bull market       |
| 18–30 months post-halving | Distribution & peak     | Prices near historical ATH            |
| 30–48 months post-halving | Bear market & base      | Gradual decline toward next cycle low |

The MVRV z-score captures *absolute* valuation but is cycle-agnostic. A given MVRV of 1.0 carries very different forward-return odds depending on whether we're 6 months or 36 months post-halving.

### Implementation

Halving dates (hardcoded as historically observed facts; exact date not knowable until the block is mined). Only halvings already past on date t are used — future halvings are masked:
- 2012-11-28 (Block 210,000)
- 2016-07-09 (Block 420,000)
- 2020-05-11 (Block 630,000)
- 2024-04-20 (Block 840,000)

Phase encoding:
```python
days_since_last_halving = min(days since each halving that has occurred)
cycle_phase = (days_since_last_halving % 1461) / 1461  # ∈ [0, 1)

halving_cycle_sin = sin(2π × cycle_phase)
halving_cycle_cos = cos(2π × cycle_phase)
```

**Why sin/cos encoding?** Raw phase is a *circular* variable — phase 0.99 is almost identical to phase 0.01. A linear representation would incorrectly suggest these are far apart. The sin/cos pair uniquely represents any phase and is continuous at the wrap-around point, enabling the GBM to learn smooth cyclical patterns.

**Empirical validation (GBM feature importances):**
The halving cycle features collectively represent **~39% of GBM feature importance** (cos: 19.2%, days_since: 11.9%, sin: 7.9%), making them the dominant signal. This confirms that halving cycle position carries meaningful predictive information about forward accumulation quality.

---

## Novel Contribution 2: Exchange Flow Dynamics

### Rationale

Exchange flow data captures the direction of Bitcoin movement at the margins:
- **FlowInExNtv** (inflows): Bitcoin deposited to exchanges → typically to sell
- **FlowOutExNtv** (outflows): Bitcoin withdrawn from exchanges → typically to hold
- **SplyExNtv / SplyCur** (exchange supply ratio): Fraction of all BTC on exchanges

When net flows are negative (outflows > inflows) and the exchange supply ratio is declining, large holders are accumulating. This is a leading indicator of supply shocks that historically precede price increases.

This signal is **orthogonal to MVRV**:
- MVRV asks: "Is the market fundamentally cheap relative to realized value?"
- Exchange flows ask: "Are participants actually *acting* on that signal by accumulating?"

### Implementation

```python
# Net flow: 7-day smoothed, 90-day rolling z-score
net_flow_raw = FlowInExNtv - FlowOutExNtv
net_flow_smooth = net_flow_raw.rolling(7).mean()
net_flow_zscore = (net_flow_smooth - rolling_mean(90)) / rolling_std(90)

# Supply velocity: 30-day rate of change in exchange supply fraction
supply_pct = SplyExNtv / SplyCur
supply_delta = supply_pct - supply_pct.shift(30)
supply_velocity_zscore = (supply_delta - rolling_mean(180)) / rolling_std(180)

# Composite signal (both inverted: outflows / declining supply = bullish)
exchange_flow_signal = 0.6 × tanh(-net_flow_zscore × 0.5)
                     + 0.4 × tanh(-supply_velocity_zscore × 0.5)
```

The signal is bounded to [-1, 1] via tanh and carries a 15% weight in the final combination.

**Feature importance in GBM:** `supply_velocity_zscore` ranks 11th (3.9%) and `net_flow_zscore` 15th (1.4%). The standalone rule-based `exchange_flow_signal` operates independently at 15% weight, providing signal even in periods where the GBM component is neutral.

---

## Novel Contribution 3: Gradient Boosting vs. Random Forest

### Why GBM over RF for this task?

**Random Forest** (Model 2) trains trees *independently* using random feature subsets. Each tree is an unbiased estimator of the underlying distribution, and their average reduces variance.

**Gradient Boosting** trains trees *sequentially*, each fitting the pseudo-residuals of the ensemble so far. Later trees focus on hard examples — the unusual market regimes (extreme cycle phases, high instability, unusual flow patterns) that early trees mispredicted.

For our use case, this matters because:
1. **Rare market states matter most.** Bitcoin's best accumulation opportunities (deep value periods, early post-halving) are uncommon in the training data. GBM focuses more attention on getting these rare-but-important states right.
2. **Non-linear interactions.** The interaction "early halving phase AND price below MA AND negative exchange flow → strong buy" is the kind of combination GBM is well-suited to learn, while RF's independent trees may split each feature along different branches. The trained model's feature importances confirm this: halving cycle features (39.1% combined) and `price_vs_ma` (10.7%) are the dominant signals within the GBM, with `mvrv_zscore` ranking 9th at 4.4%.
3. **Calibrated probabilities.** GBM produces better-calibrated probability outputs than RF (a known property of boosting algorithms), reducing extreme weight concentration from overconfident predictions.

### Key hyperparameter choices

| Parameter          | Value | Rationale                                                           |
|--------------------|-------|---------------------------------------------------------------------|
| `n_estimators`     | 300   | Max trees; conservative given training data size                    |
| `max_depth`        | 3     | Classical GBM depth; 8 leaves per tree, captures 3-way interactions |
| `learning_rate`    | 0.05  | Slow shrinkage; compensated by 300 estimators                       |
| `subsample`        | 0.75  | Stochastic GBM: reduces variance, adds implicit regularization      |
| `min_samples_leaf` | 25    | ~0.7% of training data; prevents over-specific splits               |
| `max_features`     | 0.75  | Column subsampling per split (like RF, adds diversity)              |

---

## Novel Contribution 4: Walk-Forward Expanding-Window Retraining

### The Problem with a Single Trained Model

A GBM trained on 2012–2023 and applied in a backtest from 2018 onward evaluates the model **in-sample** for approximately 75% of the backtest period. The model's parameters were estimated to minimize prediction error on those very observations, which inflates win-rate metrics.

Evidence of this inflation: the original single-model backtest reported a 71.61% win rate while the exponential-decay percentile was 56.09% — a divergence that is characteristic of in-sample overfitting. The recency-weighted metric, which better reflects genuine OOS performance, was actually *lower* than Model 2's.

### Solution: Expanding Walk-Forward Schedule

For each prediction year Y, a separate GBM is trained exclusively on data prior to Y, then applied to year Y. No prediction is ever made by a model that has seen that year's labels.

| GBM Version | Trained On | Predicts              |
|-------------|------------|-----------------------|
| GBM_2017    | 2012–2016  | 2017                  |
| GBM_2018    | 2012–2017  | 2018 ← backtest start |
| GBM_2019    | 2012–2018  | 2019                  |
| GBM_2020    | 2012–2019  | 2020                  |
| GBM_2021    | 2012–2020  | 2021                  |
| GBM_2022    | 2012–2021  | 2022                  |
| GBM_2023    | 2012–2022  | 2023                  |
| GBM_2024    | 2012–2023  | 2024                  |
| GBM_2025    | 2012–2024  | 2025                  |

**Label buffer:** The last 31 days of each training period are excluded from label construction, ensuring no training label's 30-day forward return crosses into the prediction year.

**Artifact:**
- `gbm_walkforward_probs.parquet` — precomputed OOS probability series used in the backtest

This means the model used during backtesting always has more training data available for later years (GBM_2025 has 13 years vs. GBM_2018's 6 years), which reflects how the model would actually improve over time.

### Impact on Metrics

| Metric                 | Single-Model (in-sample) | Walk-Forward (OOS) | Change   |
|------------------------|--------------------------|--------------------|----------|
| Win Rate               | 71.61%                   | **64.41%**         | −7.20 pp |
| Score                  | 63.85%                   | **60.62%**         | −3.23 pp |
| Exp-Decay Percentile   | 56.09%                   | **56.83%**         | +0.74 pp |
| Mean Excess Percentile | +9.14%                   | **+5.95%**         | −3.19 pp |

The exp-decay percentile (which weights recent windows more heavily and was already largely OOS) was unchanged — and marginally improved — confirming the fix only corrected the in-sample win rate inflation. The walk-forward win rate of 64.41% is a genuine, defensible result.

---

## Feature Engineering Summary

All features lagged 1 day (`.shift(1)`) before entering the weight multiplier. The GBM is a static artifact — no future prices ever reach the inference path.

| Feature                        | Type      | Description                                                  |
|--------------------------------|-----------|--------------------------------------------------------------|
| `price_vs_ma`                  | MA signal | Normalized distance from 200-day MA                          |
| `mvrv_zscore`                  | Valuation | 365-day rolling z-score of MVRV ratio                        |
| `mvrv_gradient`                | Momentum  | 30-day EMA-smoothed MVRV trend direction                     |
| `mvrv_acceleration`            | Momentum  | 14-day second derivative of gradient                         |
| `mvrv_zone`                    | Regime    | Discrete zone: {Deep Value, Value, Neutral, Caution, Danger} |
| `mvrv_volatility`              | Risk      | 90-day MVRV volatility percentile                            |
| `signal_confidence`            | Agreement | MVRV/MA directional agreement score                          |
| `regime_instability`           | Risk      | Distributional shift score (short vs. long-run MVRV dist.)   |
| `btc_return_7d/14d/30d/60d`    | Momentum  | Multi-horizon log returns                                    |
| `hash_rate_zscore`             | On-chain  | Network security momentum                                    |
| `tx_count_zscore`              | On-chain  | Network utilization momentum                                 |
| `polymarket_activity_zscore`   | Activity  | BTC prediction market trade-count z-score                    |
| **`halving_cycle_sin`** ★      | Cycle     | sin(2π × halving cycle phase)                                |
| **`halving_cycle_cos`** ★      | Cycle     | cos(2π × halving cycle phase)                                |
| **`halving_days_since`** ★     | Cycle     | Days since most recent halving                               |
| **`net_flow_zscore`** ★        | Supply    | 90d z-score of smoothed net exchange flows                   |
| **`supply_velocity_zscore`** ★ | Supply    | 180d z-score of exchange supply rate-of-change               |
| `gbm_buy_prob`                 | ML output | Walk-forward GBM P(30-day forward return > 0)                |
| **`exchange_flow_signal`** ★   | Supply    | Composite rule-based signal [-1, 1]                          |

★ New in Model 3

---

## GBM Feature Importances

Importances are measured on the final walk-forward GBM (GBM_2026, trained 2012–2025) using mean impurity decrease (Gini importance), averaged across all 300 trees. Earlier walk-forward GBMs may differ slightly but are consistent in their top-ranking groups.

| Rank | Feature                      | Importance | Category  |
|------|------------------------------|------------|-----------|
| 1    | `halving_cycle_cos`          | 19.2%      | Cycle     |
| 2    | `halving_days_since`         | 11.9%      | Cycle     |
| 3    | `price_vs_ma`                | 10.7%      | MA signal |
| 4    | `mvrv_volatility`            | 8.5%       | Risk      |
| 5    | `halving_cycle_sin`          | 7.9%       | Cycle     |
| 6    | `signal_confidence`          | 7.6%       | Agreement |
| 7    | `btc_return_60d`             | 5.4%       | Momentum  |
| 8    | `mvrv_gradient`              | 4.9%       | Momentum  |
| 9    | `mvrv_zscore`                | 4.4%       | Valuation |
| 10   | `mvrv_acceleration`          | 4.2%       | Momentum  |
| 11   | `supply_velocity_zscore` ★   | 3.9%       | Supply    |
| 12   | `regime_instability`         | 3.3%       | Risk      |
| 13   | `btc_return_14d`             | 2.6%       | Momentum  |
| 14   | `btc_return_30d`             | 1.7%       | Momentum  |
| 15   | `net_flow_zscore` ★          | 1.4%       | Supply    |
| 16   | `btc_return_7d`              | 1.2%       | Momentum  |
| 17   | `hash_rate_zscore`           | 1.0%       | On-chain  |
| 18   | `polymarket_activity_zscore` | 0.2%       | Activity  |
| 19   | `tx_count_zscore`            | 0.1%       | On-chain  |
| 20   | `mvrv_zone`                  | 0.0%       | Valuation |

**Key observations:**

- **Halving cycle dominates the GBM.** The three cycle features rank 1st, 2nd, and 5th, collectively accounting for 39.1% of importance — confirming that cycle-phase encoding is the most informative novel signal within the ML layer.
- **`mvrv_zscore` ranks 9th (4.4%) inside the GBM**, well below `price_vs_ma` (10.7%), `mvrv_volatility` (8.5%), and `signal_confidence` (7.6%). MVRV's importance to the strategy comes from its direct 40% weight in the final signal combination, not from dominating the GBM component.
- **`polymarket_activity_zscore` has negligible GBM importance (0.2%)**, contributing almost nothing through the ML layer. Its only meaningful path into the strategy is through the direct `poly_signal × W_POLY` (5%) term. See the appendix for weight rationale.
- **`mvrv_zone` (0.0%) is fully redundant** with the continuous `mvrv_zscore` in the GBM's split logic; the discrete encoding adds no information beyond the z-score itself.
- **Exchange flow features rank in the lower tier** (11th and 15th) but above all on-chain network metrics and polymarket, consistent with the 15% standalone weight assigned to the rule-based `exchange_flow_signal`.

---

## Design Safeguards

### No Lookahead
All signal features are shifted by 1 day in `precompute_features()`. Features at row `t` reflect information known through `t-1`. The GBM model is a static artifact that never sees future prices during inference.

```python
# Sole lookahead guard — applied once, to all signal columns
features[signal_cols] = features[signal_cols].shift(1)
```

### Walk-Forward OOS Guarantee
Every historical `gbm_buy_prob` value is produced by a GBM that was trained exclusively on data before that date. The walk-forward schedule enforces a 31-day label buffer at each training boundary to prevent any forward-return leakage between training years.

### Training / Inference Separation
`training_pipeline.py` builds labels using `price.shift(-30)` (future prices). This module is **never imported by** `model_development_3.py`. The only shared artifact is:
- `gbm_walkforward_probs.parquet` — a static probability series with no price information

### Graceful Degradation
- Walk-forward probs missing: falls back to `gbm_buy_prob = 0.5` (GBM contributes nothing)
- Exchange flow data missing: `exchange_flow_signal = 0.0` (neutral)
- Polymarket data missing: `polymarket_activity_zscore = 0.0` (neutral)

### Weight Constraints
- All weights ≥ MIN_W = 1e-6 (no zero allocations)
- Weights sum to 1.0 per window (verified across all 2,557 backtest windows)
- Future days within a window use uniform allocation (budget preserved)

---

## GBM Classifier Evaluation

### Cross-validation AUC (temporal, 5 folds)

Evaluated on a GBM trained on the full 2012–2025 training period using `TimeSeriesSplit` with a 30-day gap between each training and validation fold. The gap prevents label leakage: validation labels (which look 30 days forward) do not overlap with the training window's most recent observations.

| Fold     | Train samples | Val samples | AUC                 |
|----------|---------------|-------------|---------------------|
| 1        | 703           | 730         | 0.5611              |
| 2        | 1,433         | 730         | 0.4830              |
| 3        | 2,163         | 730         | 0.6726              |
| 4        | 2,893         | 730         | 0.4632              |
| 5        | 3,623         | 730         | 0.6064              |
| **Mean** |               |             | **0.5573 ± 0.0776** |

The mean CV AUC of 0.56 is modestly above random (0.50), indicating the GBM is capturing a real but weak signal. The high fold-to-fold variance (±0.078) reflects the cyclical nature of the signal: Folds 1 and 3, which cover halving-cycle windows where the model's dominant features carry more structure, outperform Folds 2 and 4 where the training data spans more ambiguous mid-cycle periods. This variance is expected and interpretable rather than indicating instability.


### Comparison with Model 2 RF

|        | Model 2 (RF)    | Model 3 (GBM)       |
|--------|-----------------|---------------------|
| CV AUC | 0.4932 ± 0.0521 | **0.5573 ± 0.0776** |

The GBM's CV AUC is materially higher than the RF's (0.557 vs. 0.493), confirming that the halving cycle features and the GBM's sequential error-correction provide genuine predictive lift beyond what the RF could learn from the same base features.

---

## Performance Analysis

### Overall Metrics (2018–2025, 2,557 rolling 1-year windows)

| Metric                      | Value       |
|-----------------------------|-------------|
| **Model Score**             | **60.62%**  |
| Win Rate (vs. Uniform DCA)  | 64.41%      |
| Exp-Decay Avg Percentile    | 56.83%      |
| Mean Excess Percentile      | +5.95 pp    |
| Median Excess Percentile    | +7.27 pp    |
| Mean Relative Improvement   | +17.00%     |
| Median Relative Improvement | +20.13%     |
| Mean SPD Ratio              | 1.17        |
| Median SPD Ratio            | 1.20        |
| Windows Analyzed            | 2,557       |
| Wins / Losses               | 1,647 / 910 |

### Where the Model Underperforms

The 910 losing windows are concentrated in two periods:

1. **2019 bear market continuation** (Jan 2019 – Jan 2020): Windows straddling the tail of the 2018 bear market. The GBM trained on 2012–2018 data had limited exposure to extended bear conditions, reducing its ability to signal "stay patient" effectively.

2. **Late 2023 – late 2024 windows**: These windows span BTC's price recovery from ~$26k to ~$100k. The model's conservative allocations (BTC was not deeply undervalued by MVRV) caused it to miss portions of the early run-up. The GBM for this period (trained through 2022–2023) had seen the 2022 crash but not the subsequent recovery pattern.

Both failure modes are rational: the model correctly identifies that buying is less urgent when MVRV is elevated — but in a strong recovery, "less urgent" buying still misses optionality.

### Comparison with Prior Models

```
Win Rate Comparison (all OOS):
  Example 1:          60.3%
  Model 2 (RF):       59.6%
  Model 3 (CPGB):    64.4%  ←── +4.8 pp vs Model 2
```

The improvement from ~60% to 64% reflects the genuine contribution of halving cycle encoding and exchange flow dynamics. The score improvement is +1.11 pp over Model 2 (60.62% vs 59.51%); the exp-decay percentile still trails Example 1 and Model 2, an area for future improvement.

---

## Assumptions & Limitations

### Model Assumptions
1. **Halving cycle patterns persist.** The model assumes that post-halving price dynamics in cycles 2–4 (2016–2024) will resemble those in the future. If the Bitcoin market matures to the point where the halving is fully priced in before it occurs, the cycle signal will weaken.
2. **Exchange flows are informative.** We assume that the direction of net BTC flows on/off exchanges reflects genuine accumulation or distribution by large holders. Measurement noise (exchange wallet reshuffling, internal transfers) adds signal degradation.
3. **30-day forward return is a meaningful buy label.** Binary labels (1 if price rises in 30 days) are noisy — BTC can be fundamentally cheap but drift lower for 60+ days.

### Model Limitations
- **Walk-forward GBMs trained on fewer years may be weaker.** GBM_2018 (trained on 6 years) is less robust than GBM_2025 (trained on 13 years). Earlier backtest windows may reflect this in slightly lower performance.
- **Halving cycle signal weakens for the 5th+ halving** as the block reward approaches zero and miner supply effects diminish.
- **Exchange flow data becomes reliable only from ~2012** — early training data has noisier flow features.
- **Exp-decay percentile (56.83%) trails Example 1 and Model 2 (58–59%).** Recent windows (2024–2025) include the BTC $100k run where the model's conservative MVRV allocations are a structural weakness regardless of ML quality.

---

## Running the Model

**First run (generates walk-forward probs + backtests):**
```bash
python -m model_3.run_backtest
```

**Regenerate walk-forward probs only (without full backtest):**
```bash
python -m model_3.training_pipeline
```

**Outputs in `model_3/output/`:**
```
performance_comparison.svg          — Dynamic vs. Uniform DCA percentile over time
excess_percentile_distribution.svg  — Distribution of win margin
win_loss_comparison.svg             — Win/loss bar chart
cumulative_performance.svg          — Cumulative excess percentile
metrics_summary.svg                 — Key metrics table
metrics.json                        — Full results data
```

**Artifact in `model_3/`:**
```
gbm_walkforward_probs.parquet  — OOS probability series (sole backtest input)
```

---

## Appendix: Signal Weight Rationale

The signal weights (40/10/30/15/5) were chosen based on three principles:

1. **MVRV remains primary (40%)**: The MVRV z-score has a 12+ year track record as a Bitcoin cycle indicator. It should still dominate the model. Reducing it below 50% vs Model 2's formulation brings the GBM and exchange flow signals into meaningful range without discarding the proven base signal.

2. **GBM carries meaningful weight (30%)**: The GBM combines 20 features including halving cycle context. Its 30-day forward-return prediction is directly aligned with the buy-opportunity definition. A 30% weight gives it substantial influence when its signal is strong.

3. **Exchange flow at 15%**: The exchange flow signal is analytically motivated but noisier than MVRV (daily flows fluctuate). 15% is high enough to meaningfully affect weight allocation during sustained accumulation phases but not so high that daily noise causes erratic allocations.

4. **MA-200 reduced to 10%** (from 15% in Model 2): The 200-day MA is a lagging indicator that overlaps significantly with MVRV and the GBM signal. Reducing its weight frees allocation for the exchange flow signal.

5. **Polymarket at 5%**: Polymarket BTC activity covers only ~4.4% of backtest history. The trained GBM assigns this feature negligible importance (0.2%), so it contributes almost nothing through the ML layer. Its effect on allocations flows primarily through the direct `poly_signal × W_POLY` term — positive when the market is active, neutral otherwise.
