# Stacking Sats: Improving Bitcoin Accumulation

Building and improving data-driven Bitcoin accumulation strategies, with a focus on utilizing signal from prediction market data.

See [stackingsats.org](https://www.stackingsats.org/) for more information.

---

## The Mission: Exploring Institutional Bitcoin Accumulation

As Bitcoin matures as an institutional asset, standard Dollar Cost Averaging (DCA) is a strong baseline, but there may be room for optimization. This project facilitates the design of **data-driven, long-only** accumulation strategies. The aim is to explore methods that maintain DCA's systematic discipline while potentially **improving acquisition efficiency** within fixed budgets and time horizons.

This repository contains the work of a capstone project for a Master of Science in Analytics degree. Starting from a sponsor-provided baseline template and reference implementation, we developed two additional models that progressively incorporate machine learning, on-chain analytics, and Bitcoin cycle theory to improve upon the baseline.

### Latest Tournament
Trilemma Foundation hosts tournaments to find the most efficient accumulation models.
* **Current/Recent:** [Stacking Sats Tournament - MSTR 2025](https://github.com/TrilemmaFoundation/stacking-sats-tournament-mstr-2025)

---

## Model Comparison

All models are evaluated over **2,557 rolling 1-year windows** from 2018-01-01 to 2025-12-31, measured against uniform DCA.

| Metric                   | Example 1 | Model 2 (RF) | **Model 3 (CPGB)** |
|--------------------------|-----------|---------------|---------------------|
| **Score**                | 59.54%    | 59.51%        | **60.62%**          |
| **Win Rate**             | 60.31%    | 59.56%        | **64.41%**          |
| Exp-Decay Percentile     | 58.78%    | 59.45%        | 56.83%              |
| Mean Excess Percentile   | +5.70%    | +4.54%        | **+5.95%**          |
| Median Excess Percentile | +6.43%    | +3.90%        | **+7.27%**          |
| Mean SPD Ratio           | 1.17      | 1.13          | **1.17**            |

> **Score** = 0.5 x Win Rate + 0.5 x Exp-Decay Avg Percentile.

**Key takeaways:**
- **Model 3** achieves the highest score (60.62%) and win rate (64.41%), beating uniform DCA in nearly 2 out of 3 rolling windows.
- **Model 2** performs at parity with Example 1 — the Random Forest classifier adds no meaningful lift (CV AUC ~0.49, indistinguishable from random), but the regime instability detector and Polymarket activity signal are validated as standalone components.
- **Model 3's** exp-decay percentile (56.83%) trails prior models, reflecting conservative allocations during the 2024-2025 BTC run to $100k — a known limitation of MVRV-based strategies in strong recovery periods.

---

## Model Evolution & Design Thinking

The progression from Example 1 through Model 3 reflects an iterative, hypothesis-driven approach:

### Example 1 (Sponsor-Provided Baseline)
A rule-based strategy combining **MVRV z-score** (valuation) and **200-day MA** (trend). Accumulates more when BTC is cheap relative to realized value and below its long-run trend. Simple, interpretable, and effective — 60.3% win rate.

### Model 2: Can ML and Change Detection Improve the Baseline?
**Hypothesis:** A Random Forest classifier might learn non-linear interactions between features that hand-crafted rules miss. Additionally, the 200-day MA lags badly at regime transitions — a causal change detector could identify these transitions earlier.

**What we found:**
- The **RF classifier failed to add signal** — CV AUC of 0.49 (random). It was learning a noisy restatement of the same MVRV/MA features. The Polymarket activity feature received exactly 0.0% importance (only ~249 days of coverage, insufficient for the RF to split on).
- The **regime instability detector** (distributional shift test on MVRV z-score) worked as designed — it correctly identifies regime transitions and dampens signals during uncertain periods.
- The **Polymarket activity signal** (expanding-window trade-count z-score) showed statistical significance in EDA (p=0.029 at 30 days), but small sample size and limited coverage constrains its impact.

**Lesson learned:** With the available feature set, a shallow RF cannot reliably predict 30-day forward returns beyond what rule-based signals already capture. The graceful degradation design (RF defaults to 0.5 probability when uninformative) meant this did not hurt the model — it simply fell back to the MVRV + MA + Polymarket baseline.

> Full details: [`model_2/model_2.md`](model_2/model_2.md)

### Model 3: Cycle Awareness, Exchange Flows, and Walk-Forward Training
**Hypothesis:** Two structurally motivated signals could provide information orthogonal to MVRV — Bitcoin's 4-year halving cycle (supply-side) and exchange flow dynamics (demand-side). Additionally, replacing the RF with Gradient Boosting and using walk-forward retraining would address the in-sample overlap problem.

**What we found:**
- **Halving cycle phase** is the dominant GBM signal (39% combined feature importance). The sin/cos encoding allows the GBM to learn that "MVRV of 1.0 at 6 months post-halving" is a fundamentally different opportunity than "MVRV of 1.0 at 36 months post-halving."
- **Exchange flow dynamics** (net flow z-score + supply velocity) provide a genuine orthogonal signal — MVRV asks "is it cheap?" while exchange flows ask "are large holders actually accumulating?"
- **Gradient Boosting** outperforms the RF (CV AUC 0.557 vs 0.493) because its sequential error-correction focuses on rare market states (deep value, early post-halving) where the best accumulation opportunities exist.
- **Walk-forward retraining** ensures every backtest prediction is strictly out-of-sample, correcting a ~7pp win-rate inflation found in the single-model approach.

> Full details: [`model_3/model_3.md`](model_3/model_3.md)

---

## Repository Structure

```text
.
├── template/                          # CORE FRAMEWORK (Sponsor-provided baseline)
│   ├── prelude_template.py            # Data loading & Polymarket utilities
│   ├── model_development_template.py  # Baseline MA-200 model logic
│   ├── backtest_template.py           # Evaluation engine
│   └── *.md                           # Documentation for model logic & backtesting
├── example_1/                         # REFERENCE IMPLEMENTATION (Sponsor-provided)
│   ├── run_backtest.py                # How to run the example
│   └── model_development_example_1.py # MVRV + Polymarket integration
├── model_2/                           # REGIME-AWARE RF + CHANGE DETECTION
│   ├── model_development_2.py         # Model logic (inference)
│   ├── training_pipeline.py           # RF training (isolated from inference)
│   ├── run_backtest.py                # Backtest runner
│   └── model_2.md                     # Full documentation
├── model_3/                           # CYCLE-AWARE GRADIENT BOOSTING (CPGB)
│   ├── model_development_3.py         # Model logic (inference)
│   ├── training_pipeline.py           # Walk-forward GBM training
│   ├── run_backtest.py                # Backtest runner
│   └── model_3.md                     # Full documentation
├── data/                              # Bitcoin & Polymarket source data
├── output/                            # Baseline results and visualizations
└── tests/                             # Unit tests for core logic
```

---

## Getting Started

### 1. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mmidlash/bitcoin-analytics-capstone-template
    cd bitcoin-analytics-capstone-template
    ```

2.  **Setup environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

### 2. Data Acquisition

The `data/` directory contains historical BTC price data and specific Polymarket datasets (Politics, Finance, Crypto).

Data can be [downloaded manually from Google Drive](https://drive.google.com/drive/folders/1gizJ_n-QCnE8qrFM-BU3J_ZpaR3HCjn7?usp=sharing) into the `data/` folder, or you can use the automated script:

```bash
python data/download_data.py
```

**Included Data:**
* **CoinMetrics BTC Data**: Daily OHLCV and network metrics.
  * **Bitcoin Price Source of Truth**: The `PriceUSD` column in the CoinMetrics data is the source of truth for BTC-USD prices. This is renamed to `PriceUSD_coinmetrics` in the codebase.
* **Polymarket Data**: High-fidelity parquet files containing trades, odds history, and market metadata.
  * **Timestamp note**: Some parquet timestamp columns are stored with incorrect
    units (millisecond values encoded as microseconds). Direct reads can show
    dates near 1970. Use the built-in loaders in `template/prelude_template.py`
    or `eda/eda_starter_template.py`, which detect and correct these values at
    runtime.

**External Data:**
External data is encouraged; students are responsible for ensuring that the data license permits all project participants to access and use (i.e., no proprietary data).

**System Requirements:**
Assume a modern laptop specification (think 16GB M4 Air).

### 3. Running the Models

**Backtest Date Range:** `2018-01-01` to `2025-12-31` (inclusive; daily frequency; no missing days). The backtest engine uses rolling 1-year windows starting from the start date.

#### Template Baseline
```bash
python -m template.backtest_template
```

#### Example 1 (Reference Implementation)
```bash
python -m example_1.run_backtest
```

#### Model 2 (Regime-Aware RF + Change Detection)

```bash
python -m model_2.run_backtest
```

> On first run, this trains the Random Forest (`model_2/rf_model.pkl`) and then runs the full backtest. To retrain the RF only: `python -m model_2.training_pipeline`.

Results are saved to `model_2/output/` (SVG charts + `metrics.json`).

#### Model 3 (Cycle-Aware Gradient Boosting)

Generate walk-forward probabilities, then run the backtest:
```bash
python -m model_3.run_backtest
```

> On first run, this generates `model_3/gbm_walkforward_probs.parquet` (walk-forward OOS probabilities) and then runs the full backtest. To regenerate walk-forward probs only: `python -m model_3.training_pipeline`.

Results are saved to `model_3/output/` (SVG charts + `metrics.json`).

---

## Technical Highlights

### No-Lookahead Guarantee (All Models)
Every feature entering the weight multiplier on day `t` reflects information available through day `t-1`. A single `.shift(1)` is applied to all signal columns in `precompute_features()`. ML models are static artifacts that never access current-day data during inference.

### Training / Inference Separation (Models 2 & 3)
Training pipelines (`training_pipeline.py`) use future prices to construct labels (`price.shift(-30)`). These modules are **never imported** by the inference modules (`model_development_*.py`). The only crossing artifacts are serialized model files (`rf_model.pkl`, `gbm_walkforward_probs.parquet`).

### Graceful Degradation (Models 2 & 3)
If ML artifacts are missing, the models degrade cleanly to their rule-based baselines. If external data (Polymarket, exchange flows) is absent, those signal components default to zero (neutral).

---

## Key Performance Indicators

The automated backtest engine calculates these metrics:

1.  **Win Rate**: How often the strategy outperforms uniform DCA over 1-year windows.
2.  **SPD (Sats Per Dollar)**: Raw efficiency — are you acquiring more bitcoin for the same capital?
3.  **Model Score**: Composite metric (0.5 x Win Rate + 0.5 x Exp-Decay Avg Percentile) balancing consistency with recency-weighted performance.

## Licensing

*   **Code:** This repository, including its analysis and documentation, is open-sourced under the **MIT License**.
*   **Data:** The data provided (e.g., CoinMetrics, Polymarket) is not covered by the MIT license and retains its original licensing terms. Please refer to the respective data providers for their terms of use.

---

## Contacts & Community

* **App:** [stackingsats.org](https://www.stackingsats.org/)
* **Website:** [trilemma.foundation](https://www.trilemma.foundation/)
* **Foundation:** [Trilemma Foundation](https://github.com/TrilemmaFoundation)
