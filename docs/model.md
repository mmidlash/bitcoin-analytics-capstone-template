# Bitcoin DCA Weight Computation Model

This document explains the ML model that computes dynamic DCA (Dollar Cost Averaging) weights for Bitcoin investment strategies based on MVRV (Market Value to Realized Value) and price relative to the 200-day moving average.

## Overview

The model computes daily investment weights that determine how much of your DCA budget to allocate on each day within a given investment window. Unlike uniform DCA (equal daily amounts), this model dynamically adjusts weights based on:

1. **MVRV Z-score**: Buy more when undervalued (low MVRV), less when overvalued
2. **Price vs 200-day MA**: Buy more when price is below long-term trend
3. **MVRV Momentum**: Acceleration/deceleration of MVRV trends
4. **Signal Confidence**: Amplify signals when multiple indicators agree
5. **Volatility Dampening**: Reduce exposure during high uncertainty periods

**Key Properties:**
- Weights sum to exactly 1.0 for each date range (within tolerance of 1e-6)
- All weights are ≥ MIN_W (1e-6) to prevent zero allocations
- Past weights are **locked** once computed and never change
- Future weights are **uniform** (remaining budget distributed evenly)
- Weights are deterministic given the same inputs

## Model Architecture

The weight computation uses an enhanced **MVRV + 200-day MA Signal Strategy** with multiple signal layers.

```mermaid
flowchart LR
    subgraph inputs [Inputs]
        P[Price History]
        M[MVRV Data]
    end
    
    subgraph features [Feature Construction]
        MA200[200-Day SMA]
        MVRVZ[MVRV Z-Score<br/>365-day window]
        GRAD[MVRV Gradient<br/>30-day EMA]
        ACCEL[MVRV Acceleration<br/>14-day window]
        VOL[MVRV Volatility<br/>90-day window]
        ZONE[Zone Classification]
    end
    
    subgraph signals [Signal Processing]
        VALUE[Value Signal<br/>80% weight]
        MASIG[MA Signal<br/>20% weight]
    end
    
    subgraph modifiers [Signal Modifiers]
        BOOST[Asymmetric<br/>Extreme Boost]
        TREND[Adaptive Trend<br/>Modifier]
        CONF[Confidence<br/>Boost]
        DAMP[Volatility<br/>Dampening]
    end
    
    subgraph output [Output]
        DYN[Dynamic Multiplier]
        ALLOC[Sequential Allocation]
        W[Final Weights]
    end
    
    P --> MA200
    M --> MVRVZ
    MVRVZ --> GRAD
    GRAD --> ACCEL
    MVRVZ --> VOL
    MVRVZ --> ZONE
    MA200 --> MASIG
    
    MVRVZ --> VALUE
    ZONE --> BOOST
    BOOST --> VALUE
    
    GRAD --> TREND
    TREND --> MASIG
    
    VALUE --> DYN
    MASIG --> DYN
    
    ACCEL --> DYN
    CONF --> DYN
    DAMP --> DYN
    
    DYN --> ALLOC
    ALLOC --> W
```

For a window of `n` days, the weight for day `i` is computed as:

```
weight[i] = allocate_sequential_stable(base[i] × dynamic[i])
```

Where:
- `base[i]` = 1/n (uniform base distribution)
- `dynamic[i]` = exp(combined_signal × DYNAMIC_STRENGTH)

## Signal Composition

The model combines two primary signals with modulation factors:

### Primary Signals (Weighted Combination)

| Signal | Weight | Description |
|--------|--------|-------------|
| MVRV Value Signal | 80% | Low MVRV Z-score → buy more, with asymmetric extreme boost |
| MA Signal | 20% | Below 200-day MA → buy more, with adaptive trend modulation |

### Signal Modifiers

| Modifier | Effect | Trigger |
|----------|--------|---------|
| Acceleration | ×[0.85, 1.15] | Momentum building/reversing |
| Confidence Boost | ×[1.0, 1.15] | Only when confidence > 0.7 |
| Volatility Dampening | ×[0.8, 1.0] | Only when volatility > 80th percentile |

## Feature Construction

### 200-Day Simple Moving Average

```python
price_ma200 = price.rolling(200, min_periods=100).mean()
price_vs_ma = (price / price_ma200) - 1  # Clipped to [-1, 1]
```

| Value | Interpretation |
|-------|----------------|
| `price_vs_ma > 0` | Price above 200-day MA (bullish) |
| `price_vs_ma < 0` | Price below 200-day MA (bearish) |
| `price_vs_ma = 0` | Price at 200-day MA |

### MVRV Z-Score

The MVRV Z-score normalizes the MVRV ratio using a 365-day rolling window:

```python
mvrv_zscore = (mvrv - rolling_mean(365)) / rolling_std(365)
# Clipped to [-4, 4]
```

| Z-Score | Zone | Interpretation |
|---------|------|----------------|
| Z < -2.0 | Deep Value | Historically rare buying opportunity |
| -2.0 ≤ Z < -1.0 | Value | Undervalued, increase buying |
| -1.0 ≤ Z < 1.5 | Neutral | Fair value, normal DCA |
| 1.5 ≤ Z < 2.5 | Caution | Overvalued, reduce buying |
| Z ≥ 2.5 | Danger | Extreme overvaluation, minimize buying |

### MVRV Gradient

The gradient detects MVRV trend direction over a 30-day window with EMA smoothing:

```python
gradient_raw = mvrv_zscore.diff(30)
mvrv_gradient = tanh(gradient_raw.ewm(span=30).mean() * 2)
# Result in [-1, 1]
```

| Value | Interpretation |
|-------|----------------|
| `> 0` | MVRV rising (moving toward overvaluation) |
| `< 0` | MVRV falling (moving toward undervaluation) |
| `≈ 0` | Neutral/stable |

### MVRV Acceleration

Second derivative for momentum detection:

```python
accel_raw = mvrv_gradient.diff(14)
mvrv_acceleration = tanh(accel_raw.ewm(span=14).mean() * 3)
# Result in [-1, 1]
```

### MVRV Volatility

Rolling volatility percentile for uncertainty detection:

```python
def compute_mvrv_volatility(mvrv_zscore, window=90):
    # Compute rolling standard deviation
    vol = mvrv_zscore.rolling(window, min_periods=window//4).std()

    # Compute percentile rank over 4x longer window (360 days)
    # For each point, what percentile is current volatility in historical distribution?
    vol_pct = vol.rolling(window * 4, min_periods=window).apply(
        lambda x: (x.iloc[-1] > x[:-1]).sum() / max(len(x) - 1, 1)
        if len(x) > 1 else 0.5,
        raw=False,
    )

    return vol_pct.fillna(0.5)
# Result in [0, 1] where 1 = high volatility relative to history
```

### Signal Confidence

Computes confidence based on signal agreement and gradient alignment:

```python
def compute_signal_confidence(mvrv_zscore, mvrv_gradient, price_vs_ma):
    # Normalize all signals to [-1, 1] where negative = buy signal
    z_signal = -mvrv_zscore / 4  # Scale to [-1, 1]
    ma_signal = -price_vs_ma     # Below MA = buy signal

    # Gradient alignment: does gradient confirm the buy/sell direction?
    # Buy signals + rising gradient = confirmation (1.0)
    # Buy signals + falling gradient = divergence (0.5)
    gradient_alignment = np.where(
        z_signal < 0,  # Buy signal from Z-score
        np.where(mvrv_gradient > 0, 1.0, 0.5),  # Rising = confirmation
        np.where(mvrv_gradient < 0, 1.0, 0.5),  # Falling = confirmation for sell
    )

    # Agreement: how much do Z-score and MA signals agree?
    signals = np.stack([z_signal, ma_signal], axis=0)
    signal_std = signals.std(axis=0)
    max_std = 1.0  # Maximum possible std when signals fully disagree
    agreement = 1.0 - np.clip(signal_std / max_std, 0, 1)

    # Combine: 70% agreement, 30% gradient alignment
    confidence = agreement * 0.7 + gradient_alignment * 0.3

    return np.clip(confidence, 0, 1)
```

### Look-Ahead Bias Prevention

All features are **lagged by 1 day** to prevent information leakage:

```python
features = features.shift(1).fillna(defaults)
```

This ensures the weight for day `t` only uses information available up to day `t-1`.

## Dynamic Multiplier Computation

### Asymmetric Extreme Boost

Bitcoin's MVRV is asymmetric - extreme lows are rare opportunities, while extreme highs often precede corrections:

```python
def compute_asymmetric_extreme_boost(mvrv_zscore):
    # Deep value (Z < -2): Strong positive boost
    # Quadratic: 0.8 * (Z + 2)² + 0.5
    
    # Value zone (-2 ≤ Z < -1): Linear positive boost
    # -0.5 * Z
    
    # Neutral (-1 ≤ Z < 1.5): No boost
    # 0
    
    # Caution (1.5 ≤ Z < 2.5): Moderate negative boost
    # -0.3 * (Z - 1.5)
    
    # Danger (Z ≥ 2.5): Strong negative boost
    # Quadratic: -0.5 * (Z - 2.5)² - 0.3
```

### Adaptive Trend Modifier

Thresholds adapt based on current MVRV level:

```python
def compute_adaptive_trend_modifier(mvrv_gradient, mvrv_zscore):
    # In value zone (Z < -1): Low threshold (0.1) - more sensitive
    # In danger zone (Z > 1.5): High threshold (0.4) - require confirmation
    # Neutral: Standard threshold (0.2)
    
    # Bull (gradient > threshold): up to 1.5x multiplier
    # Bear (gradient < -threshold): down to 0.3x multiplier
```

### Final Multiplier

```python
def compute_dynamic_multiplier(...):
    # 1. Value signal with asymmetric boost
    value_signal = -mvrv_zscore + extreme_boost
    
    # 2. MA signal with adaptive trend modulation
    ma_signal = -price_vs_ma * trend_modifier
    
    # 3. Combine with weights
    combined = value_signal * 0.80 + ma_signal * 0.20
    
    # 4. Apply modifiers
    combined *= acceleration_modifier  # [0.85, 1.15]
    combined *= confidence_boost       # [1.0, 1.15] only if confidence > 0.7
    combined *= volatility_dampening   # [0.8, 1.0] only if volatility > 0.8
    
    # 5. Scale and exponentiate
    adjustment = clip(combined * DYNAMIC_STRENGTH, -5, 100)
    return exp(adjustment)
```

## Sequential Allocation

The `allocate_sequential_stable` function ensures all constraints are met:

```python
def allocate_sequential_stable(raw, n_past, locked_weights=None):
    n_total = len(raw)
    base_weight = 1.0 / n_total
    
    # Past days: use signal-based weights
    for i in range(n_past):
        signal = compute_stable_signal_weights(raw[:i+1])[-1]
        w[i] = signal * base_weight
    
    # Future days: uniform distribution
    w[n_past:] = base_weight
    
    # Last day absorbs remainder to ensure sum = 1.0
    w[-1] = 1.0 - sum(w[:-1])
    
    return w
```

## Weight Computation Functions

### compute_weights_fast

Core weight computation for a date range using all enhanced features:

```python
def compute_weights_fast(features_df, start_date, end_date) -> pd.Series:
    # Extract all features: price_vs_ma, mvrv_zscore, mvrv_gradient,
    #   mvrv_acceleration, mvrv_volatility, signal_confidence
    # Compute uniform base PDF
    # Apply dynamic multiplier with all signal layers
    # Return allocated weights
```

### compute_window_weights

Wrapper that handles the past/future weight split for production use:

```python
def compute_window_weights(features_df, start_date, end_date, current_date) -> pd.Series:
    # 1. Extend features with placeholders for future dates
    # 2. Compute FULL range weights
    # 3. Split at current_date boundary:
    #    - Past/current dates: signal-based weights (LOCKED)
    #    - Future dates: uniform weights for remaining budget
    # 4. Return combined weights summing to 1.0
```

**Key Invariant**: As `current_date` advances, more weights get locked in and the remaining budget for future dates decreases.

## Constants Reference

| Constant | Value | Description |
|----------|-------|-------------|
| MIN_W | 1e-6 | Minimum weight floor |
| PRICE_COL | "PriceUSD_coinmetrics" | Price column name |
| MVRV_COL | "CapMVRVCur" | MVRV column from CoinMetrics |
| MA_WINDOW | 200 | 200-day moving average window |
| MVRV_GRADIENT_WINDOW | 30 | Window for MVRV gradient (EMA smoothed) |
| MVRV_ROLLING_WINDOW | 365 | Window for MVRV Z-score normalization |
| MVRV_ACCEL_WINDOW | 14 | Window for acceleration calculation |
| MVRV_VOLATILITY_WINDOW | 90 | Window for volatility calculation |
| DYNAMIC_STRENGTH | 5.0 | Weight adjustment multiplier |
| MVRV_VOLATILITY_DAMPENING | 0.2 | Max dampening in extreme volatility |

### MVRV Zone Thresholds

| Zone | Z-Score Threshold | Action |
|------|-------------------|--------|
| Deep Value | Z < -2.0 | Aggressive accumulation |
| Value | -2.0 ≤ Z < -1.0 | Increased buying |
| Neutral | -1.0 ≤ Z < 1.5 | Normal DCA |
| Caution | 1.5 ≤ Z < 2.5 | Reduced buying |
| Danger | Z ≥ 2.5 | Minimized buying |

## Feature List

All features computed by `precompute_features()`:

| Feature | Range | Description |
|---------|-------|-------------|
| `price_vs_ma` | [-1, 1] | Normalized distance from 200-day MA |
| `mvrv_zscore` | [-4, 4] | MVRV Z-score (365-day window) |
| `mvrv_gradient` | [-1, 1] | Smoothed MVRV trend direction |
| `mvrv_acceleration` | [-1, 1] | Second derivative of gradient |
| `mvrv_zone` | {-2,-1,0,1,2} | Discrete zone classification |
| `mvrv_volatility` | [0, 1] | Rolling volatility percentile |
| `signal_confidence` | [0, 1] | Signal agreement score |

## Data Requirements

The model requires the following columns from the CoinMetrics BTC data:

| Column | Description | Required |
|--------|-------------|----------|
| `PriceUSD_coinmetrics` | Daily BTC price in USD | Yes |
| `CapMVRVCur` | Market Value to Realized Value ratio | Optional* |

*If MVRV data is not available, the model falls back to neutral signals (no dynamic adjustment).

## Improvements Over a Simple 200-Day MA Model

A simple 200-day moving average strategy typically generates binary signals: buy when price is below the MA, reduce exposure when above. While effective as a trend filter, this approach has significant limitations that our enhanced model addresses.

### Limitations of Simple 200-Day MA

| Limitation | Description |
|------------|-------------|
| Binary signals | Only considers above/below MA, ignoring magnitude |
| No valuation context | Doesn't account for whether Bitcoin is fundamentally over/undervalued |
| Whipsaw prone | Frequent false signals when price oscillates around the MA |
| No cycle awareness | Ignores Bitcoin's 4-year halving cycle dynamics |
| Static thresholds | Same response regardless of market regime |
| No confidence weighting | Treats all signals with equal conviction |

### How This Model Improves

#### 1. Multi-Signal Integration

Instead of relying solely on price vs. MA, the model combines two weighted signals:

```
combined = MVRV_value × 0.80 + MA_signal × 0.20
```

The 200-day MA contributes 20% of the final signal, acting as a secondary confirmation rather than the primary driver. The MVRV Z-score (80% weight) provides fundamental valuation context that the MA cannot.

#### 2. Continuous vs. Binary Response

**Simple MA Model:**
```python
signal = 1 if price < ma_200 else 0  # Binary
```

**Enhanced Model:**
```python
price_vs_ma = (price / ma_200) - 1  # Continuous in [-1, 1]
ma_signal = -price_vs_ma * trend_modifier  # Scaled and modulated
```

The enhanced model responds proportionally to *how far* price is from the MA, not just whether it's above or below.

#### 3. Adaptive Trend Modulation

The MA signal is dynamically adjusted based on MVRV regime:

| MVRV Zone | Trend Threshold | Effect |
|-----------|-----------------|--------|
| Value (Z < -1) | 0.1 (low) | More sensitive to reversals—catch bottoms earlier |
| Neutral | 0.2 (standard) | Normal responsiveness |
| Danger (Z > 1.5) | 0.4 (high) | Require stronger confirmation—avoid false signals at tops |

This means the model becomes more aggressive when MVRV indicates undervaluation, even if price is near the MA.

#### 4. Asymmetric Extreme Response

A simple MA model treats a 10% deviation above the MA the same as 10% below. Our model recognizes Bitcoin's asymmetric behavior:

```python
# Deep value (Z < -2): Quadratic positive boost
boost = 0.8 * (Z + 2)² + 0.5  # Aggressive accumulation

# Danger zone (Z ≥ 2.5): Quadratic negative boost  
boost = -0.5 * (Z - 2.5)² - 0.3  # Strong reduction
```

Extreme lows are rare opportunities; extreme highs often precede corrections.

#### 5. Momentum and Acceleration Detection

The simple MA is a lagging indicator with no momentum awareness. Our model adds:

- **MVRV Gradient**: 30-day EMA-smoothed trend direction
- **MVRV Acceleration**: Second derivative for momentum detection

```python
accel_modifier = [0.85, 1.15]  # Amplify when momentum builds, dampen at reversals
```

#### 6. Signal Confidence Weighting

When multiple signals agree, the model increases conviction:

```python
# High confidence (> 0.7): boost up to 1.15×
confidence_boost = 1.0 + 0.15 * (confidence - 0.7) / 0.3
```

A simple MA model has no concept of signal agreement or confidence.

#### 7. Volatility-Aware Dampening

During high uncertainty periods (volatility > 80th percentile), the model reduces signal strength:

```python
volatility_dampening = 1.0 - 0.2 * (volatility - 0.8) / 0.2  # Down to 0.8×
```

This prevents overreaction during chaotic market conditions where the MA generates unreliable signals.

### Comparative Summary

| Aspect | Simple 200-Day MA | Enhanced Model |
|--------|-------------------|----------------|
| Signal type | Binary (above/below) | Continuous with magnitude |
| Valuation awareness | None | MVRV Z-score (80% weight) |
| Trend sensitivity | Fixed | Adaptive to MVRV regime |
| Extreme response | Linear/symmetric | Asymmetric quadratic boost |
| Momentum detection | None | Gradient + acceleration |
| Confidence weighting | None | Signal agreement scoring |
| Volatility handling | None | Dampening in high uncertainty |
| Look-ahead protection | Often violated | 1-day lag on all features |

### When the Simple MA Still Wins

The simple MA model has advantages in specific scenarios:

1. **Simplicity**: Easier to understand, audit, and explain
2. **Robustness**: Fewer parameters means less overfitting risk
3. **No data dependencies**: Works with price data alone (no MVRV required)
4. **Faster computation**: Single rolling mean vs. multiple feature calculations

Our model falls back to MA-only behavior when MVRV data is unavailable, ensuring graceful degradation.