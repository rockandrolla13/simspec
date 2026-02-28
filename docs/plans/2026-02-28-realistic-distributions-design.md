# Realistic Market Distributions Design

**Date**: 2026-02-28
**Status**: Draft
**Goal**: Replace simplified stochastic processes with empirically-grounded distributions for RFQ arrivals, spreads, and buy/sell imbalance.

---

## 1. RFQ Arrival Process

### Current Implementation
- Poisson process with constant rate `λ = rfq_rate_per_day / minutes_per_day`
- No clustering, no intraday seasonality

### Proposed: Hawkes Process with Intraday Seasonality

#### References
1. **Hawkes, A.G. (1971)** - "Spectra of Some Self-Exciting and Mutually Exciting Point Processes" - *Biometrika*
2. **Bacry, E., Mastromatteo, I., Muzy, J.F. (2015)** - "Hawkes Processes in Finance" - *Market Microstructure and Liquidity*
3. **Hendershott, T., Seasholes, M. (2007)** - "Market Maker Inventories and Stock Prices" - *American Economic Review*
4. **Fleming, M., Remolona, E. (1999)** - "Price Formation and Liquidity in the U.S. Treasury Market" - *Journal of Finance*

#### Model Specification

**Intensity function**:
```
λ(t) = μ(t) + Σ_{t_i < t} α · exp(-β · (t - t_i))
```

Where:
- `μ(t)` = baseline intensity with intraday seasonality
- `α` = excitation magnitude (how much each RFQ increases future intensity)
- `β` = decay rate (how quickly excitation fades)
- `t_i` = times of previous RFQs

**Intraday seasonality** (U-shaped pattern):
```
μ(t) = μ_base · S(hour)

S(hour) = {
    1.8   if hour in [9, 10]      # Morning surge
    1.2   if hour in [10, 11]     # Late morning
    0.7   if hour in [12, 13]     # Lunch lull
    1.0   if hour in [13, 15]     # Afternoon normal
    1.4   if hour in [15, 16]     # End-of-day pickup
    0.3   if hour in [16, 17]     # After-hours trickle
}
```

#### Parameters from Literature

| Parameter | Value | Source |
|-----------|-------|--------|
| `α` (excitation) | 0.3 - 0.6 | Bacry et al. (2015), calibrated to equity order flow |
| `β` (decay, per minute) | 0.1 - 0.5 | Half-life of 1-7 minutes |
| `α/β` (branching ratio) | 0.3 - 0.7 | Must be < 1 for stationarity |
| Morning multiplier | 1.5 - 2.0× | Hendershott & Seasholes (2007) |
| Lunch multiplier | 0.5 - 0.8× | Fleming & Remolona (1999) |

#### Implementation Notes
- Use Ogata's thinning algorithm for simulation
- Reset Hawkes state at start of each trading day
- Add event-driven spikes (Fed announcements, earnings) as additional impulses

---

## 2. Spread Distribution

### Current Implementation
- Fixed base markup + linear adjustments for size/toxicity
- Approximately normal distribution

### Proposed: Log-Normal Mixture with Regime Switching

#### References
1. **Bao, J., Pan, J., Wang, J. (2011)** - "The Illiquidity of Corporate Bonds" - *Journal of Finance*
2. **Dick-Nielsen, J., Feldhütter, P., Lando, D. (2012)** - "Corporate Bond Liquidity Before and After the Crisis" - *Journal of Financial Economics*
3. **Friewald, N., Jankowitsch, R., Subrahmanyam, M. (2012)** - "Illiquidity or Credit Deterioration" - *Review of Financial Studies*
4. **Edwards, A., Harris, L., Piwowar, M. (2007)** - "Corporate Bond Market Transaction Costs and Transparency" - *Journal of Finance*

#### Model Specification

**Base spread (log-normal)**:
```
log(spread) ~ Normal(μ_regime, σ_regime)
spread = exp(log(spread))
```

**Size adjustment** (concave, per Edwards et al.):
```
spread_adj = spread_base × (1 + γ × log(1 + size/size_ref))
```

**Full model**:
```
spread = exp(μ + σ·Z) × (1 + γ·log(1 + size)) × (1 + τ·toxicity)

where Z ~ Normal(0, 1)
```

#### Parameters from Literature

| Parameter | Calm Regime | Stressed Regime | Source |
|-----------|-------------|-----------------|--------|
| `μ` (log-spread mean) | 2.0 (≈7 bps) | 3.2 (≈25 bps) | Dick-Nielsen et al. (2012) |
| `σ` (log-spread std) | 0.5 | 0.8 | Bao et al. (2011) |
| Size coefficient `γ` | 0.15 | 0.25 | Edwards et al. (2007) |
| Toxicity coefficient `τ` | 0.3 | 0.5 | Friewald et al. (2012) |

**Spread ranges by credit quality** (median, bps):

| Rating | Calm | Stressed | Source |
|--------|------|----------|--------|
| IG (A/BBB) | 5-15 | 20-50 | Dick-Nielsen et al. (2012) |
| HY (BB) | 15-40 | 50-150 | Friewald et al. (2012) |
| HY (B/CCC) | 40-100 | 150-400 | Bao et al. (2011) |

#### Implementation Notes
- Sample from log-normal, not normal
- Clip extreme values at [1, 500] bps for numerical stability
- Consider mixture of two log-normals for bimodal stressed markets

---

## 3. Buy/Sell Imbalance

### Current Implementation
- `is_client_buy ~ Bernoulli(0.5)` - symmetric, IID

### Proposed: Autoregressive Process with Regime-Dependent Mean

#### References
1. **Chordia, T., Roll, R., Subrahmanyam, A. (2002)** - "Order Imbalance, Liquidity, and Market Returns" - *Journal of Financial Economics*
2. **O'Hara, M., Zhou, X. (2021)** - "Anatomy of a Liquidity Crisis" - *Journal of Financial Economics*
3. **Bessembinder, H., Jacobsen, S., Maxwell, W., Venkataraman, K. (2018)** - "Capital Commitment and Illiquidity in Corporate Bonds" - *Journal of Finance*
4. **Goldstein, M., Hotchkiss, E. (2020)** - "Providing Liquidity in an Illiquid Market" - *Journal of Financial Economics*

#### Model Specification

**Latent imbalance process** (continuous):
```
imbalance_t = ρ · imbalance_{t-1} + (1 - ρ) · μ_regime + σ_imb · ε_t

where ε_t ~ Normal(0, 1)
```

**Buy probability** (from latent):
```
p_buy = Φ(imbalance_t)  # Standard normal CDF

or simpler:
p_buy = clip(0.5 + imbalance_t, 0.2, 0.8)
```

**Regime-dependent mean**:
```
μ_calm = 0.0      # Balanced flow
μ_stressed = -0.3  # Sell bias (risk-off)
```

#### Parameters from Literature

| Parameter | Value | Source |
|-----------|-------|--------|
| `ρ` (autocorrelation) | 0.3 - 0.5 | Chordia et al. (2002) |
| `σ_imb` (innovation std) | 0.2 - 0.4 | Calibrated |
| `μ_calm` | -0.05 to 0.05 | Slight structural sell bias |
| `μ_stressed` | -0.2 to -0.4 | O'Hara & Zhou (2021) |
| Buy fraction (calm) | 45-55% | Bessembinder et al. (2018) |
| Buy fraction (stressed) | 30-45% | O'Hara & Zhou (2021) |

**Daily patterns**:
- Morning: slight buy bias (portfolio rebalancing)
- Afternoon: slight sell bias (end-of-day de-risking)
- Month-end: buy bias (index rebalancing)

#### Implementation Notes
- Update imbalance state at each RFQ arrival
- Reset with persistence across days (not fully IID)
- Consider correlation with price moves (informed flow)

---

## 4. Configuration Parameters

New parameters to add to `SimConfig`:

```python
@dataclass
class SimConfig:
    # === Existing ===
    ...

    # === RFQ Arrival (Hawkes) ===
    hawkes_alpha: float = 0.4           # Excitation magnitude
    hawkes_beta: float = 0.2            # Decay rate (per minute)
    intraday_seasonality: bool = True   # Enable U-shaped pattern

    # === Spread Distribution ===
    spread_dist: str = "lognormal"      # "normal" | "lognormal" | "mixture"
    spread_log_mu_calm: float = 2.0     # ~7 bps median
    spread_log_mu_stress: float = 3.2   # ~25 bps median
    spread_log_sigma_calm: float = 0.5
    spread_log_sigma_stress: float = 0.8
    spread_size_gamma: float = 0.15     # Size impact coefficient

    # === Buy/Sell Imbalance ===
    imbalance_rho: float = 0.4          # AR(1) coefficient
    imbalance_sigma: float = 0.3        # Innovation std
    imbalance_mu_calm: float = 0.0      # Balanced
    imbalance_mu_stress: float = -0.25  # Sell bias
```

---

## 5. Module Structure

### New Files

```
src/rfq_simulator/world/
├── arrivals.py          # Hawkes process + seasonality
├── spread_dist.py       # Log-normal spread sampling
└── imbalance.py         # AR(1) buy/sell imbalance
```

### Modified Files

```
src/rfq_simulator/
├── config.py            # Add new parameters
├── world/
│   └── rfq_stream.py    # Use new arrival/imbalance processes
└── world/
    └── competitors.py   # Use new spread distribution
```

---

## 6. Validation Tests

### Statistical Tests to Add

1. **RFQ arrivals**:
   - Test clustering: inter-arrival times should show positive autocorrelation
   - Test seasonality: morning count > lunch count
   - Test stationarity: no drift in daily totals

2. **Spread distribution**:
   - Test log-normality: Shapiro-Wilk on log(spreads)
   - Test regime shift: mean(stressed) > 2× mean(calm)
   - Test fat tails: kurtosis > 3

3. **Buy/sell imbalance**:
   - Test autocorrelation: lag-1 ACF ≈ ρ
   - Test regime mean: stressed period has lower buy fraction
   - Test stationarity: imbalance mean-reverts

---

## 7. References (Full Citations)

### RFQ Arrivals
- Hawkes, A.G. (1971). "Spectra of Some Self-Exciting and Mutually Exciting Point Processes." *Biometrika*, 58(1), 83-90.
- Bacry, E., Mastromatteo, I., & Muzy, J.F. (2015). "Hawkes Processes in Finance." *Market Microstructure and Liquidity*, 1(1).

### Spread Distributions
- Bao, J., Pan, J., & Wang, J. (2011). "The Illiquidity of Corporate Bonds." *Journal of Finance*, 66(3), 911-946.
- Dick-Nielsen, J., Feldhütter, P., & Lando, D. (2012). "Corporate Bond Liquidity Before and After the Onset of the Subprime Crisis." *Journal of Financial Economics*, 103(3), 471-492.
- Edwards, A., Harris, L., & Piwowar, M. (2007). "Corporate Bond Market Transaction Costs and Transparency." *Journal of Finance*, 62(3), 1421-1451.

### Buy/Sell Imbalance
- Chordia, T., Roll, R., & Subrahmanyam, A. (2002). "Order Imbalance, Liquidity, and Market Returns." *Journal of Financial Economics*, 65(1), 111-130.
- O'Hara, M., & Zhou, X. (2021). "Anatomy of a Liquidity Crisis: Corporate Bonds in the COVID-19 Crisis." *Journal of Financial Economics*, 142(1), 46-68.
- Bessembinder, H., Jacobsen, S., Maxwell, W., & Venkataraman, K. (2018). "Capital Commitment and Illiquidity in Corporate Bonds." *Journal of Finance*, 73(4), 1615-1661.

---

## 8. Success Criteria

1. **RFQ arrivals**: Inter-arrival time ACF(1) > 0.1 (clustering detected)
2. **Spreads**: Log-spread distribution passes normality test (p > 0.05)
3. **Imbalance**: Buy fraction in stressed regime < 45%
4. **Overall**: Simulated market "looks right" to practitioner inspection
