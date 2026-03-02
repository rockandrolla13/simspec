# RFQ Simulator Domain Reference

**Version**: 0.1.0
**Tests**: 110 passing
**Architecture Health**: Strong

---

## 1. Project Purpose

The RFQ Simulator is a Monte Carlo simulation engine for **corporate bond market-making** via the Request-for-Quote (RFQ) protocol. It models a dealer who uses an **alpha signal** (predictive signal about future price moves) to optimally price quotes against a pool of simulated competitors.

**Primary use cases:**
- Strategy backtesting and parameter optimization
- P&L decomposition and attribution analysis
- Risk/reward tradeoff exploration
- Understanding market microstructure dynamics

---

## 2. Domain Concepts

### 2.1 Request-for-Quote (RFQ) Protocol

In corporate bond markets, trading occurs via RFQ rather than limit order books:

1. **Client** sends RFQ to multiple dealers specifying: direction (buy/sell), size, and bond
2. **Dealers** respond with executable prices (or decline)
3. **Client** selects best price and executes (winner-take-all auction)
4. **Filled dealer** now has inventory to manage

```
Client → RFQ(buy 5MM ABC Corp 5.5% 2030) → [Dealer1, Dealer2, Dealer3, ...]
         ←── Price quotes ←──────────────── [99.25,   99.20,   DECLINE, ...]
         → Execute with Dealer1 (best price)
```

### 2.2 Alpha Signal

The dealer has a **predictive signal** about future price movements:

- **IC (Information Coefficient)**: Correlation between signal and realized returns (typical: 0.03-0.15)
- **Horizon**: How far ahead the signal predicts (typical: 5-20 days)
- **Regime-dependent**: IC degrades in stressed markets

When the signal predicts the bond will appreciate, the dealer wants to accumulate inventory (lean bid). When it predicts depreciation, the dealer wants to reduce inventory (lean ask).

### 2.3 Inventory Management

The dealer balances:

- **Alpha capture**: Build positions aligned with signal
- **Risk limits**: Stay within position bounds (q_max)
- **Urgency**: Unwind before signal expires
- **Transaction costs**: Aggressive exit costs spread + impact

### 2.4 Competitor Modeling

Other dealers quote with:

- **Base markup**: Compensation for adverse selection and inventory risk
- **Size adjustment**: Wider spreads for larger trades
- **Toxicity adjustment**: Wider spreads for informed flow
- **Response probability**: May decline to quote (especially large/toxic RFQs)

### 2.5 Market Regimes

Two-state Markov process:

| Regime | Characteristics |
|--------|----------------|
| **CALM** | Normal IC, tighter spreads, balanced flow |
| **STRESSED** | Degraded IC, wider spreads, sell-biased flow |

Transition probabilities create ~25% time in stressed state at equilibrium.

---

## 3. Stochastic Processes

### 3.1 Price Process (Ornstein-Uhlenbeck)

```
dp = κ(p̄ - p)dt + σ·dW
```

- **Mean-reverting** around `p̄` (long-run price)
- **κ**: Mean-reversion speed (per day)
- **σ**: Daily volatility in bps
- Includes **intraday seasonality** (higher vol at open/close)

### 3.2 RFQ Arrivals (Hawkes Process)

```
λ(t) = μ(t) + Σ α·exp(-β·(t - t_i))
```

**Self-exciting**: Each RFQ temporarily increases arrival intensity (clustering).

| Parameter | Meaning | Typical |
|-----------|---------|---------|
| `α` | Excitation magnitude | 0.3-0.6 |
| `β` | Decay rate | 0.1-0.5 per minute |
| `α/β` | Branching ratio (must be < 1) | 0.3-0.7 |

**Enable with**: `arrivals=ArrivalConfig(use_hawkes=True)`

### 3.3 Spread Distribution (Log-Normal)

```
log(spread) ~ N(μ_regime, σ_regime)
```

- **Log-normal** ensures positive spreads
- **Regime-switching**: μ_stressed > μ_calm (wider spreads in stress)
- **Size adjustment**: Concave increase with trade size

| Parameter | Calm | Stressed |
|-----------|------|----------|
| μ (log-mean) | 2.0 (~7 bps median) | 3.2 (~25 bps median) |
| σ (log-std) | 0.5 | 0.8 |

**Enable with**: `spreads=SpreadConfig(use_lognormal=True)`

### 3.4 Buy/Sell Imbalance (AR(1) Process)

```
z_t = ρ·z_{t-1} + √(1-ρ²)·ε_t
P(buy) = clip(0.5 + z_t + μ_regime, 0.2, 0.8)
```

- **Autocorrelated**: Flow direction persists
- **Regime-dependent mean**: Sell bias in stress (`μ_stressed < 0`)

| Parameter | Meaning | Typical |
|-----------|---------|---------|
| `ρ` | Persistence | 0.3-0.5 |
| `μ_calm` | Calm bias | 0.0 (balanced) |
| `μ_stressed` | Stressed bias | -0.25 (sell bias) |

**Enable with**: `imbalance=ImbalanceConfig(use_ar1=True)`

### 3.5 Street Lean (Ornstein-Uhlenbeck)

Aggregate dealer inventory position:

```
db̄ = θ(b̄_eq - b̄)dt + σ_b·dW
```

- **Mean-reverting** to equilibrium (usually 0)
- **Half-life**: ln(2)/θ days
- Observable via noisy proxies (bid-ask asymmetry, flow imbalance, ETF premium)

---

## 4. Architecture

```
src/rfq_simulator/
├── config.py              # SimConfig dataclass (all parameters)
├── world/                 # Exogenous market processes
│   ├── clock.py           # Trading calendar and time
│   ├── price.py           # OU price process
│   ├── regime.py          # Calm/Stressed Markov chain
│   ├── hawkes.py          # Hawkes arrival process
│   ├── imbalance.py       # AR(1) flow imbalance
│   ├── spread.py          # Log-normal spread distribution
│   ├── street_lean.py     # OU street lean process
│   ├── rfq_stream.py      # RFQ event generation
│   └── competitors.py     # Dealer quote simulation
├── agent/                 # Strategy decisions
│   ├── alpha.py           # Alpha signal computation
│   ├── target.py          # Position target calculation
│   ├── observable.py      # Observed mid and skew
│   ├── lean.py            # Quote lean computation
│   ├── winrate.py         # Win probability estimation
│   ├── quoting.py         # Optimal markup selection
│   └── exit.py            # Hybrid patient/aggressive exit
├── core/                  # State and accounting
│   ├── state.py           # SimulationState dataclass
│   └── accounting.py      # P&L decomposition
├── simulation/            # Event loop and runners
│   ├── event_loop.py      # Main simulation loop
│   ├── baseline.py        # Baseline strategy comparison
│   └── batch.py           # Monte Carlo batch runs
└── output/                # Diagnostics and visualization
    ├── diagnostics.py     # Basic plots and summaries
    ├── realistic_diagnostics.py  # Statistical validation
    └── narrative.py       # Report formatting
```

### Dependency Direction

```
config ← world ← agent ← core ← simulation ← output
```

Each layer depends only on layers to its left. No circular dependencies.

---

## 5. Key Modules

### 5.1 `config.py` — SimConfig

Single dataclass with ~80 parameters organized by spec section:

| Section | Parameters | Example |
|---------|------------|---------|
| Simulation Control | T_days, dt_minutes, seed | T_days=60 |
| Price Process | p0, sigma_daily_bps, kappa_daily | sigma_daily_bps=50 |
| Alpha Signal | IC, alpha_horizon_days | IC=0.10 |
| RFQ Arrivals | rfq_rate_per_day, A_open, A_close | rfq_rate_per_day=15 |
| Competitor Model | markup_base_bps, n_dealers_mean | markup_base_bps=8 |
| Street Lean | street_lean_vol_bps, street_lean_mean_rev | street_lean_vol_bps=2 |

Sub-configs for realistic distributions:
- `ArrivalConfig` — Hawkes process parameters
- `SpreadConfig` — Log-normal spread parameters
- `ImbalanceConfig` — AR(1) imbalance parameters

### 5.2 `simulation/event_loop.py` — run_simulation()

Main entry point:

```python
from rfq_simulator import SimConfig, run_simulation

cfg = SimConfig(T_days=60, IC=0.10)
result = run_simulation(cfg, verbose=True)
print(result.summary())
```

Returns `SimulationResult` with:
- `rfq_log`: List of all RFQ events with full details
- `pnl_series`: Time series of cumulative P&L
- `position_series`: Time series of inventory
- `price_path`, `regime_path`, `street_lean_path`: Market state histories
- `summary()`: Text summary of key metrics

### 5.3 `output/realistic_diagnostics.py` — ValidationReport

Statistical validation of simulation outputs:

```python
from rfq_simulator.output import ValidationReport

report = ValidationReport(result)
report.display()  # In Jupyter
report.to_html("report.html")  # Export
```

**Diagnostics included:**

| Diagnostic | Validates | Key Stats |
|------------|-----------|-----------|
| HawkesDiagnostics | RFQ arrival clustering | ACF(1), branching ratio |
| SpreadDiagnostics | Log-normal spread shape | Shapiro-Wilk p-value, regime ratio |
| ImbalanceDiagnostics | AR(1) flow persistence | Direction ACF, regime buy fraction |
| StreetLeanDiagnostics | OU process calibration | Vol ratio, half-life |

---

## 6. P&L Decomposition

The simulator decomposes realized P&L into components:

| Component | Formula | Meaning |
|-----------|---------|---------|
| **Spread P&L** | Σ(markup × filled_size) | Edge captured on winning quotes |
| **Alpha P&L** | Σ(position × price_change) | Gain from directional bets |
| **Adverse P&L** | Σ(adverse_move × filled_size) | Loss from immediate post-fill moves |
| **Exit P&L** | Σ(exit_cost) | Cost of aggressive unwinding |

**Sharpe Ratio** computed from daily P&L series.

---

## 7. Entry Points

### Basic Simulation
```python
from rfq_simulator import SimConfig, run_simulation

cfg = SimConfig(T_days=60, IC=0.10)
result = run_simulation(cfg)
```

### With Realistic Distributions
```python
from rfq_simulator import SimConfig
from rfq_simulator.config import ArrivalConfig, SpreadConfig, ImbalanceConfig

cfg = SimConfig(
    T_days=60,
    arrivals=ArrivalConfig(use_hawkes=True, hawkes_alpha=0.4),
    spreads=SpreadConfig(use_lognormal=True),
    imbalance=ImbalanceConfig(use_ar1=True, mu_stressed=-0.25),
)
result = run_simulation(cfg)
```

### Batch Monte Carlo
```python
from rfq_simulator import run_batch

batch = run_batch(cfg, n_paths=500)
print(f"Mean Sharpe: {batch.mean_sharpe:.2f}")
print(f"Sharpe CI: [{batch.sharpe_ci[0]:.2f}, {batch.sharpe_ci[1]:.2f}]")
```

### Baseline Comparison
```python
from rfq_simulator import run_baseline, compare_strategies

baseline = run_baseline(cfg)  # No alpha, symmetric quotes
compare_strategies(result, baseline)
```

### Validation Report
```python
from rfq_simulator.output import ValidationReport

report = ValidationReport(result)
results = report.run_all(generate_plots=True)
print(report.summary())
```

---

## 8. Configuration Reference

### Simulation Control
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| T_days | 60 | 20-250 | Trading days to simulate |
| dt_minutes | 5.0 | 1-15 | Price update interval |
| n_mc_paths | 500 | 100-10000 | Monte Carlo paths for batch |
| seed | 42 | any | Random seed |

### Alpha Signal
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| IC | 0.10 | 0.03-0.15 | Information coefficient |
| IC_stress_mult | 0.4 | 0.2-0.8 | IC multiplier in stress |
| alpha_horizon_days | 10 | 5-20 | Signal forecast horizon |

### RFQ Arrivals
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| rfq_rate_per_day | 15 | 2-40 | Mean daily RFQ count |
| use_hawkes | False | — | Enable Hawkes clustering |
| hawkes_alpha | 0.4 | 0.2-0.6 | Excitation magnitude |
| hawkes_beta | 0.8 | 0.4-1.2 | Decay rate |

### Spreads
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| markup_base_bps | 8 | 3-40 | Base dealer markup |
| use_lognormal | False | — | Enable log-normal spreads |
| mu_calm | 2.0 | 1.5-2.5 | Log-mean (calm) |
| mu_stressed | 3.2 | 2.5-4.0 | Log-mean (stressed) |

### Street Lean
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| street_lean_vol_bps | 2.0 | 1-5 | OU volatility |
| street_lean_mean_rev | 0.1 | 0.05-0.3 | Mean-reversion θ |
| street_lean_eq | 0.0 | -5 to 5 | Equilibrium level |

---

## 9. Testing

**110 tests** covering:

| Test File | Count | Coverage |
|-----------|-------|----------|
| test_config.py | 8 | Configuration validation |
| test_simulation.py | 15 | End-to-end simulation |
| test_hawkes.py | 12 | Hawkes process |
| test_spread.py | 10 | Log-normal spreads |
| test_imbalance.py | 10 | AR(1) imbalance |
| test_street_lean.py | 10 | Street lean OU process |
| test_realistic_diagnostics.py | 24 | Diagnostic validation |
| test_diagnostics_integration.py | 4 | Full report integration |
| test_competitors.py | 12 | Competitor modeling |
| test_quoting.py | 5 | Optimal quoting |

Run tests:
```bash
PYTHONPATH=src pytest tests/ -v
```

---

## 10. Glossary

| Term | Definition |
|------|------------|
| **RFQ** | Request-for-Quote — client solicits prices from multiple dealers |
| **IC** | Information Coefficient — correlation between signal and realized returns |
| **bps** | Basis points — 1 bps = 0.01% |
| **Hawkes process** | Self-exciting point process where events increase future intensity |
| **OU process** | Ornstein-Uhlenbeck mean-reverting stochastic process |
| **Branching ratio** | α/β for Hawkes — must be < 1 for stationarity |
| **Half-life** | Time for OU process to decay halfway to equilibrium: ln(2)/θ |
| **Toxicity** | Probability that RFQ is from informed trader |
| **Lean** | Quote adjustment to favor one side based on inventory/alpha |
| **Street lean** | Aggregate dealer inventory position across market |
| **Regime** | Market state (CALM or STRESSED) affecting parameters |

---

## 11. References

### Academic
1. Hawkes, A.G. (1971) — "Spectra of Some Self-Exciting and Mutually Exciting Point Processes"
2. Bacry, E. et al. (2015) — "Hawkes Processes in Finance"
3. Bao, J. et al. (2011) — "The Illiquidity of Corporate Bonds"
4. Edwards, A. et al. (2007) — "Corporate Bond Market Transaction Costs and Transparency"

### Project Documents
- `sim_spec_v2.pdf` — Primary specification
- `docs/plans/` — Design documents
- `reviews/` — Architecture reviews

---

*Last updated: 2026-03-02*
