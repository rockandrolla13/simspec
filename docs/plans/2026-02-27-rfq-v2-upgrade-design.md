# RFQ Simulator V2 Upgrade — Implementation Plan

**Date:** 2026-02-27
**Status:** Final (3 iterations complete)
**Spec Source:** `sim_spec.pdf` (V2, 20 pages)

---

## Executive Summary

Upgrade the existing ~5,500 LOC v1 implementation to v2 spec. Critical changes:

1. **Explicit competitor model** — replaces parametric logistic with dealer-by-dealer simulation
2. **Regime-dependent IC** — Markov chain modulating signal quality
3. **Multi-event architecture** — unified event loop handling RFQs, day transitions, signal refreshes
4. **Price path forking** — clean baseline comparison without adverse move contamination

**Estimated scope:** ~2,000 new/modified lines across 12 files.

---

## Key Assumptions Made

| Question | Assumption |
|----------|------------|
| Full rewrite vs incremental? | **Incremental** — existing code is solid |
| Price mutation semantics? | **Permanent level shift** — mutate `price_path[t:]` |
| Baseline isolation? | **Forked path** — baseline uses unmutated original |
| Street lean proxies? | **Simplified for V2.0** — noisy observation of true value |
| Primary goal? | **Parameter calibration + sensitivity analysis** |

---

## Architecture Changes

### A1. Event-Driven Architecture

**Current:** Sequential RFQ processing
**Target:** Priority queue with multiple event types

```
EventType:
├── DAY_START (t=0, 480, 960, ...) — regime, dealer bias, hedge rebalance
├── SIGNAL_REFRESH (t=0, 1440, 2880, ...) — new alpha generation
├── RFQ (t=random) — quote optimization, fill resolution
├── PRICE_STEP (t=0, 5, 10, ...) — for aggressor execution only
└── AGGRESS_CHECK (continuous) — exit mode transition
```

### A2. Price Path Management

```
┌─────────────────────────────────────────────────┐
│ generate_price_path(cfg, rng) → base_path       │
└─────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│ LP Simulation │       │ Baseline Run  │
│ (mutates copy)│       │ (uses base)   │
└───────────────┘       └───────────────┘
```

### A3. Config Restructuring

```python
@dataclass
class SimConfig:
    # Core (existing)
    T_days: int = 60
    dt_minutes: float = 5.0
    ...

    # NEW: Nested configs
    regime: RegimeConfig
    competitor: CompetitorConfig
    seasonality: SeasonalityConfig
    exit: ExitConfig
    street_lean: StreetLeanConfig
    hedge: HedgeConfig
```

---

## Implementation Phases

### PHASE 0: Foundation (Blocking)

| Module | Change | LOC Est |
|--------|--------|---------|
| `config.py` | Add nested dataclasses for 6 config groups | +150 |
| `core/events.py` | NEW: EventType enum, Event dataclass, PriorityQueue wrapper | +80 |
| `simulation/event_loop.py` | Refactor to multi-event dispatch | +200 (rewrite) |

**Validation:** Event loop runs with existing functionality intact.

---

### PHASE 1: Core V2 Components (Critical Path)

#### 1.1 Regime System

| File | Change |
|------|--------|
| `world/regime.py` | Add Markov chain: `generate_regime_path(T_days, p_cs, p_sc, rng)` |
| `agent/alpha.py` | Modulate IC: `ρ = IC * (IC_stress_mult if stressed else 1.0)` |

**Validation:** Run 10,000 days, verify π_stress ≈ p_cs/(p_cs + p_sc) ≈ 0.25

#### 1.2 Intraday Seasonality

| File | Change |
|------|--------|
| `world/price.py` | Add σ(h) = σ_base * (1 + v_open*exp(-...) + v_close*exp(-...)) |
| `world/rfq_stream.py` | Thinning algorithm for μ(h) with open/close bumps |

**Validation:** Plot hourly arrival rate, verify U-shape.

#### 1.3 Competitor Model Overhaul (HIGHEST PRIORITY)

| File | Change |
|------|--------|
| `world/competitors.py` | **Major rewrite** |

New structure:

```python
class DealerPool:
    def __init__(cfg, rng):
        self.biases = None  # refreshed daily

    def refresh_biases(street_lean: float):
        """Called at DAY_START"""
        self.biases = rng.normal(street_lean, cfg.dealer_bias_std_bps, N_max)

    def simulate_quotes(rfq, p_true) -> List[float]:
        """Generate N-1 competitor quotes for this RFQ"""
        # Participation filter
        # Markup: m̄ + m_N*(N-N̄) + m_size*log(size) + m_tox*τ_c
        # Quote = p_true + bias + markup + noise

    def resolve_fill(trader_quote, competitor_quotes, direction) -> bool:
        """Compare trader vs best competitor"""
        if direction == BUY:  # client buys, we offer
            return trader_quote < min(competitor_quotes)
        else:  # client sells, we bid
            return trader_quote > max(competitor_quotes)
```

**Validation:** 10,000 RFQ sweep at varying markups → plot win-rate curve → verify sigmoid with center ~8bps, steepness ~0.8 for IG defaults.

#### 1.4 Soft Limit Penalty

| File | Change |
|------|--------|
| `agent/lean.py` | Add quadratic penalty term near q_max |

```
φ = (1 + κ_c * |q - q*|) * (1 + κ_lim * max(0, |q|/q_max - θ_lim)²)
```

**Validation:** Verify lean doubles at 80% capacity, quadruples at 90%.

---

### PHASE 2: Street Lean & Exit

#### 2.1 Street Lean Process

| File | Change |
|------|--------|
| `world/street_lean.py` | OU process: `generate_street_lean_path(T_days, θ, σ, b_eq, rng)` |

Feeds into DealerPool.refresh_biases() as the mean.

#### 2.2 Street Lean Observation (Simplified V2.0)

```
b̂_t = b̄_t + σ_obs * ξ_t
```

Full proxy model (bid-ask asymmetry, flow imbalance, ETF premium) deferred to V2.1.

#### 2.3 Hybrid Exit Logic

| File | Change |
|------|--------|
| `agent/exit.py` | Phase detection + aggressor execution |

```python
if t >= t_sig + H - Δt_aggress:
    enter_aggressor_mode()

def aggressor_step(t, remaining_time, q):
    lots_this_step = ceil(|q| / remaining_steps)
    cost = (c_aggress + c_impact * sqrt(|q|)) * lots_this_step
    execute(lots_this_step, cost)
```

**Validation:** Verify aggressor cost tracked separately in P&L decomposition.

---

### PHASE 3: Baseline & Comparison

#### 3.1 Price Path Forking

| File | Change |
|------|--------|
| `simulation/event_loop.py` | `run_simulation()` returns mutated path |
| `simulation/baseline.py` | Uses original unmutated path |

```python
def run_comparison(cfg):
    base_path = generate_price_path(cfg, rng)
    regime_path = generate_regime_path(cfg, rng)
    alpha_signals = generate_alpha_signals(base_path, regime_path, cfg, rng)

    # LP simulation (mutates working copy)
    lp_result = run_simulation(
        prices=base_path.copy(),  # COPY
        regime_path=regime_path,
        alpha_signals=alpha_signals,
        cfg=cfg,
    )

    # Baseline (uses original)
    baseline_result = run_baseline(
        prices=base_path,  # ORIGINAL
        regime_path=regime_path,
        alpha_signals=alpha_signals,
        cfg=cfg,
    )

    return compare(lp_result, baseline_result)
```

#### 3.2 Enhanced P&L Decomposition

| File | Change |
|------|--------|
| `core/accounting.py` | Add `aggressor_cost` as separate line item |

```
PnL_total = PnL_alpha + PnL_spread + PnL_carry + PnL_hedge - Cost_aggress
```

---

### PHASE 4: Validation & Diagnostics

#### 4.1 Testing Suite

| Test | Purpose |
|------|---------|
| `test_regime.py` | Stationary distribution, transition counts |
| `test_competitors.py` | Win-rate curve shape, fill resolution logic |
| `test_seasonality.py` | Arrival rate distribution, vol profile |
| `test_event_loop.py` | Deterministic single-path replay |
| `test_baseline_isolation.py` | Verify no adverse move contamination |

#### 4.2 Diagnostic Enhancements

| File | Change |
|------|--------|
| `output/diagnostics.py` | Regime shading, street lean overlay, aggressor window marking |

New plots:
1. Price panel with regime background shading (gray = stressed)
2. Street lean: true b̄_t vs estimated b̂_t
3. P&L breakdown with aggressor cost as red bar
4. Win-rate calibration: estimated vs realized

#### 4.3 Scenario Sweep

Priority dimensions per spec:

| Dimension | Values | Question |
|-----------|--------|----------|
| `lambda_base_bps` | [0.5, 1, 2, 4, 8] | Optimal lean aggressiveness? |
| `IC` | [0.03, 0.05, 0.10, 0.15] | Minimum viable alpha? |
| `rfq_rate_per_day` | [3, 8, 15, 30] | Illiquid name viability? |
| `markup_base_bps` | [3, 8, 15, 30] | Dealer competitiveness effect? |
| `aggress_window_hours` | [0, 4, 8, 16] | Exit patience value? |
| `flow_bias` | [-0.2, 0, +0.2] | One-sided flow impact? |

---

## Deferred to V2.1

1. **Full street lean proxy model** — bid-ask asymmetry, flow imbalance, ETF premium
2. **Hedging overlay** — CDS/ETF/Treasury with basis risk
3. **Multi-name portfolio** — architecture supports it, not implemented
4. **Continuous signal updates** — currently daily refresh only

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| Competitor model calibration drift | Validate win-rate curve before every batch run |
| Event ordering edge cases | Deterministic tie-breaking by event type priority |
| RNG stream contamination | Explicit child RNG per process, seeded from master |
| Baseline unfair comparison | Fork price path before LP mutation |
| Config proliferation | Nested dataclasses with sensible defaults |

---

## Build Order (Dependency-Aware)

```
1.  config.py (nested configs)           ← No deps
2.  core/events.py                       ← No deps
3.  world/regime.py                      ← config
4.  world/price.py (seasonality)         ← config
5.  world/rfq_stream.py (seasonality)    ← config
6.  world/street_lean.py                 ← config
7.  world/competitors.py (OVERHAUL)      ← config, street_lean
8.  agent/alpha.py (regime modulation)   ← regime
9.  agent/lean.py (soft limit)           ← config
10. agent/exit.py (hybrid)               ← config
11. simulation/event_loop.py (REWRITE)   ← events, all world/agent
12. simulation/baseline.py (forking)     ← event_loop
13. core/accounting.py (aggress cost)    ← No deps
14. output/diagnostics.py (enhanced)     ← All above
15. tests/*                              ← All above
```

---

## New Parameters (V2 Additions)

### RegimeConfig
- `IC_stress_mult` (float, 0.4) — IC degradation in stressed regime
- `p_calm_to_stress` (float, 0.05) — daily transition probability
- `p_stress_to_calm` (float, 0.15) — daily transition probability

### SeasonalityConfig
- `v_open` (float, 0.5) — open vol multiplier
- `v_close` (float, 0.3) — close vol multiplier
- `tau_v_hours` (float, 0.75) — vol seasonality width
- `A_open` (float, 0.8) — open RFQ activity multiplier
- `A_close` (float, 0.5) — close RFQ activity multiplier
- `tau_f_hours` (float, 0.75) — RFQ seasonality width

### CompetitorConfig
- `dealer_bias_std_bps` (float, 3.0) — cross-dealer lean dispersion
- `markup_base_bps` (float, 8.0) — baseline dealer markup
- `markup_N_bps` (float, -1.0) — markup shift per dealer
- `markup_size_bps` (float, 2.0) — shift per log-lot
- `markup_tox_bps` (float, 10.0) — shift per unit toxicity
- `markup_noise_bps` (float, 3.0) — per-quote markup noise
- `quote_noise_bps` (float, 2.0) — dealer model error
- `respond_base` (float, 0.85) — base response rate
- `respond_size` (float, 0.05) — size impact on response
- `respond_tox` (float, 0.1) — toxicity impact on response
- `winrate_est_error` (float, 0.0) — trader estimation degradation

### ExitConfig
- `aggress_window_hours` (float, 8.0) — time before expiry to start aggressing
- `aggress_halfspread_bps` (float, 8.0) — half-spread cost
- `aggress_impact_bps` (float, 2.0) — impact per sqrt(lot)

### StreetLeanConfig
- `street_lean_mr` (float, 0.1) — mean-reversion θ_b
- `street_lean_vol_bps` (float, 2.0) — vol σ_b
- `street_lean_eq` (float, 0.0) — equilibrium b̄_eq
- `street_obs_noise` (float, 0.5) — observation noise

### LeanConfig (additions)
- `kappa_limit` (float, 10.0) — soft limit penalty strength
- `theta_limit` (float, 0.7) — fraction of q_max where penalty activates

### HedgeConfig (V2.1)
- `hedge_cds_on`, `hedge_etf_on`, `hedge_tsy_on` (bool, False)
- `rho_cds`, `rho_etf`, `rho_tsy` (float)
- `cds_cost_bps`, `etf_cost_bps`, `tsy_cost_bps` (float)

---

## Iteration History

| Round | Focus | Key Changes |
|-------|-------|-------------|
| 1 | Scope & structure | 4-phase plan, identified competitor model as critical |
| 2 | Architecture depth | Added Phase 0, multi-event system, RNG streams, simplified street lean |
| 3 | Consolidation | Nested configs, test suite, build order, risk register, deferred items |

---

**End of Plan**
