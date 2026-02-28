# RFQ Simulator Implementation Plan (Final v3)

## Executive Summary

Build a discrete-event simulator for alpha-driven RFQ liquidity provision per `sim_spec_v2.pdf`. Core question: **Does spread income exceed alpha lost from slower convergence?**

**Key insight:** Separate WORLD (exogenous processes) from AGENT (strategy logic) from ACCOUNTING (output).

---

## Architecture

```
src/rfq_simulator/
├── config.py              # SimConfig dataclass (70+ params, all defaults from spec)
├── world/
│   ├── clock.py           # TimeGrid: minute↔step conversion, trading day logic
│   ├── price.py           # ABM + mean reversion + intraday vol + adverse move mutation
│   ├── regime.py          # 2-state Markov chain (calm/stressed)
│   ├── rfq_stream.py      # Inhomogeneous Poisson + RFQ attributes
│   ├── street_lean.py     # OU process for aggregate dealer lean
│   └── competitors.py     # Explicit N-1 dealer quote simulation
├── agent/
│   ├── alpha.py           # Alpha signal: IC, regime modulation, decay, refresh
│   ├── target.py          # α_rem → q* via risk aversion scaling
│   ├── observable.py      # Lagged mid + skew correction
│   ├── lean.py            # Lean with urgency, convexity, soft limit penalty
│   ├── winrate.py         # Trader's ESTIMATED logistic win-rate (may be miscalibrated)
│   ├── quoting.py         # Grid search optimization: m* = argmax P(win)*(edge + ΔV)
│   └── exit.py            # Hybrid exit: patient RFQ → aggressive cross
├── core/
│   ├── state.py           # SimulationState: q, cash, signal state, regime, etc.
│   ├── accounting.py      # P&L decomposition: alpha + spread + carry + hedge - aggress
│   └── hedging.py         # CDS/ETF/Treasury overlay (optional)
├── simulation/
│   ├── event_loop.py      # Main simulation: process RFQs, update state
│   ├── baseline.py        # Naive aggressor: same q*, immediate execution
│   └── batch.py           # Monte Carlo runner + scenario sweeps
└── output/
    └── diagnostics.py     # Plots, summary stats, RFQ log DataFrame
```

---

## Event Model

### Event Types & Timing

| Event | Timing | Triggered By | Updates |
|-------|--------|--------------|---------|
| Price step | Every 5 min | Clock | price_path[i] |
| RFQ arrival | Continuous (Poisson) | rfq_stream | Triggers quote/fill logic |
| Alpha refresh | Every 24h at market open | Clock | α, t_signal |
| Regime transition | Daily | Markov chain | r_t (calm/stressed) |
| Dealer bias refresh | Daily | Clock | dealer_biases[] |
| Hedge rebalance | End of day | Clock | hedge positions |
| Exit mode check | Each RFQ | Time relative to t_signal + H | exit_mode flag |

### Main Loop Pseudocode

```
1. Generate WORLD upfront:
   - price_path = generate_price_path(cfg, rng)
   - regime_path = generate_regime_path(cfg, rng)
   - rfq_events = generate_rfq_stream(cfg, rng)  # sorted by time
   - street_lean_path = generate_street_lean(cfg, rng)

2. Initialize STATE:
   - q = 0, cash = 0
   - α = None, t_signal = 0
   - dealer_biases = draw_initial_biases()

3. FOR each rfq in rfq_events:
   a. Update time-dependent state:
      - If new day: refresh dealer_biases, check regime transition
      - If signal_refresh_due: generate new α from price_path lookahead

   b. Compute agent decisions:
      - α_rem = decay(α, t, t_signal, H)
      - q* = compute_target(α_rem, cfg)
      - mid_obs = observe_mid(price_path, t, lag, noise)
      - skew = compute_skew(price_path, mid_obs, t)
      - lean = compute_lean(q, q*, t, t_signal, H, cfg)
      - theo = mid_obs + skew + lean

   c. Check exit mode:
      - If t >= t_signal + H - Δt_aggress AND q != 0:
        Execute aggressive exit, skip RFQ processing

   d. Compute optimal quote:
      - P_hat(win|m) = logistic model (trader's estimate)
      - m* = grid_search(maximize P_hat * (edge + ΔV))
      - Apply bounds (alpha, position)
      - If no feasible m: decline RFQ

   e. Simulate competitors:
      - competitor_quotes = simulate_competitors(p_true, rfq, street_lean, biases)
      - filled = (our_quote beats best_competitor)

   f. If filled:
      - Update q, cash
      - Apply adverse move to price_path (mutation!)
      - Record spread_pnl

   g. Log RFQ event

4. Run BASELINE on same price_path:
   - Same α signals, same q* targets
   - Execute immediately when q* changes
   - Pay aggressor costs (no RFQ interaction)

5. Compute final P&L decomposition for both strategies
```

---

## Critical Implementation Details

### 1. Price Path with Mutation (Eq 1-3, 25)

**Generation:**
```
p_{t+dt} = p_t + κ(p̄ - p_t)dt + σ(h)√dt ε_t
σ(h) = σ_base * (1 + v_open*e^{-(h-h0)²/τ²} + v_close*e^{-(h-hc)²/τ²})
```

**Mutation on fill:**
```
p_{t+} = p_t + direction * τ_c * σ_adverse * |Z|
```
Applied as permanent level shift to all future prices.

### 2. Alpha Signal (Eq 4-7)

**At refresh time t_signal:**
```
α* = p_{t_signal + H} - p_{t_signal}  # Perfect foresight
α = ρ(r) * α* + √(1-ρ²) * σ_α * η
```
where ρ(r) = IC if calm, IC * IC_stress_mult if stressed.

**Decay:**
```
α_rem(t) = α * max(0, (t_signal + H - t) / H)
```

### 3. Competitor Model (Eq 16-20) — HIGHEST PRIORITY

**Dealer quote:**
```
p_j = p_true + b_j + m_j(z) + ε_j

b_j ~ N(b̄_street, σ_b)           # Daily bias from street lean
m_j(z) = m̄ + m_N*(N-N̄) + m_size*log(size) + m_tox*τ_c + u_j
ε_j ~ N(0, σ_q)                   # Quote noise
```

**Participation:**
```
P(respond) = clip(r̄ - r_size*log(size) - r_tox*τ_c, 0.3, 1.0)
```

**Fill resolution:**
- Client buy (offers): trader wins if quote < min(competitor_quotes)
- Client sell (bids): trader wins if quote > max(competitor_quotes)

### 4. Optimal Quoting (Eq 21-24)

**Objective:**
```
m* = argmax_m P̂(win|m) * [edge(m) + ΔV]

edge(m) = m * direction
ΔV = +α_rem*|Δq|           if toward q*
    = -γσ²|q-q*||Δq|/q_max  if away from q*
```

**Bounds:**
- Alpha: edge + ΔV >= -|α_rem|*|Δq|
- Position: |q + Δq| <= q_max

**Method:** Grid search, m ∈ [-30, +30] bps at 0.1 bps resolution.

### 5. Hybrid Exit (Eq 26-27)

**Phase 1 (patient):** t < t_signal + H - Δt_aggress
- Lean naturally reverses as q* → 0
- Earn spread on unwinding fills

**Phase 2 (aggressive):** t >= t_signal + H - Δt_aggress
- Unwind at constant rate over remaining time
- Cost per lot: c_aggress + c_impact * √|q|

### 6. P&L Decomposition (Eq 35-40)

```
PnL_total = PnL_alpha + PnL_spread + PnL_carry + PnL_hedge - Cost_aggress

PnL_alpha = Σ q_{t-1} * (p_t - p_{t-1}) * lot_size
PnL_spread = Σ (p_fill - p_true) * signed_size * lot_size
PnL_carry = Σ_days q * lot_size * coupon_bps / 360
PnL_hedge = Σ h * Δp_hedge - hedge_txn_costs
Cost_aggress = Σ spread_paid + impact_cost
```

---

## Validation Checkpoints

| Checkpoint | Validation | Pass Criteria |
|------------|------------|---------------|
| After price.py | Plot 100 paths | Vol matches 20-80 bps (IG) or 50-200 bps (HY) |
| After regime.py | Check stationary dist | ~25% stress with default params |
| After competitor.py | 10,000 RFQ sweep | Win-rate curve is sigmoid, center 3-8 bps (IG) |
| After simulator.py | Single path | Inventory converges toward q* |
| After baseline.py | Compare LP vs baseline | LP earns spread, baseline has higher ACR |
| After batch.py | 500 paths | Sharpe confidence interval |

---

## Load-Bearing Assumptions

1. **Price mutation:** Adverse moves permanently shift price level (not temporary)
2. **Time indexing:** RFQ at minute t uses price at step floor(t/dt)
3. **Competitor independence:** Dealers quote independently given street lean
4. **Baseline isolation:** Baseline doesn't interact with RFQs, doesn't cause adverse moves
5. **Win-rate estimation:** Trader's model can be miscalibrated (winrate_est_error param)

---

## RNG Management

Use `numpy.random.Generator` with explicit streams:
```python
rng = np.random.default_rng(seed)
price_rng = np.random.default_rng(rng.integers(2**32))
rfq_rng = np.random.default_rng(rng.integers(2**32))
competitor_rng = np.random.default_rng(rng.integers(2**32))
```

This ensures reproducibility even if number of RFQs varies.

---

## Build Order

| # | Module | Description | Test |
|---|--------|-------------|------|
| 1 | config.py | All 70+ params with spec defaults | - |
| 2 | world/clock.py | TimeGrid utilities | Unit |
| 3 | world/price.py | ABM + intraday vol + mutation | Plot vol |
| 4 | world/regime.py | Markov chain | Stationary dist |
| 5 | agent/alpha.py | IC, decay, refresh | Decay curve |
| 6 | agent/target.py | α → q* | Trivial |
| 7 | world/rfq_stream.py | Poisson thinning + attributes | Rate check |
| 8 | world/competitors.py | **CRITICAL** dealer quotes | Win-rate curve |
| 9 | agent/winrate.py | Logistic estimate | - |
| 10 | agent/quoting.py | Grid search | Boundary cases |
| 11 | agent/observable.py | Lagged mid + skew | - |
| 12 | agent/lean.py | Urgency, convexity, soft limit | - |
| 13 | core/state.py | SimulationState | - |
| 14 | core/accounting.py | P&L tracking | - |
| 15 | simulation/event_loop.py | **First runnable sim** | Single path |
| 16 | simulation/baseline.py | Naive aggressor | Compare |
| 17 | agent/exit.py | Hybrid exit logic | - |
| 18 | world/street_lean.py | OU + proxies | - |
| 19 | core/hedging.py | Optional overlay | - |
| 20 | output/diagnostics.py | Plots, stats | - |
| 21 | simulation/batch.py | Monte Carlo | 500 paths |
| 22 | notebooks/rfq_simulator.ipynb | Final assembly | Full run |

---

## Notebook Structure

```
# Alpha-Driven RFQ Simulator

## 1. Setup
- Imports, SimConfig, parameter overrides

## 2. Calibration Validation
- 10,000 RFQ sweep at varying markups
- Plot empirical win-rate curve vs expected

## 3. Single Path Demo
- Time series: price, inventory, P&L
- RFQ event log

## 4. LP vs Baseline
- Side-by-side metrics table
- Overlaid cumulative P&L

## 5. Sensitivity Analysis
- Lambda, IC, RFQ rate sweeps
- Heatmaps

## 6. Monte Carlo
- 500 paths, P&L distribution
- Sharpe, ACR, drawdown histograms

## 7. Scenario Sweep
- Full grid over priority dimensions
- Identify crossover where baseline dominates
```

---

## Decision Summary

| Aspect | Decision |
|--------|----------|
| **Architecture** | World/Agent/Accounting separation |
| **Critical path** | config → clock → price → competitor → event_loop |
| **Key risk** | Competitor model calibration |
| **Validation** | Win-rate curve sweep before proceeding |
| **Trade-off** | More modules = more files, but testable |
| **Deferred** | Multi-name portfolio (architecture supports it) |
