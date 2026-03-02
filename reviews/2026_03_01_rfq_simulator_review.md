# Code Review Report

**Files reviewed:**
- `src/rfq_simulator/__init__.py`
- `src/rfq_simulator/config.py`
- `src/rfq_simulator/core/__init__.py`
- `src/rfq_simulator/core/protocols.py`
- `src/rfq_simulator/core/state.py`
- `src/rfq_simulator/core/accounting.py`
- `src/rfq_simulator/world/__init__.py`
- `src/rfq_simulator/world/clock.py`
- `src/rfq_simulator/world/price.py`
- `src/rfq_simulator/world/regime.py`
- `src/rfq_simulator/world/rfq_stream.py`
- `src/rfq_simulator/world/competitors.py`
- `src/rfq_simulator/world/spread.py`
- `src/rfq_simulator/world/street_lean.py`
- `src/rfq_simulator/world/hawkes.py`
- `src/rfq_simulator/world/imbalance.py`
- `src/rfq_simulator/agent/__init__.py`
- `src/rfq_simulator/agent/alpha.py`
- `src/rfq_simulator/agent/target.py`
- `src/rfq_simulator/agent/observable.py`
- `src/rfq_simulator/agent/lean.py`
- `src/rfq_simulator/agent/quoting.py`
- `src/rfq_simulator/agent/winrate.py`
- `src/rfq_simulator/agent/exit.py`
- `src/rfq_simulator/simulation/__init__.py`
- `src/rfq_simulator/simulation/event_loop.py`
- `src/rfq_simulator/simulation/baseline.py`
- `src/rfq_simulator/simulation/batch.py`
- `src/rfq_simulator/output/__init__.py`
- `src/rfq_simulator/output/diagnostics.py`
- `src/rfq_simulator/output/narrative.py`
- `src/rfq_simulator/output/realistic_diagnostics.py`

**Date:** 2026-03-01
**Overall health:** :yellow_circle: Needs attention

## Executive Summary

The codebase is well-structured with clear module separation matching the spec (world/agent/core/simulation/output). Naming is consistent, docstrings reference spec equations, and randomness is seeded throughout. However, there are two likely bugs: a duplicate `ExitMode` class that shadows an import in the event loop, and an `is not None` guard on a field typed as `float` (always true). The `compute_street_lean_impact` function has identical branches, negating its directional logic. Several modules carry dead imports and legacy `typing` generics that Python 3.11+ no longer needs. Addressing the shadowed import and the street lean symmetry bug should be the top priority.

## Findings

### CR-BUG-001: Duplicate ExitMode class shadows import in event_loop
- **Severity:** :red_circle: Critical
- **Pillar:** Correctness
- **Location:** `core/state.py:L23-L28`, `agent/exit.py:L28-L33`, `simulation/event_loop.py:L26,L32`

BEFORE:
```python
# simulation/event_loop.py
from ..agent.exit import HybridExitManager, ExitMode
...
from ..core.state import SimulationState, RFQLog, ExitMode, create_initial_state
```

AFTER:
```python
# Remove ExitMode from one of the two imports, and delete
# the duplicate class definition. Keep it in core/state.py
# since SimulationState depends on it.
from ..agent.exit import HybridExitManager
from ..core.state import SimulationState, RFQLog, ExitMode, create_initial_state
```

WHY:
The second import silently shadows the first `ExitMode`. Both classes define identical members, so this happens to work today, but any future divergence between the two definitions would cause silent, hard-to-trace bugs. The canonical location should be `core/state.py` (where `SimulationState` uses it); `agent/exit.py` should import from there.

---

### CR-BUG-002: `t_signal is not None` guard is always True
- **Severity:** :orange_circle: Major
- **Pillar:** Correctness
- **Location:** `simulation/event_loop.py:L206`

BEFORE:
```python
if state.t_signal is not None and alpha_manager.should_refresh(current_minute):
    exit_manager.reset()
```

AFTER:
```python
# t_signal is typed as float (default 0.0) -- it is never None.
# Guard should check whether a signal has actually been generated:
if alpha_manager.current_signal is not None and alpha_manager.should_refresh(current_minute):
    exit_manager.reset()
```

WHY:
`SimulationState.t_signal` is declared as `float = 0.0`. The `is not None` check never fails, so the exit manager may be reset before any signal has been generated, which is a latent logic error. The intent is clearly to check whether a signal exists.

---

### CR-BUG-003: `compute_street_lean_impact` returns identical values for both directions
- **Severity:** :orange_circle: Major
- **Pillar:** Correctness
- **Location:** `world/street_lean.py:L175-L204`

BEFORE:
```python
def compute_street_lean_impact(street_lean_bps, is_client_buy):
    if is_client_buy:
        return -street_lean_bps * 0.5
    else:
        return -street_lean_bps * 0.5
```

AFTER:
```python
def compute_street_lean_impact(street_lean_bps, is_client_buy):
    # The sign convention should differ by side.
    # Positive lean + offer -> lower; positive lean + bid -> lower too
    # If this is intentional (symmetric pass-through), collapse to one return.
    return -street_lean_bps * 0.5
```

WHY:
The branch on `is_client_buy` is dead code -- both arms return the same expression. Either the sign should differ between offer and bid sides (making one `-0.5` and the other `+0.5`), or the branch should be eliminated entirely. As written, the function misleads the reader into thinking directionality is accounted for. Note: this function is currently unused (no callers found in the codebase), which limits immediate impact, but if integrated it would produce incorrect directional effects.

---

### CR-BUG-004: `q_before` reconstruction in `_create_rfq_log` uses post-update state
- **Severity:** :orange_circle: Major
- **Pillar:** Correctness
- **Location:** `simulation/event_loop.py:L480`

BEFORE:
```python
q_before=state.q - ((-1 if rfq.is_client_buy else 1) * rfq.size if filled else 0),
```

AFTER:
```python
# Capture q_before BEFORE calling state.update_position(),
# then pass the saved value into _create_rfq_log.
q_before = state.q
# ... state.update_position(delta_q, quote_result.quote_price, rfq.size)
# ... _create_rfq_log(..., q_before=q_before, ...)
```

WHY:
The current approach reconstructs `q_before` by subtracting the fill from the already-updated `state.q`. The inversion formula uses `(-1 if is_client_buy else 1) * size`, but `update_position` uses `delta_q * size` where `delta_q` is `-1 or +1`. The reconstruction works if the size factor aligns exactly, but it is fragile: any change to how `update_position` handles sizing (e.g., partial fills) would silently break `q_before`. Capturing the value before mutation is simpler and correct by construction.

---

### CR-STYLE-005: Legacy `typing` imports on Python 3.11+ codebase
- **Severity:** :yellow_circle: Minor
- **Pillar:** Style / Conventions
- **Location:** `config.py:L8`, `world/clock.py:L12`, `world/spread.py:L17`, `world/regime.py:L18`, `world/competitors.py:L20`, `world/street_lean.py:L18`, `world/rfq_stream.py:L20`, `world/hawkes.py:L17`, `core/state.py:L14`, `core/accounting.py:L15`, `agent/exit.py:L21`, `agent/quoting.py:L17`, `simulation/event_loop.py:L12`, `simulation/baseline.py:L18`, `simulation/batch.py:L11`, `output/diagnostics.py:L11`, `output/realistic_diagnostics.py:L9`

BEFORE:
```python
from typing import List, Tuple, Optional, Dict, Any
```

AFTER:
```python
# Python 3.11+ supports built-in generics:
# list[int], tuple[float, float], dict[str, Any], X | None
```

WHY:
CLAUDE.md specifies Python 3.11+. Using `typing.List`, `typing.Tuple`, etc. is unnecessary on 3.11+ where native generics (`list`, `tuple`, `dict`) suffice. The codebase already uses `float | None` (PEP 604 syntax) in protocols.py, hawkes.py, imbalance.py, and street_lean.py, creating an inconsistency with files that still use `Optional[float]`.

---

### CR-STYLE-006: Unused import `Callable` in batch.py
- **Severity:** :yellow_circle: Minor
- **Pillar:** Style / Conciseness
- **Location:** `simulation/batch.py:L11`

BEFORE:
```python
from typing import List, Dict, Any, Optional, Callable
```

AFTER:
```python
from typing import List, Dict, Any, Optional
```

WHY:
`Callable` is imported but never used in the module.

---

### CR-DRY-007: `TimeGrid` delegates all properties to `SimConfig` without adding logic
- **Severity:** :blue_circle: Suggestion
- **Pillar:** DRY / Conciseness
- **Location:** `world/clock.py:L17-L186`

BEFORE:
```python
class TimeGrid:
    cfg: SimConfig

    @property
    def dt(self) -> float:
        return self.cfg.dt_minutes

    @property
    def minutes_per_day(self) -> float:
        return self.cfg.minutes_per_day
    # ... ~15 more delegating methods
```

AFTER:
```python
# Keep only conversion methods (minute_to_step, step_to_day, etc.)
# that perform actual computation. The pure-delegation properties
# (dt, minutes_per_day, steps_per_day, total_steps, total_minutes)
# add an indirection layer without value. Callers can use cfg directly.
```

WHY:
Five properties (`dt`, `minutes_per_day`, `steps_per_day`, `total_steps`, `total_minutes`) simply forward to `cfg` with no additional logic. The conversion methods (`minute_to_step`, `step_to_day`, etc.) earn their place, but the delegation properties add surface area without benefit.

---

### CR-DRY-008: Duplicate urgency calculation in `compute_lean` and `compute_urgency_adjusted_lean`
- **Severity:** :yellow_circle: Minor
- **Pillar:** DRY
- **Location:** `agent/lean.py:L39-L67`, `agent/exit.py:L211-L244`

BEFORE:
```python
# lean.py:L39
def compute_urgency_factor(signal_age_minutes, horizon_minutes, cfg):
    age_fraction = np.clip(signal_age_minutes / horizon_minutes, 0, 1)
    return 1.0 + cfg.kappa_urgency * age_fraction

# exit.py:L211
def compute_urgency_adjusted_lean(base_lean, current_minute, t_signal, horizon_minutes, cfg):
    time_elapsed = current_minute - t_signal
    urgency_factor = 1.0 + cfg.kappa_urgency * (time_elapsed / horizon_minutes)
    urgency_factor = min(urgency_factor, 1.0 + cfg.kappa_urgency)
    return base_lean * urgency_factor
```

AFTER:
```python
# exit.py should call lean.compute_urgency_factor rather than
# reimplementing the same formula with slightly different clamping.
```

WHY:
The urgency factor formula is implemented twice with subtle differences: `lean.py` uses `np.clip(age/H, 0, 1)` while `exit.py` uses `min(factor, 1+kappa)`. Both achieve the same capping but diverge in edge case behavior. Consolidating to a single canonical computation prevents future drift.

---

### CR-PERF-009: Grid search in `compute_optimal_quote` iterates ~600 markups per RFQ
- **Severity:** :blue_circle: Suggestion
- **Pillar:** Performance
- **Location:** `agent/quoting.py:L224-L256`

BEFORE:
```python
markups = np.arange(-cfg.m_max_bps, cfg.m_max_bps + cfg.m_grid_bps, cfg.m_grid_bps)
# With defaults: arange(-30, 30.1, 0.1) = 601 points
for m in markups:
    obj, win_prob, edge, delta_v = compute_objective(m, rfq, ...)
```

AFTER:
```python
# The objective P(win|m) * [edge(m) + DeltaV] is unimodal for logistic P(win).
# A golden-section search or scipy.optimize.minimize_scalar would converge
# in ~15-20 evaluations instead of 600. For batch runs (500+ paths x 900+ RFQs),
# this is the main computational bottleneck.
```

WHY:
With defaults (`m_max_bps=30`, `m_grid_bps=0.1`), the grid has ~601 points. Each evaluation calls `estimate_win_probability` and `compute_continuation_value`. For 500 MC paths with ~900 RFQs each, this is ~270M function evaluations. The logistic win-probability ensures the objective is quasi-concave; a scalar optimizer would reduce evaluations by ~30x.

---

### CR-BUG-010: `PnLDecomposition.total_pnl_bps` normalization is incorrect
- **Severity:** :orange_circle: Major
- **Pillar:** Correctness
- **Location:** `core/accounting.py:L59-L65`

BEFORE:
```python
@property
def total_pnl_bps(self) -> float:
    if self.lot_size_mm <= 0:
        return 0.0
    return self.total_pnl / (self.lot_size_mm * 10000) * 10000
```

AFTER:
```python
@property
def total_pnl_bps(self) -> float:
    # Should normalize by notional = q * lot_size * price, not just lot_size.
    # As written, divides by lot_size_mm * 10000 then multiplies by 10000,
    # which simplifies to total_pnl / lot_size_mm -- losing the bps conversion.
    # Needs the actual traded notional or position for meaningful bps.
    ...
```

WHY:
The expression `total_pnl / (lot_size_mm * 10000) * 10000` simplifies algebraically to `total_pnl / lot_size_mm`. This does not produce a basis-point figure -- it divides dollars by millions-of-dollars, giving a dimensionless ratio, not bps. To get bps of notional, the denominator should include the position quantity and price (or total traded notional).

---

### CR-STYLE-011: `compute_edge` function ignores its `is_client_buy` parameter
- **Severity:** :yellow_circle: Minor
- **Pillar:** Correctness / Conciseness
- **Location:** `agent/quoting.py:L53-L72`

BEFORE:
```python
def compute_edge(markup_bps: float, is_client_buy: bool) -> float:
    # ... docstring discusses direction ...
    return markup_bps
```

AFTER:
```python
def compute_edge(markup_bps: float) -> float:
    return markup_bps
```

WHY:
The `is_client_buy` parameter is accepted but never used in the function body. The docstring explains the convention that positive markup always means earning edge, which is correct, but the unused parameter suggests directional handling was intended and forgotten.

---

### CR-BUG-012: Baseline strategy trades every price step, not just on signal refresh
- **Severity:** :orange_circle: Major
- **Pillar:** Correctness / Performance
- **Location:** `simulation/baseline.py:L134-L180`

BEFORE:
```python
for step in range(len(prices)):
    ...
    if q != q_target:
        trade_size = q_target - q
        aggress_cost = pnl_tracker.record_aggress_cost(...)
        q = q_target
        total_trades += 1
```

AFTER:
```python
# q_target changes only when alpha_remaining decays (every step) and
# on signal refresh. This means the baseline re-trades on every single
# price step (5760 steps for 60 days @ 5-min intervals) even though
# the target may have moved by a fraction of a lot. Consider a
# minimum trade size threshold or only trading on refresh events.
```

WHY:
Because `alpha_remaining` decays linearly every step, `q_target` changes fractionally at every 5-minute step, causing the baseline to execute ~5760 trades over 60 days. Each incurs aggression costs with a `sqrt(size)` impact term. This makes the baseline unrealistically expensive and inflates the LP strategy's relative advantage. The spec describes the baseline as executing "immediately when target changes" -- implying signal-refresh boundaries, not continuous rebalancing at sub-lot granularity.

---

### CR-STYLE-013: `Regime(IntEnum)` vs `Regime(Enum)` inconsistency
- **Severity:** :yellow_circle: Minor
- **Pillar:** Style
- **Location:** `world/regime.py:L17`, `core/state.py:L23`, `agent/exit.py:L28`

BEFORE:
```python
# regime.py
class Regime(IntEnum):
    CALM = 0
    STRESSED = 1

# state.py, exit.py
class ExitMode(Enum):
    PATIENT = "patient"
    AGGRESSIVE = "aggressive"
```

AFTER:
```python
# Both are fine individually, but the mix is notable.
# Regime uses IntEnum because it serves as an array index;
# ExitMode uses string Enum for readability. This is acceptable
# but should be documented.
```

WHY:
`Regime` uses `IntEnum` (allowing `regime_path` to store ints) while `ExitMode` uses `Enum` with string values. The choice is defensible but creates a stylistic inconsistency.

---

### CR-SOLID-014: `StreetLeanProcess.step()` signature diverges from `StochasticProcess` protocol
- **Severity:** :yellow_circle: Minor
- **Pillar:** SOLID (Liskov Substitution)
- **Location:** `world/street_lean.py:L225`

BEFORE:
```python
# Protocol definition (protocols.py:L17):
def step(self, dt: float = 1.0) -> float: ...

# StreetLeanProcess (street_lean.py:L225):
def step(self, dt: float | None = None) -> float:
```

AFTER:
```python
def step(self, dt: float = 1.0) -> float:
    # Use self._dt if dt == 1.0 (indicating default)
    # or accept dt directly
```

WHY:
The protocol declares `step(dt: float = 1.0)` but `StreetLeanProcess.step()` uses `dt: float | None = None`. While Python's runtime protocol checking is duck-typed, the type signatures are incompatible: a caller following the protocol interface would pass `dt=1.0` and get unexpected behavior (the process would step by 1 day instead of `self._dt`).

---

### CR-PERF-015: `_estimate_sigma_alpha` recomputes full forward-return array on every signal refresh
- **Severity:** :blue_circle: Suggestion
- **Pillar:** Performance
- **Location:** `agent/alpha.py:L174-L198`

BEFORE:
```python
def _estimate_sigma_alpha(self, prices: np.ndarray) -> float:
    horizon_steps = int(self.horizon_minutes / self.cfg.dt_minutes)
    forward_returns = prices[horizon_steps:] - prices[:-horizon_steps]
    return np.std(forward_returns)
```

AFTER:
```python
# Cache the result -- sigma_alpha depends on the full price path which
# does not change between signal refreshes (ignoring adverse moves,
# which are small perturbations). Compute once at initialization.
```

WHY:
This computes `std(prices[H:] - prices[:-H])` on every signal refresh. For a 60-day simulation with 24-hour refresh and 5760-step price path, the forward-return array has ~4800 elements computed ~60 times. Caching the result or computing it once at initialization avoids redundant work.

---

### CR-TYPE-016: `compute_theo_price` returns untyped `tuple` instead of `NamedTuple`
- **Severity:** :blue_circle: Suggestion
- **Pillar:** Types
- **Location:** `agent/observable.py:L137-L174`

BEFORE:
```python
def compute_theo_price(...) -> tuple:
    ...
    return theo, mid_obs, skew
```

AFTER:
```python
@dataclass
class TheoResult:
    theo: float
    mid_obs: float
    skew: float

def compute_theo_price(...) -> TheoResult:
    ...
```

WHY:
The bare `tuple` return type loses semantic meaning. Callers must know the positional convention (`theo, mid_obs, skew`). A named container prevents index-swap bugs and makes the API self-documenting.

---

### CR-STYLE-017: `compute_realized_volatility` uses variance scaling that conflicts with ABM process
- **Severity:** :yellow_circle: Minor
- **Pillar:** Correctness
- **Location:** `world/price.py:L147-L171`

BEFORE:
```python
def compute_realized_volatility(prices, cfg):
    log_returns = np.diff(np.log(prices))
    daily_var = np.var(log_returns) * steps_per_day
    daily_vol_bps = np.sqrt(daily_var) * 10000
```

AFTER:
```python
# The price process is ABM (arithmetic), not GBM (geometric).
# Using log returns for vol calculation introduces a model mismatch.
# For ABM, arithmetic returns (p[t+1]-p[t]) / p0 are more appropriate.
arithmetic_returns = np.diff(prices) / cfg.p0
daily_var = np.var(arithmetic_returns) * steps_per_day
daily_vol_bps = np.sqrt(daily_var) * 10000
```

WHY:
The spec defines the price process as arithmetic Brownian motion (Eq 1). Using `np.log(prices)` computes geometric returns. For prices near `p0=100` the difference is negligible, but for large deviations (HY bonds with high vol), the mismatch grows. Using arithmetic returns aligns with the process definition.

---

### CR-STYLE-018: `assert` statements in `SimConfig.validate()` can be stripped by `-O`
- **Severity:** :yellow_circle: Minor
- **Pillar:** Correctness
- **Location:** `config.py:L370-L386`

BEFORE:
```python
def validate(self) -> None:
    assert self.T_days > 0, "T_days must be positive"
    assert self.dt_minutes > 0, "dt_minutes must be positive"
    ...
```

AFTER:
```python
def validate(self) -> None:
    if self.T_days <= 0:
        raise ValueError("T_days must be positive")
    if self.dt_minutes <= 0:
        raise ValueError("dt_minutes must be positive")
    ...
```

WHY:
`assert` statements are removed when Python runs with `-O` (optimize) flag. Validation logic should use explicit `if/raise` to guarantee it always executes. The Hawkes branching ratio check at line 386 already correctly uses `raise ValueError`.

## Summary Table

| Finding ID   | Severity          | Pillar                | Location                                  | Finding                                                                 |
|------------- |-------------------|-----------------------|-------------------------------------------|-------------------------------------------------------------------------|
| CR-BUG-001   | :red_circle: Critical   | Correctness           | `core/state.py:L23`, `agent/exit.py:L28`, `simulation/event_loop.py:L26,L32` | Duplicate `ExitMode` class; second import shadows first                 |
| CR-BUG-002   | :orange_circle: Major    | Correctness           | `simulation/event_loop.py:L206`           | `t_signal is not None` guard is always True (field is `float`, not `Optional`) |
| CR-BUG-003   | :orange_circle: Major    | Correctness           | `world/street_lean.py:L175-L204`          | `compute_street_lean_impact` returns identical values for both directions |
| CR-BUG-004   | :orange_circle: Major    | Correctness           | `simulation/event_loop.py:L480`           | `q_before` reconstructed from post-update state; fragile inversion      |
| CR-STYLE-005 | :yellow_circle: Minor    | Style / Conventions   | 17 files (see details)                    | Legacy `typing` imports (`List`, `Tuple`, `Optional`) on Python 3.11+   |
| CR-STYLE-006 | :yellow_circle: Minor    | Style / Conciseness   | `simulation/batch.py:L11`                 | Unused import `Callable`                                                |
| CR-DRY-007   | :blue_circle: Suggestion | DRY / Conciseness     | `world/clock.py:L17-L186`                 | `TimeGrid` delegates 5 properties to `SimConfig` without added logic    |
| CR-DRY-008   | :yellow_circle: Minor    | DRY                   | `agent/lean.py:L39-L67`, `agent/exit.py:L211-L244` | Duplicate urgency factor calculation with subtle clamping differences   |
| CR-PERF-009  | :blue_circle: Suggestion | Performance           | `agent/quoting.py:L224-L256`              | Grid search iterates ~601 points per RFQ; unimodal objective allows scalar opt |
| CR-BUG-010   | :orange_circle: Major    | Correctness           | `core/accounting.py:L59-L65`             | `total_pnl_bps` normalization simplifies to `total_pnl / lot_size_mm` -- not bps |
| CR-STYLE-011 | :yellow_circle: Minor    | Correctness/Conciseness | `agent/quoting.py:L53-L72`              | `compute_edge` ignores its `is_client_buy` parameter                    |
| CR-BUG-012   | :orange_circle: Major    | Correctness / Perf    | `simulation/baseline.py:L134-L180`        | Baseline trades every price step; unrealistically inflated costs         |
| CR-STYLE-013 | :yellow_circle: Minor    | Style                 | `world/regime.py:L17`, `core/state.py:L23`, `agent/exit.py:L28` | `IntEnum` vs string `Enum` inconsistency across similar enum types      |
| CR-SOLID-014 | :yellow_circle: Minor    | SOLID (LSP)           | `world/street_lean.py:L225`               | `step(dt: float | None)` diverges from `StochasticProcess` protocol     |
| CR-PERF-015  | :blue_circle: Suggestion | Performance           | `agent/alpha.py:L174-L198`                | `_estimate_sigma_alpha` recomputes full forward-return array on every refresh |
| CR-TYPE-016  | :blue_circle: Suggestion | Types                 | `agent/observable.py:L137-L174`           | `compute_theo_price` returns bare `tuple`; named container preferred    |
| CR-STYLE-017 | :yellow_circle: Minor    | Correctness           | `world/price.py:L147-L171`                | Log returns used for realized vol of ABM process (model mismatch)       |
| CR-STYLE-018 | :yellow_circle: Minor    | Correctness           | `config.py:L370-L386`                     | `assert` in `validate()` stripped by `-O` flag                          |

## Positive Highlights

1. **Clear spec traceability.** Every module references the relevant spec equations in its docstring (e.g., "Implements Eq 13-15 from spec"). This makes it straightforward to verify each implementation against the spec.

2. **Well-designed RNG isolation.** The event loop creates separate RNG streams (`price_rng`, `rfq_rng`, `competitor_rng`, `agent_rng`) from a master seed, ensuring that changing one module's behavior does not perturb randomness in another. This is a best practice for reproducible simulation.

3. **Clean separation between trader beliefs and reality.** The architecture correctly distinguishes the trader's estimated win-rate model (`winrate.py` -- logistic approximation) from actual competition outcome (`competitors.py` -- explicit dealer simulation). This is the core structural insight of the simulator and it is implemented cleanly with explicit documentation of the distinction.

---

## Handoff

| Severity          | Pillar                | Location                                  | Finding                                                                 | Finding ID   |
|-------------------|-----------------------|-------------------------------------------|-------------------------------------------------------------------------|------------- |
| :red_circle: Critical   | Correctness           | `core/state.py:L23`, `agent/exit.py:L28`, `simulation/event_loop.py:L26,L32` | Duplicate `ExitMode` class; second import shadows first                 | CR-BUG-001   |
| :orange_circle: Major    | Correctness           | `simulation/event_loop.py:L206`           | `t_signal is not None` guard always True (float field, not Optional)    | CR-BUG-002   |
| :orange_circle: Major    | Correctness           | `world/street_lean.py:L175-L204`          | `compute_street_lean_impact` identical return for both directions       | CR-BUG-003   |
| :orange_circle: Major    | Correctness           | `simulation/event_loop.py:L480`           | `q_before` reconstructed from post-update state; fragile                | CR-BUG-004   |
| :orange_circle: Major    | Correctness           | `core/accounting.py:L59-L65`             | `total_pnl_bps` normalization yields `total_pnl / lot_size_mm`, not bps | CR-BUG-010   |
| :orange_circle: Major    | Correctness / Perf    | `simulation/baseline.py:L134-L180`        | Baseline trades every price step; inflated aggression costs              | CR-BUG-012   |
| :yellow_circle: Minor    | Style / Conventions   | 17 files                                  | Legacy `typing` imports on Python 3.11+ codebase                        | CR-STYLE-005 |
| :yellow_circle: Minor    | Style / Conciseness   | `simulation/batch.py:L11`                 | Unused import `Callable`                                                | CR-STYLE-006 |
| :yellow_circle: Minor    | DRY                   | `agent/lean.py`, `agent/exit.py`          | Duplicate urgency factor with subtle clamping differences               | CR-DRY-008   |
| :yellow_circle: Minor    | Correctness/Conciseness | `agent/quoting.py:L53-L72`              | `compute_edge` ignores `is_client_buy` parameter                        | CR-STYLE-011 |
| :yellow_circle: Minor    | Style                 | `world/regime.py`, `core/state.py`, `agent/exit.py` | `IntEnum` vs string `Enum` inconsistency                               | CR-STYLE-013 |
| :yellow_circle: Minor    | SOLID (LSP)           | `world/street_lean.py:L225`               | `step()` signature diverges from `StochasticProcess` protocol           | CR-SOLID-014 |
| :yellow_circle: Minor    | Correctness           | `world/price.py:L147-L171`                | Log returns for realized vol of ABM process                             | CR-STYLE-017 |
| :yellow_circle: Minor    | Correctness           | `config.py:L370-L386`                     | `assert` in `validate()` stripped by `-O`                               | CR-STYLE-018 |
| :blue_circle: Suggestion | DRY / Conciseness     | `world/clock.py:L17-L186`                 | `TimeGrid` delegates 5 pure-forwarding properties                       | CR-DRY-007   |
| :blue_circle: Suggestion | Performance           | `agent/quoting.py:L224-L256`              | Grid search ~601 points; unimodal objective allows scalar optimizer     | CR-PERF-009  |
| :blue_circle: Suggestion | Performance           | `agent/alpha.py:L174-L198`                | `_estimate_sigma_alpha` recomputes on every refresh; cacheable          | CR-PERF-015  |
| :blue_circle: Suggestion | Types                 | `agent/observable.py:L137-L174`           | Bare `tuple` return; named container preferred                          | CR-TYPE-016  |
