# Final Implementation Plan: Realistic Market Distributions

**Date**: 2026-02-28
**Status**: Final (incorporating all architecture refinements)

---

## Summary of Architectural Decisions

| Question | Decision | Impact |
|----------|----------|--------|
| Config organization | Nested dataclasses | `ArrivalConfig`, `SpreadConfig`, `ImbalanceConfig` |
| Process abstraction | `StochasticProcess` Protocol | Unified interface for all processes |
| Feature flag testing | Exhaustive 8-combination | Complete coverage matrix |
| Protocol location | `core/protocols.py` | New module in core |
| Refactor scope | All existing processes | `regime.py`, `street_lean.py` conform |
| Config compatibility | Deprecation period | Old params work with warnings |

---

## Refined Architecture

```
src/rfq_simulator/
├── core/
│   └── protocols.py        # NEW: StochasticProcess Protocol
├── config.py               # Nested dataclasses + deprecation
└── world/
    ├── hawkes.py           # NEW: Implements StochasticProcess
    ├── spread.py           # NEW: LogNormal spread sampling
    ├── imbalance.py        # NEW: Implements StochasticProcess
    ├── regime.py           # REFACTOR: Implement Protocol
    ├── street_lean.py      # REFACTOR: Implement Protocol
    ├── rfq_stream.py       # MODIFY: Use process abstractions
    └── competitors.py      # MODIFY: Use new spread model
```

---

## Phase 1: Core Protocol (`core/protocols.py`)

Create unified interface for all stochastic processes:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class StochasticProcess(Protocol):
    """Unified interface for all stochastic processes."""

    def step(self, dt: float = 1.0) -> float:
        """Advance by dt and return new state value."""
        ...

    def reset(self, initial_value: float | None = None) -> None:
        """Reset to initial state."""
        ...

    @property
    def value(self) -> float:
        """Current state value."""
        ...
```

---

## Phase 2: Nested Config Structure (`config.py`)

Replace flat parameters with nested dataclasses:

```python
@dataclass
class ArrivalConfig:
    """RFQ arrival process configuration."""
    use_hawkes: bool = False
    hawkes_alpha: float = 0.4
    hawkes_beta: float = 0.8
    hawkes_reset_daily: bool = True
    intraday_seasonality: bool = True

@dataclass
class SpreadConfig:
    """Spread distribution configuration."""
    use_lognormal: bool = False
    mu_calm: float = 2.0        # ~7 bps median
    sigma_calm: float = 0.5
    mu_stressed: float = 3.2    # ~25 bps median
    sigma_stressed: float = 0.8
    size_gamma: float = 0.15

@dataclass
class ImbalanceConfig:
    """Buy/sell imbalance configuration."""
    use_ar1: bool = False
    rho: float = 0.4
    sigma: float = 0.3
    mu_calm: float = 0.0
    mu_stressed: float = -0.25
    clip_low: float = 0.2
    clip_high: float = 0.8

@dataclass
class SimConfig:
    # === Nested configs ===
    arrivals: ArrivalConfig = field(default_factory=ArrivalConfig)
    spreads: SpreadConfig = field(default_factory=SpreadConfig)
    imbalance: ImbalanceConfig = field(default_factory=ImbalanceConfig)

    # === Deprecated flat params (with warnings) ===
    @property
    def use_hawkes_arrivals(self) -> bool:
        warnings.warn("use_hawkes_arrivals deprecated, use arrivals.use_hawkes",
                      DeprecationWarning)
        return self.arrivals.use_hawkes
```

---

## Phase 3: Hawkes Process (`world/hawkes.py`)

Implement `StochasticProcess` Protocol:

```python
class HawkesProcess:
    """Self-exciting point process implementing StochasticProcess."""

    def __init__(self, cfg: ArrivalConfig, rng: Generator):
        self._alpha = cfg.hawkes_alpha
        self._beta = cfg.hawkes_beta
        self._recursive_sum = 0.0
        self._last_time = 0.0
        self._rng = rng

    @property
    def value(self) -> float:
        """Current intensity (excluding baseline)."""
        return self._alpha * self._recursive_sum

    def step(self, dt: float = 1.0) -> float:
        """Decay kernel and return current excitation."""
        self._recursive_sum *= np.exp(-self._beta * dt)
        self._last_time += dt
        return self.value

    def record_event(self) -> None:
        """Add excitation from new event."""
        self._recursive_sum += 1.0

    def reset(self, initial_value: float | None = None) -> None:
        self._recursive_sum = initial_value or 0.0
        self._last_time = 0.0
```

---

## Phase 4: AR(1) Imbalance (`world/imbalance.py`)

Implement `StochasticProcess` Protocol:

```python
class ImbalanceProcess:
    """AR(1) flow imbalance implementing StochasticProcess."""

    def __init__(self, cfg: ImbalanceConfig, regime: Regime, rng: Generator):
        self._cfg = cfg
        self._regime = regime
        self._value = 0.0
        self._rng = rng

    @property
    def value(self) -> float:
        return self._value

    def step(self, dt: float = 1.0) -> float:
        mu = self._cfg.mu_calm if self._regime == Regime.CALM else self._cfg.mu_stressed
        self._value = (self._cfg.rho * self._value +
                       (1 - self._cfg.rho) * mu +
                       self._cfg.sigma * self._rng.standard_normal())
        return self._value

    def set_regime(self, regime: Regime) -> None:
        self._regime = regime

    def get_buy_probability(self) -> float:
        return np.clip(0.5 + self._value, self._cfg.clip_low, self._cfg.clip_high)

    def reset(self, initial_value: float | None = None) -> None:
        self._value = initial_value or 0.0
```

---

## Phase 5: Refactor Existing Processes

### `world/regime.py` — Add Protocol conformance:

```python
class RegimeProcess:
    """2-state Markov chain implementing StochasticProcess."""

    def __init__(self, cfg: SimConfig, rng: Generator):
        self._cfg = cfg
        self._rng = rng
        self._value = self._sample_stationary()

    @property
    def value(self) -> float:
        return float(self._regime.value)  # 0=CALM, 1=STRESSED

    def step(self, dt: float = 1.0) -> float:
        # Existing transition logic
        ...
        return self.value

    def reset(self, initial_value: float | None = None) -> None:
        self._regime = Regime(int(initial_value)) if initial_value else self._sample_stationary()
```

### `world/street_lean.py` — Already close to Protocol:

```python
class StreetLeanProcess:
    """OU process implementing StochasticProcess."""
    # Minor refactor: add reset() method, ensure value property exists
```

---

## Phase 6: LogNormal Spreads (`world/spread.py`)

Standalone spread sampling (not a process):

```python
def sample_dealer_spread(
    regime: Regime,
    size: int,
    cfg: SpreadConfig,
    rng: Generator
) -> float:
    """Sample spread from regime-dependent log-normal."""
    if regime == Regime.CALM:
        mu, sigma = cfg.mu_calm, cfg.sigma_calm
    else:
        mu, sigma = cfg.mu_stressed, cfg.sigma_stressed

    base_spread = np.exp(mu + sigma * rng.standard_normal())
    size_adj = 1 + cfg.size_gamma * np.log(1 + size)
    return base_spread * size_adj
```

---

## Phase 7: Integration

### `world/rfq_stream.py`:

```python
def generate_rfq_stream(
    cfg: SimConfig,
    rng: Generator,
    regime_path: np.ndarray | None = None
) -> List[RFQEvent]:
    # Dispatch based on nested config
    if cfg.arrivals.use_hawkes:
        arrival_times = _generate_hawkes_arrivals(cfg.arrivals, rng)
    else:
        arrival_times = _generate_poisson_arrivals(cfg, rng)

    # Initialize imbalance process if enabled
    imbalance = None
    if cfg.imbalance.use_ar1 and regime_path is not None:
        imbalance = ImbalanceProcess(cfg.imbalance, Regime.CALM, rng)

    events = []
    for t in arrival_times:
        if imbalance:
            day = int(t // cfg.minutes_per_day)
            regime = Regime(regime_path[day])
            imbalance.set_regime(regime)
            imbalance.step()
            is_buy = rng.random() < imbalance.get_buy_probability()
        else:
            is_buy = rng.random() < 0.5 + cfg.flow_bias

        events.append(_generate_rfq_attributes(t, is_buy, cfg, rng))

    return events
```

### `world/competitors.py`:

```python
def simulate_dealer_quote(
    rfq: RFQEvent,
    cfg: SimConfig,
    rng: Generator,
    regime: Regime = Regime.CALM
) -> float:
    if cfg.spreads.use_lognormal:
        spread = sample_dealer_spread(regime, rfq.size, cfg.spreads, rng)
    else:
        # Legacy normal model
        spread = cfg.base_markup + cfg.markup_sigma * rng.standard_normal()

    return spread + _toxicity_adjustment(rfq, cfg)
```

---

## Testing Strategy: 8-Combination Matrix

```python
@pytest.mark.parametrize("use_hawkes,use_lognormal,use_ar1", [
    (False, False, False),  # Baseline
    (True,  False, False),  # Hawkes only
    (False, True,  False),  # LogNormal only
    (False, False, True),   # AR(1) only
    (True,  True,  False),  # Hawkes + LogNormal
    (True,  False, True),   # Hawkes + AR(1)
    (False, True,  True),   # LogNormal + AR(1)
    (True,  True,  True),   # All enabled
])
def test_feature_combinations(use_hawkes, use_lognormal, use_ar1):
    cfg = SimConfig(
        arrivals=ArrivalConfig(use_hawkes=use_hawkes),
        spreads=SpreadConfig(use_lognormal=use_lognormal),
        imbalance=ImbalanceConfig(use_ar1=use_ar1),
    )
    result = run_simulation(cfg)
    assert result.total_rfqs > 0
    assert len(result.trades) >= 0
```

---

## Implementation Order

| Phase | Files | Dependencies |
|-------|-------|--------------|
| 1 | `core/protocols.py` | None |
| 2 | `config.py` | None |
| 3 | `world/hawkes.py` | Phase 1 |
| 4 | `world/imbalance.py` | Phase 1 |
| 5a | `world/regime.py` refactor | Phase 1 |
| 5b | `world/street_lean.py` refactor | Phase 1 |
| 6 | `world/spread.py` | Phase 2 |
| 7a | `world/rfq_stream.py` | Phases 3, 4 |
| 7b | `world/competitors.py` | Phase 6 |
| 8 | `simulation/event_loop.py` | Phase 7 |
| 9 | Test files | All phases |

---

## New Files Summary

| File | Purpose |
|------|---------|
| `core/__init__.py` | New core package |
| `core/protocols.py` | `StochasticProcess` Protocol |
| `world/hawkes.py` | Hawkes arrival process |
| `world/spread.py` | LogNormal spread sampling |
| `world/imbalance.py` | AR(1) flow imbalance |
| `tests/test_hawkes.py` | Hawkes unit tests |
| `tests/test_spread.py` | Spread unit tests |
| `tests/test_imbalance.py` | Imbalance unit tests |
| `tests/test_feature_matrix.py` | 8-combination tests |

---

## Success Criteria

- [ ] `StochasticProcess` Protocol defined and documented
- [ ] All processes (regime, street_lean, hawkes, imbalance) implement Protocol
- [ ] Nested configs with deprecation warnings for old params
- [ ] Hawkes: Inter-arrival ACF(1) > 0.1
- [ ] Spreads: Stressed mean > 2× calm mean
- [ ] Imbalance: Direction ACF(1) > 0.05
- [ ] All 58 existing tests pass
- [ ] All 8 feature combinations tested
- [ ] Simulation runs end-to-end with all flags enabled
