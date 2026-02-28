"""
RFQ Simulator: Alpha-driven liquidity provision in corporate bond markets.

This simulator implements the strategy described in sim_spec_v2.pdf:
- Request-for-Quote (RFQ) based trading
- Alpha signal with regime-dependent IC
- Explicit competitor modeling
- P&L decomposition and analysis

Quick start:
    from rfq_simulator import SimConfig, run_simulation

    cfg = SimConfig(T_days=60, IC=0.10)
    result = run_simulation(cfg, verbose=True)
    print(result.summary())

Modules:
    config: SimConfig dataclass with all parameters
    world: Market environment (prices, regimes, RFQs, competitors)
    agent: Strategy logic (alpha, quoting, lean, exit)
    core: State management and P&L accounting
    simulation: Event loop, baseline, batch runner
    output: Diagnostics and visualization
"""

from .config import SimConfig
from .simulation import run_simulation, SimulationResult
from .simulation.baseline import run_baseline, BaselineResult, compare_strategies
from .simulation.batch import run_batch, run_scenario_sweep, BatchResult

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "SimConfig",

    # Simulation
    "run_simulation",
    "SimulationResult",
    "run_baseline",
    "BaselineResult",
    "compare_strategies",

    # Batch
    "run_batch",
    "run_scenario_sweep",
    "BatchResult",
]
