"""
Simulation module: Main simulation loop and strategies.

- event_loop: Core discrete-event simulation
- baseline: Naive aggressor benchmark
- batch: Monte Carlo runner
"""

from .event_loop import run_simulation, SimulationResult

__all__ = [
    "run_simulation",
    "SimulationResult",
]
