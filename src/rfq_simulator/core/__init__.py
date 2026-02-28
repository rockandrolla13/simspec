"""
Core module: State management, accounting, and protocols.

- protocols: StochasticProcess protocol for unified process interface
- state: SimulationState tracking
- accounting: P&L decomposition and tracking
- hedging: Optional hedging overlay
"""

from .protocols import StochasticProcess
from .state import SimulationState, RFQLog
from .accounting import PnLTracker, PnLDecomposition

__all__ = [
    "StochasticProcess",
    "SimulationState",
    "RFQLog",
    "PnLTracker",
    "PnLDecomposition",
]
