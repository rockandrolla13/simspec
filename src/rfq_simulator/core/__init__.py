"""
Core module: State management, accounting, and protocols.

- state: SimulationState tracking
- accounting: P&L decomposition and tracking
- hedging: Optional hedging overlay
"""

from .state import SimulationState, RFQLog
from .accounting import PnLTracker, PnLDecomposition

__all__ = [
    "SimulationState",
    "RFQLog",
    "PnLTracker",
    "PnLDecomposition",
]
