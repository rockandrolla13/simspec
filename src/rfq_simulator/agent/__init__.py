"""
Agent module: Strategy logic that reacts to the market environment.

- alpha: Alpha signal generation, decay, refresh
- target: Position targeting from alpha
- observable: Lagged mid observation and skew correction
- lean: Lean computation with urgency and convexity
- winrate: Trader's estimated win-rate model
- quoting: Optimal quote computation
- exit: Hybrid exit logic
"""

from .alpha import AlphaSignalManager, AlphaSignal
from .target import compute_target_position, compute_continuation_value
from .observable import compute_observable_mid, compute_skew, compute_theo_price, TheoResult
from .lean import compute_lean, compute_lean_decomposition
from .winrate import estimate_win_probability, generate_estimated_win_curve
from .quoting import compute_optimal_quote, QuoteResult

__all__ = [
    "AlphaSignalManager",
    "AlphaSignal",
    "compute_target_position",
    "compute_continuation_value",
    "compute_observable_mid",
    "compute_skew",
    "compute_theo_price",
    "TheoResult",
    "compute_lean",
    "compute_lean_decomposition",
    "estimate_win_probability",
    "generate_estimated_win_curve",
    "compute_optimal_quote",
    "QuoteResult",
]
