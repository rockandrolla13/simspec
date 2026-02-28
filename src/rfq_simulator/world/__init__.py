"""
World module: Exogenous processes that define the market environment.

- clock: Time grid utilities
- price: True price path generation
- regime: Calm/stressed regime Markov chain
- rfq_stream: RFQ arrival process
- street_lean: Aggregate dealer lean process
- competitors: Explicit competitor quote simulation
"""

from .clock import TimeGrid
from .price import generate_price_path, apply_adverse_move
from .regime import generate_regime_path
from .rfq_stream import generate_rfq_stream, RFQEvent
from .competitors import (
    DealerPool,
    DealerQuote,
    CompetitionResult,
    simulate_competition,
    generate_win_rate_curve,
)

__all__ = [
    "TimeGrid",
    "generate_price_path",
    "apply_adverse_move",
    "generate_regime_path",
    "generate_rfq_stream",
    "RFQEvent",
    "DealerPool",
    "DealerQuote",
    "CompetitionResult",
    "simulate_competition",
    "generate_win_rate_curve",
]
