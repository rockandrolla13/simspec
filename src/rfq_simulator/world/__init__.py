"""
World module: Exogenous processes that define the market environment.

- clock: Time grid utilities
- price: True price path generation
- regime: Calm/stressed regime Markov chain
- rfq_stream: RFQ arrival process
- street_lean: Aggregate dealer lean process
- competitors: Explicit competitor quote simulation
- spread: Regime-dependent spread distribution
- imbalance: Buy/sell flow imbalance process
- hawkes: Hawkes process for clustered arrivals
"""

from .clock import TimeGrid
from .price import generate_price_path, apply_adverse_move
from .regime import generate_regime_path
from .imbalance import ImbalanceProcess, compute_expected_buy_fraction
from .rfq_stream import generate_rfq_stream, RFQEvent
from .hawkes import HawkesProcess, generate_hawkes_arrivals
from .competitors import (
    DealerPool,
    DealerQuote,
    CompetitionResult,
    simulate_competition,
    generate_win_rate_curve,
)
from .spread import (
    get_regime_spread_params,
    sample_base_spread,
    apply_size_adjustment,
    sample_dealer_spread,
    compute_expected_spread,
    compute_median_spread,
)

__all__ = [
    "TimeGrid",
    "generate_price_path",
    "apply_adverse_move",
    "generate_regime_path",
    "generate_rfq_stream",
    "RFQEvent",
    "HawkesProcess",
    "generate_hawkes_arrivals",
    "DealerPool",
    "DealerQuote",
    "CompetitionResult",
    "simulate_competition",
    "generate_win_rate_curve",
    "ImbalanceProcess",
    "compute_expected_buy_fraction",
    # Spread functions
    "get_regime_spread_params",
    "sample_base_spread",
    "apply_size_adjustment",
    "sample_dealer_spread",
    "compute_expected_spread",
    "compute_median_spread",
]
