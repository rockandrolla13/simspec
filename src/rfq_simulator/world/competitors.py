"""
Competitor model: Explicit simulation of N-1 dealer quotes per RFQ.

Implements Eq 16-20 from spec:
    p_j = p_true + b_j + m_j(z) + ε_j

where:
    b_j ~ N(b̄_street, σ_b)           # Daily bias from street lean
    m_j(z) = m̄ + m_N*(N-N̄) + m_size*log(size) + m_tox*τ_c + u_j
    ε_j ~ N(0, σ_q)                   # Quote noise

Participation:
    P(respond) = clip(r̄ - r_size*log(size) - r_tox*τ_c, 0.3, 1.0)

This is the CRITICAL module - win probability emerges from competition,
not from an assumed logistic curve.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.random import Generator

from ..config import SimConfig
from .rfq_stream import RFQEvent


@dataclass
class DealerQuote:
    """
    A quote from a single competing dealer.

    Attributes:
        dealer_id: Index of the dealer (0 to n_dealers-2, excluding "us")
        price: Quoted price in dollars
        responded: Whether dealer chose to respond
        markup: The markup component m_j(z) in bps (for diagnostics)
        bias: The dealer's daily bias b_j in bps (for diagnostics)
    """

    dealer_id: int
    price: float
    responded: bool
    markup: float
    bias: float


@dataclass
class CompetitionResult:
    """
    Result of simulating competition for an RFQ.

    Attributes:
        n_competitors_responded: How many of N-1 dealers responded
        competitor_quotes: List of DealerQuote objects
        best_competitor_price: Best competing price (None if no responses)
        trader_won: Whether trader's quote won
        trader_price: The trader's submitted price
    """

    n_competitors_responded: int
    competitor_quotes: List[DealerQuote]
    best_competitor_price: Optional[float]
    trader_won: bool
    trader_price: float


class DealerPool:
    """
    Manages a pool of competing dealers with daily bias refreshes.

    Each dealer has a persistent bias b_j that is refreshed daily,
    plus per-quote noise.
    """

    def __init__(self, n_dealers_max: int, cfg: SimConfig, rng: Generator):
        """
        Initialize the dealer pool.

        Args:
            n_dealers_max: Maximum number of dealers in any RFQ
            cfg: SimConfig
            rng: Random generator
        """
        self.cfg = cfg
        self.rng = rng
        self.n_dealers_max = n_dealers_max

        # Initialize dealer biases (in bps)
        # These represent daily "lean" - some dealers run long, some short
        self.dealer_biases = self._draw_biases()
        self.current_day = 0

    def _draw_biases(self, street_lean: float = 0.0) -> np.ndarray:
        """
        Draw new daily biases for all dealers.

        Each dealer's bias is centered on the street lean with idiosyncratic dispersion.

        Args:
            street_lean: Current street lean in bps (mean for bias distribution)

        Returns:
            Array of dealer biases in bps
        """
        return self.rng.normal(street_lean, self.cfg.dealer_bias_std_bps, size=self.n_dealers_max)

    def refresh_biases(self, day: int, street_lean: float = 0.0) -> None:
        """
        Refresh dealer biases at the start of a new day.

        Biases are drawn centered on the current street lean, representing
        how dealer positioning tracks aggregate market positioning.

        Args:
            day: Current trading day
            street_lean: Current street lean in bps
        """
        if day > self.current_day:
            self.dealer_biases = self._draw_biases(street_lean)
            self.current_day = day

    def get_bias(self, dealer_id: int) -> float:
        """Get the current bias for a dealer (in bps)."""
        return self.dealer_biases[dealer_id % self.n_dealers_max]


def compute_dealer_markup(
    rfq: RFQEvent,
    cfg: SimConfig,
    rng: Generator,
) -> float:
    """
    Compute the systematic markup for an RFQ (shared basis for all dealers).

    Eq 17: m(z) = m̄ + m_N*(N-N̄) + m_size*log(size) + m_tox*τ_c

    This is the expected markup before per-dealer noise.

    Args:
        rfq: RFQ event
        cfg: SimConfig
        rng: Random generator (for per-dealer noise)

    Returns:
        Systematic markup in bps
    """
    # Baseline markup
    markup = cfg.markup_base_bps

    # Dealer count effect (more dealers = tighter)
    markup += cfg.markup_N_bps * (rfq.n_dealers - cfg.n_dealers_mean)

    # Size effect (larger = wider)
    markup += cfg.markup_size_bps * np.log(max(1, rfq.size))

    # Toxicity effect (more toxic = wider)
    markup += cfg.markup_tox_bps * rfq.toxicity

    return markup


def compute_response_probability(rfq: RFQEvent, cfg: SimConfig) -> float:
    """
    Compute probability that a dealer responds to an RFQ.

    Eq 20: P(respond) = clip(r̄ - r_size*log(size) - r_tox*τ_c, 0.3, 1.0)

    Larger and more toxic RFQs have lower response rates.

    Args:
        rfq: RFQ event
        cfg: SimConfig

    Returns:
        Response probability in [0.3, 1.0]
    """
    prob = cfg.respond_base
    prob -= cfg.respond_size * np.log(max(1, rfq.size))
    prob -= cfg.respond_tox * rfq.toxicity

    return np.clip(prob, 0.3, 1.0)


def simulate_dealer_quote(
    rfq: RFQEvent,
    p_true: float,
    dealer_id: int,
    dealer_bias: float,
    street_lean: float,
    cfg: SimConfig,
    rng: Generator,
) -> DealerQuote:
    """
    Simulate a single dealer's quote decision and price.

    Eq 16-19:
        p_j = p_true + b_j + m_j(z) + ε_j
        b_j = b̄_street + daily_bias_j
        m_j(z) = m(z) + u_j  (where u_j is per-quote noise)
        ε_j ~ N(0, σ_q)

    For client-buy (offers): lower is more aggressive
    For client-sell (bids): higher is more aggressive

    Args:
        rfq: RFQ event
        p_true: True market price
        dealer_id: Index of this dealer
        dealer_bias: This dealer's daily bias (in bps)
        street_lean: Current street lean (in bps)
        cfg: SimConfig
        rng: Random generator

    Returns:
        DealerQuote with price and response decision
    """
    # Check if dealer responds
    p_respond = compute_response_probability(rfq, cfg)
    responded = rng.random() < p_respond

    # Compute dealer's effective bias
    # Combines street lean (common to all) + dealer's idiosyncratic bias
    effective_bias = street_lean + dealer_bias

    # Compute markup with per-dealer noise
    base_markup = compute_dealer_markup(rfq, cfg, rng)
    markup_noise = rng.normal(0.0, cfg.markup_noise_bps)
    total_markup = base_markup + markup_noise

    # Quote noise (execution uncertainty)
    quote_noise = rng.normal(0.0, cfg.quote_noise_bps)

    # Direction adjustment:
    # - Client buy (offer): dealer sells, markup is ADDED to true price
    # - Client sell (bid): dealer buys, markup is SUBTRACTED from true price
    direction = 1 if rfq.is_client_buy else -1

    # Convert bps to dollars
    bps_to_dollars = cfg.p0 / 10000.0

    # Final price
    price = p_true + direction * (
        (effective_bias + total_markup + quote_noise) * bps_to_dollars
    )

    return DealerQuote(
        dealer_id=dealer_id,
        price=price,
        responded=responded,
        markup=total_markup,
        bias=effective_bias,
    )


def simulate_competition(
    rfq: RFQEvent,
    p_true: float,
    trader_price: float,
    dealer_pool: DealerPool,
    street_lean: float,
    rng: Generator,
) -> CompetitionResult:
    """
    Simulate the full competition for an RFQ.

    Generates quotes from all N-1 competing dealers, determines
    who responded, and resolves the winner.

    Args:
        rfq: RFQ event (includes n_dealers which is total including us)
        p_true: True market price
        trader_price: Our submitted price
        dealer_pool: DealerPool for getting biases
        street_lean: Current street lean (in bps)
        rng: Random generator

    Returns:
        CompetitionResult with full details
    """
    cfg = dealer_pool.cfg

    # N-1 other dealers (n_dealers includes us)
    n_competitors = rfq.n_dealers - 1

    # Simulate each competitor
    quotes = []
    for i in range(n_competitors):
        bias = dealer_pool.get_bias(i)
        quote = simulate_dealer_quote(
            rfq=rfq,
            p_true=p_true,
            dealer_id=i,
            dealer_bias=bias,
            street_lean=street_lean,
            cfg=cfg,
            rng=rng,
        )
        quotes.append(quote)

    # Filter to responding dealers
    responded_quotes = [q for q in quotes if q.responded]
    n_responded = len(responded_quotes)

    # Determine best competitor price
    if n_responded == 0:
        # No competition - we win by default (if we responded)
        best_price = None
        trader_won = True
    else:
        if rfq.is_client_buy:
            # Client buying, we're offering: lowest price wins
            best_price = min(q.price for q in responded_quotes)
            trader_won = trader_price <= best_price
        else:
            # Client selling, we're bidding: highest price wins
            best_price = max(q.price for q in responded_quotes)
            trader_won = trader_price >= best_price

    return CompetitionResult(
        n_competitors_responded=n_responded,
        competitor_quotes=quotes,
        best_competitor_price=best_price,
        trader_won=trader_won,
        trader_price=trader_price,
    )


def compute_empirical_win_rate(
    markup_bps: float,
    rfq: RFQEvent,
    p_true: float,
    dealer_pool: DealerPool,
    street_lean: float,
    n_simulations: int,
    rng: Generator,
) -> float:
    """
    Compute empirical win rate at a given markup via Monte Carlo.

    This is useful for validation - should produce a smooth sigmoid.

    Args:
        markup_bps: Trader's markup over true price in bps
        rfq: RFQ event
        p_true: True market price
        dealer_pool: DealerPool
        street_lean: Current street lean
        n_simulations: Number of MC samples
        rng: Random generator

    Returns:
        Estimated win probability
    """
    cfg = dealer_pool.cfg
    bps_to_dollars = cfg.p0 / 10000.0

    # Direction: client buy means we offer (add markup), client sell means we bid (subtract)
    direction = 1 if rfq.is_client_buy else -1
    trader_price = p_true + direction * markup_bps * bps_to_dollars

    wins = 0
    for _ in range(n_simulations):
        result = simulate_competition(
            rfq=rfq,
            p_true=p_true,
            trader_price=trader_price,
            dealer_pool=dealer_pool,
            street_lean=street_lean,
            rng=rng,
        )
        if result.trader_won:
            wins += 1

    return wins / n_simulations


def generate_win_rate_curve(
    rfq: RFQEvent,
    p_true: float,
    dealer_pool: DealerPool,
    street_lean: float,
    markup_range_bps: Tuple[float, float] = (-10, 30),
    n_points: int = 41,
    n_simulations: int = 1000,
    rng: Generator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an empirical win-rate curve for validation.

    Args:
        rfq: RFQ event
        p_true: True market price
        dealer_pool: DealerPool
        street_lean: Current street lean
        markup_range_bps: (min, max) markup range
        n_points: Number of markup points
        n_simulations: MC samples per point
        rng: Random generator

    Returns:
        (markups, win_rates) arrays
    """
    if rng is None:
        rng = np.random.default_rng()

    markups = np.linspace(markup_range_bps[0], markup_range_bps[1], n_points)
    win_rates = []

    for m in markups:
        wr = compute_empirical_win_rate(
            markup_bps=m,
            rfq=rfq,
            p_true=p_true,
            dealer_pool=dealer_pool,
            street_lean=street_lean,
            n_simulations=n_simulations,
            rng=rng,
        )
        win_rates.append(wr)

    return markups, np.array(win_rates)
