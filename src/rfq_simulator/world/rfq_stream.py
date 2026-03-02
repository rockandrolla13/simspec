"""
RFQ arrival process: Inhomogeneous Poisson with intraday seasonality.

Implements Section 6 from spec:
    μ(h) = μ_base * f(h)
    f(h) = 1 + A_open * e^{-(h-h0)²/τ²} + A_close * e^{-(h-hc)²/τ²}

Each RFQ has attributes:
    - Direction: P(client buy) = 0.5 + δ_flow
    - Size: ceil(exp(μ_s + σ_s * Z)), clipped to [1, size_max]
    - Dealer count: 1 + Poisson(N̄ - 1), clipped to [1, N_max]
    - Toxicity: Beta(a_tox, b_tox)

Optionally supports:
    - Hawkes self-exciting arrivals (when cfg.arrivals.use_hawkes=True)
    - AR(1) buy/sell imbalance (when cfg.imbalance.use_ar1=True)
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.random import Generator

from ..config import SimConfig
from .clock import TimeGrid, compute_intraday_intensity
from .hawkes import generate_hawkes_arrivals
from .imbalance import ImbalanceProcess
from .regime import Regime


@dataclass
class RFQEvent:
    """
    A single RFQ event.

    Attributes:
        time: Arrival time in minutes from simulation start
        is_client_buy: True if client is buying (we would sell/offer)
        size: Trade size in lots
        n_dealers: Number of competing dealers (including us)
        toxicity: Client toxicity τ_c ∈ [0, 1]
    """

    time: float
    is_client_buy: bool
    size: int
    n_dealers: int
    toxicity: float

    @property
    def direction(self) -> int:
        """Direction from trader's perspective: +1 if we buy, -1 if we sell."""
        return -1 if self.is_client_buy else +1


def generate_rfq_stream(
    cfg: SimConfig,
    rng: Generator,
    regime_path: np.ndarray | None = None,
) -> List[RFQEvent]:
    """
    Generate RFQ arrivals via thinning algorithm for inhomogeneous Poisson.

    Uses the thinning method:
    1. Generate candidates from homogeneous Poisson with rate μ_max
    2. Accept each candidate with probability f(h) / max_f

    When cfg.arrivals.use_hawkes=True, uses Hawkes self-exciting process.
    When cfg.imbalance.use_ar1=True and regime_path is provided, uses AR(1)
    imbalance process for buy/sell direction.

    Args:
        cfg: SimConfig with all RFQ parameters
        rng: Random generator
        regime_path: Optional regime path array for AR(1) imbalance

    Returns:
        List of RFQEvent sorted by time
    """
    # Generate arrival times - dispatch to Hawkes or Poisson
    if cfg.arrivals.use_hawkes:
        arrival_times = generate_hawkes_arrivals(cfg, rng)
    else:
        arrival_times = _generate_poisson_arrivals(cfg, rng)

    # Initialize imbalance process if enabled
    imbalance = None
    if cfg.imbalance.use_ar1 and regime_path is not None:
        imbalance = ImbalanceProcess(cfg.imbalance, rng)

    # Generate RFQ events
    events = []
    for t in arrival_times:
        if imbalance is not None:
            day = int(t // cfg.minutes_per_day)
            day = min(day, len(regime_path) - 1)
            regime = Regime(regime_path[day])
            imbalance.set_regime(regime)
            imbalance.step()
            is_client_buy = imbalance.sample_direction()
        else:
            is_client_buy = rng.random() < (0.5 + cfg.flow_bias)

        event = _generate_rfq_attributes(t, is_client_buy, cfg, rng)
        events.append(event)

    return events


def _generate_poisson_arrivals(cfg: SimConfig, rng: Generator) -> List[float]:
    """
    Generate arrival times via thinning algorithm for inhomogeneous Poisson.

    Uses the thinning method:
    1. Generate candidates from homogeneous Poisson with rate μ_max
    2. Accept each candidate with probability f(h) / max_f

    Args:
        cfg: SimConfig with all RFQ parameters
        rng: Random generator

    Returns:
        List of arrival times in minutes from simulation start
    """
    time_grid = TimeGrid(cfg)
    total_minutes = cfg.total_minutes

    # Compute max intensity for thinning
    # Sample intensity at many hours to find max
    hours = np.linspace(0, cfg.trading_hours, 100)
    intensities = [compute_intraday_intensity(h, cfg) for h in hours]
    max_f = max(intensities)

    # Base rate per minute
    mu_base = cfg.rfq_rate_per_day / cfg.minutes_per_day

    # Max rate for thinning
    mu_max = mu_base * max_f

    # Generate candidates using thinning
    arrivals = []
    t = 0.0

    while t < total_minutes:
        # Draw inter-arrival time from Exp(μ_max)
        dt = rng.exponential(1.0 / mu_max) if mu_max > 0 else float("inf")
        t += dt

        if t >= total_minutes:
            break

        # Get hour of day and intensity
        hour = time_grid.minute_to_hour_of_day(t)
        f_t = compute_intraday_intensity(hour, cfg)

        # Accept with probability f(t) / max_f
        if rng.random() < f_t / max_f:
            arrivals.append(t)

    return arrivals


def _generate_rfq_attributes(
    time: float, is_client_buy: bool, cfg: SimConfig, rng: Generator
) -> RFQEvent:
    """
    Generate attributes for a single RFQ.

    Args:
        time: Arrival time in minutes
        is_client_buy: Whether client is buying (pre-determined by caller)
        cfg: SimConfig
        rng: Random generator

    Returns:
        RFQEvent with all attributes
    """
    # Size: ceil(exp(μ_s + σ_s * Z)), clipped to [1, size_max]
    log_size = cfg.size_mu + cfg.size_sigma * rng.standard_normal()
    size = int(np.ceil(np.exp(log_size)))
    size = max(1, min(size, cfg.size_max))

    # Dealer count: 1 + Poisson(N̄ - 1), clipped to [1, N_max]
    n_dealers = 1 + rng.poisson(max(0, cfg.n_dealers_mean - 1))
    n_dealers = max(1, min(n_dealers, cfg.n_dealers_max))

    # Toxicity: Beta(a, b)
    toxicity = rng.beta(cfg.tox_a, cfg.tox_b)

    return RFQEvent(
        time=time,
        is_client_buy=is_client_buy,
        size=size,
        n_dealers=n_dealers,
        toxicity=toxicity,
    )


def compute_expected_rfq_rate(cfg: SimConfig) -> float:
    """
    Compute the expected daily RFQ rate accounting for intraday seasonality.

    Should equal rfq_rate_per_day when seasonality integrates to trading_hours.

    Args:
        cfg: SimConfig

    Returns:
        Expected daily RFQ count
    """
    # Numerical integration over trading day
    hours = np.linspace(0, cfg.trading_hours, 1000)
    intensities = [compute_intraday_intensity(h, cfg) for h in hours]
    avg_intensity = np.mean(intensities)

    # Base rate scaled by average intensity
    mu_base = cfg.rfq_rate_per_day / cfg.minutes_per_day
    return mu_base * avg_intensity * cfg.minutes_per_day


def filter_rfq_by_day(events: List[RFQEvent], day: int, cfg: SimConfig) -> List[RFQEvent]:
    """
    Filter RFQ events to a specific trading day.

    Args:
        events: Full list of RFQ events
        day: Trading day (0-indexed)
        cfg: SimConfig

    Returns:
        Events within the specified day
    """
    time_grid = TimeGrid(cfg)
    start, end = time_grid.day_range(day)
    return [e for e in events if start <= e.time < end]
