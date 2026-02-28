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
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.random import Generator

from ..config import SimConfig
from .clock import TimeGrid


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


def compute_intraday_intensity(hour: float, cfg: SimConfig) -> float:
    """
    Compute the intraday RFQ intensity multiplier.

    Eq 10: f(h) = 1 + A_open * e^{-(h-h0)²/τ²} + A_close * e^{-(h-hc)²/τ²}

    Args:
        hour: Hour within the trading day (0 to trading_hours)
        cfg: SimConfig with A_open, A_close, tau_f_hours, trading_hours

    Returns:
        Intensity multiplier >= 1.0
    """
    h0 = 0.0  # Market open
    hc = cfg.trading_hours  # Market close
    tau_sq = cfg.tau_f_hours ** 2

    open_bump = cfg.A_open * np.exp(-(hour - h0) ** 2 / tau_sq)
    close_bump = cfg.A_close * np.exp(-(hour - hc) ** 2 / tau_sq)

    return 1.0 + open_bump + close_bump


def generate_rfq_stream(cfg: SimConfig, rng: Generator) -> List[RFQEvent]:
    """
    Generate RFQ arrivals via thinning algorithm for inhomogeneous Poisson.

    Uses the thinning method:
    1. Generate candidates from homogeneous Poisson with rate μ_max
    2. Accept each candidate with probability f(h) / max_f

    Args:
        cfg: SimConfig with all RFQ parameters
        rng: Random generator

    Returns:
        List of RFQEvent sorted by time
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
    events = []
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
            # Generate RFQ attributes
            event = _generate_rfq_attributes(t, cfg, rng)
            events.append(event)

    return events


def _generate_rfq_attributes(time: float, cfg: SimConfig, rng: Generator) -> RFQEvent:
    """
    Generate attributes for a single RFQ.

    Args:
        time: Arrival time in minutes
        cfg: SimConfig
        rng: Random generator

    Returns:
        RFQEvent with all attributes
    """
    # Direction: P(client buy) = 0.5 + δ_flow
    is_client_buy = rng.random() < (0.5 + cfg.flow_bias)

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
