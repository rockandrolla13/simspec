"""
Hawkes self-exciting point process for RFQ arrivals.

Implements intensity function:
    λ(t) = μ(t) + Σ_{t_i < t} α · exp(-β · (t - t_i))

Uses recursive sum for O(1) intensity computation:
    R(t) = Σ_{t_i < t} exp(-β · (t - t_i))
    λ(t) = μ(t) + α · R(t)

References:
- Hawkes (1971) "Spectra of Some Self-Exciting and Mutually Exciting Point Processes"
- Bacry et al. (2015) "Hawkes Processes in Finance"
"""

from dataclasses import dataclass
from typing import List
import numpy as np
from numpy.random import Generator

from ..config import ArrivalConfig, SimConfig
from .clock import TimeGrid


@dataclass
class HawkesState:
    """State for recursive sum computation."""
    recursive_sum: float = 0.0
    last_update_time: float = 0.0


class HawkesProcess:
    """Self-exciting point process implementing StochasticProcess protocol."""

    def __init__(self, cfg: ArrivalConfig, rng: Generator):
        self._alpha = cfg.hawkes_alpha
        self._beta = cfg.hawkes_beta
        self._reset_daily = cfg.hawkes_reset_daily
        self._rng = rng
        self._state = HawkesState()

    @property
    def value(self) -> float:
        """Current excitation level (excluding baseline)."""
        return self._alpha * self._state.recursive_sum

    def step(self, dt: float = 1.0) -> float:
        """Decay the kernel by dt and return current excitation."""
        self._state.recursive_sum *= np.exp(-self._beta * dt)
        self._state.last_update_time += dt
        return self.value

    def record_event(self) -> None:
        """Add excitation from a new event arrival."""
        self._state.recursive_sum += 1.0

    def reset(self, initial_value: float | None = None) -> None:
        """Reset to initial state."""
        self._state.recursive_sum = initial_value if initial_value is not None else 0.0
        self._state.last_update_time = 0.0

    def get_intensity(self, baseline: float) -> float:
        """Get total intensity = baseline + excitation."""
        return baseline + self.value


def generate_hawkes_arrivals(
    cfg: SimConfig,
    rng: Generator,
) -> List[float]:
    """
    Generate arrival times using Hawkes process with Ogata's thinning.

    Algorithm:
    1. Propose candidate from Exp(λ_max)
    2. Accept with probability λ(t) / λ_max
    3. Update λ_max after acceptance (intensity increased)

    Args:
        cfg: SimConfig with arrivals config and seasonality params
        rng: Random generator

    Returns:
        List of arrival times in minutes from simulation start
    """
    from .rfq_stream import compute_intraday_intensity  # Avoid circular import

    time_grid = TimeGrid(cfg)
    total_minutes = cfg.total_minutes
    arrival_cfg = cfg.arrivals

    # Base rate per minute
    mu_base = cfg.rfq_rate_per_day / cfg.minutes_per_day

    # Initialize Hawkes state
    hawkes = HawkesProcess(arrival_cfg, rng)

    # Compute max seasonality multiplier
    hours = np.linspace(0, cfg.trading_hours, 100)
    max_seasonality = max(compute_intraday_intensity(h, cfg) for h in hours)

    # Initial upper bound for thinning
    lambda_max = mu_base * max_seasonality + arrival_cfg.hawkes_alpha

    arrivals = []
    t = 0.0
    current_day = 0

    while t < total_minutes:
        # Draw candidate inter-arrival time
        dt = rng.exponential(1.0 / lambda_max) if lambda_max > 0 else float("inf")
        t += dt

        if t >= total_minutes:
            break

        # Check for day change and reset if needed
        new_day = int(t // cfg.minutes_per_day)
        if new_day > current_day and arrival_cfg.hawkes_reset_daily:
            hawkes.reset()
            current_day = new_day

        # Decay Hawkes state to current time
        hawkes.step(dt)

        # Compute current intensity
        hour = time_grid.minute_to_hour_of_day(t)
        baseline = mu_base * compute_intraday_intensity(hour, cfg)
        lambda_t = hawkes.get_intensity(baseline)

        # Accept with probability λ(t) / λ_max
        if rng.random() < lambda_t / lambda_max:
            arrivals.append(t)
            hawkes.record_event()
            # Update upper bound after event (intensity just jumped)
            lambda_max = max(lambda_max, lambda_t + arrival_cfg.hawkes_alpha)

    return arrivals
