"""
Street lean: OU process for aggregate dealer positioning.

Implements Eq 28-32 from spec:
    db̄_t = θ_b * (b̄_eq - b̄_t) * dt + σ_b * √dt * ε_t

The street lean represents the aggregate bias of all dealers:
- When street is long (b̄ > 0), dealers quote lower to reduce inventory
- When street is short (b̄ < 0), dealers quote higher to rebuild

Proxies for street lean estimation:
1. Bid-ask asymmetry
2. Flow imbalance
3. ETF premium/discount
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.random import Generator

from ..config import SimConfig
from ..world.clock import TimeGrid


def generate_street_lean_path(
    cfg: SimConfig,
    rng: Generator,
) -> np.ndarray:
    """
    Generate the true street lean as an OU process.

    db̄_t = θ * (b̄_eq - b̄_t) * dt + σ * √dt * ε

    Args:
        cfg: SimConfig with street_lean_mean_rev, street_lean_vol_bps, street_lean_eq
        rng: Random generator

    Returns:
        Array of street lean values (in bps), one per time step
    """
    n_steps = cfg.n_steps
    dt = cfg.dt_minutes / cfg.minutes_per_day  # In days

    theta = cfg.street_lean_mean_rev
    sigma = cfg.street_lean_vol_bps
    b_eq = cfg.street_lean_eq

    # Initialize from equilibrium
    lean = np.zeros(n_steps)
    lean[0] = b_eq

    # Simulate OU process
    for t in range(1, n_steps):
        drift = theta * (b_eq - lean[t - 1]) * dt
        diffusion = sigma * np.sqrt(dt) * rng.standard_normal()
        lean[t] = lean[t - 1] + drift + diffusion

    return lean


def get_street_lean_at_step(
    street_lean_path: np.ndarray,
    step: int,
) -> float:
    """
    Get street lean at a given time step.

    Args:
        street_lean_path: Full street lean path
        step: Time step index

    Returns:
        Street lean in bps
    """
    step = max(0, min(step, len(street_lean_path) - 1))
    return street_lean_path[step]


@dataclass
class StreetLeanEstimate:
    """
    Trader's estimate of street lean from observable proxies.

    Attributes:
        estimate_bps: Combined estimate in bps
        bid_ask_signal: Signal from bid-ask asymmetry
        flow_signal: Signal from recent flow imbalance
        etf_signal: Signal from ETF premium/discount
        confidence: Estimation confidence (0 to 1)
    """

    estimate_bps: float
    bid_ask_signal: float
    flow_signal: float
    etf_signal: float
    confidence: float


def estimate_street_lean_from_proxies(
    bid_ask_asymmetry: float,
    flow_imbalance: float,
    etf_premium_bps: float,
    cfg: SimConfig,
    rng: Generator,
) -> StreetLeanEstimate:
    """
    Estimate street lean from observable proxies.

    Eq 30-31: Each proxy is a noisy signal of true street lean.

    Args:
        bid_ask_asymmetry: (best_bid - mid) / (ask - bid), in [-1, 1]
        flow_imbalance: (buy_volume - sell_volume) / total_volume, in [-1, 1]
        etf_premium_bps: ETF premium to NAV in bps
        cfg: SimConfig with proxy weights and noise
        rng: Random generator

    Returns:
        StreetLeanEstimate with combined estimate
    """
    w1, w2, w3 = cfg.street_proxy_weights
    noise = cfg.street_obs_noise

    # Add noise to each proxy
    noisy_ba = bid_ask_asymmetry + noise * rng.standard_normal()
    noisy_flow = flow_imbalance + noise * rng.standard_normal()
    noisy_etf = etf_premium_bps / 10.0 + noise * rng.standard_normal()  # Normalize

    # Scale proxies to bps
    ba_signal = noisy_ba * 5.0  # Scale asymmetry to ~5 bps
    flow_signal = noisy_flow * 3.0  # Scale flow to ~3 bps
    etf_signal = noisy_etf * 2.0  # ETF already in bps-like units

    # Weighted combination
    estimate = w1 * ba_signal + w2 * flow_signal + w3 * etf_signal

    # Confidence based on signal agreement
    signals = [ba_signal, flow_signal, etf_signal]
    signal_std = np.std(signals)
    confidence = 1.0 / (1.0 + signal_std / 3.0)  # Higher agreement = higher confidence

    return StreetLeanEstimate(
        estimate_bps=estimate,
        bid_ask_signal=ba_signal,
        flow_signal=flow_signal,
        etf_signal=etf_signal,
        confidence=confidence,
    )


def simulate_noisy_street_lean_observation(
    true_lean: float,
    cfg: SimConfig,
    rng: Generator,
) -> float:
    """
    Simulate a noisy observation of true street lean.

    Simplified version: add Gaussian noise to true value.

    Args:
        true_lean: True street lean in bps
        cfg: SimConfig
        rng: Random generator

    Returns:
        Noisy observation in bps
    """
    noise = cfg.street_obs_noise * cfg.street_lean_vol_bps * rng.standard_normal()
    return true_lean + noise


def compute_street_lean_impact(
    street_lean_bps: float,
) -> float:
    """
    Compute how street lean affects competitor quotes.

    When street is long (positive lean), dealers quote lower to sell.
    When street is short (negative lean), dealers quote higher to buy.
    The effect is symmetric: both bid and offer sides shift in the same direction.

    Args:
        street_lean_bps: Current street lean in bps

    Returns:
        Adjustment to add to dealer quotes in bps
    """
    return -street_lean_bps * 0.5  # Partial pass-through


class StreetLeanProcess:
    """OU process for street lean dynamics."""

    def __init__(self, cfg: SimConfig, rng: Generator):
        self._cfg = cfg
        self._rng = rng
        self._dt = cfg.dt_minutes / cfg.minutes_per_day  # In days
        self._theta = cfg.street_lean_mean_rev
        self._sigma = cfg.street_lean_vol_bps
        self._b_eq = cfg.street_lean_eq
        # Initialize from equilibrium
        self._lean = self._b_eq

    @property
    def value(self) -> float:
        """Current street lean in bps."""
        return self._lean

    def step(self, dt: float | None = None) -> float:
        """Advance the OU process by one time step and return new value."""
        if dt is None:
            dt = self._dt
        drift = self._theta * (self._b_eq - self._lean) * dt
        diffusion = self._sigma * np.sqrt(dt) * self._rng.standard_normal()
        self._lean = self._lean + drift + diffusion
        return self._lean

    def reset(self, initial_value: float | None = None) -> None:
        """Reset to specified value or equilibrium."""
        self._lean = initial_value if initial_value is not None else self._b_eq
