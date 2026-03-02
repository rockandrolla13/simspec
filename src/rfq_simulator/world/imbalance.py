"""
AR(1) buy/sell imbalance process.

Implements latent imbalance process:
    imbalance_t = ρ · imbalance_{t-1} + (1 - ρ) · μ_regime + σ · ε_t

Buy probability derived from latent:
    P(buy) = clip(0.5 + imbalance_t, clip_low, clip_high)

References:
- Chordia, Roll, Subrahmanyam (2002) "Order Imbalance, Liquidity, and Market Returns"
- O'Hara & Zhou (2021) "Anatomy of a Liquidity Crisis"
"""

import numpy as np
from numpy.random import Generator

from ..config import ImbalanceConfig
from .regime import Regime


class ImbalanceProcess:
    """AR(1) flow imbalance process with regime-dependent drift."""

    def __init__(self, cfg: ImbalanceConfig, rng: Generator, initial_regime: Regime = Regime.CALM):
        self._cfg = cfg
        self._rng = rng
        self._regime = initial_regime
        self._value = 0.0

    @property
    def value(self) -> float:
        """Current imbalance value."""
        return self._value

    @property
    def regime(self) -> Regime:
        """Current regime."""
        return self._regime

    def set_regime(self, regime: Regime) -> None:
        """Update the regime (affects mean reversion target)."""
        self._regime = regime

    def step(self, dt: float = 1.0) -> float:
        """
        Advance AR(1) process by one step.

        Note: dt is ignored for AR(1) - process steps discretely per RFQ.
        """
        mu = self._cfg.mu_calm if self._regime == Regime.CALM else self._cfg.mu_stressed

        self._value = (
            self._cfg.rho * self._value +
            (1 - self._cfg.rho) * mu +
            self._cfg.sigma * self._rng.standard_normal()
        )
        return self._value

    def get_buy_probability(self) -> float:
        """Get current probability of client buy."""
        return np.clip(
            0.5 + self._value,
            self._cfg.clip_low,
            self._cfg.clip_high
        )

    def sample_direction(self) -> bool:
        """Sample whether client is buying (True) or selling (False)."""
        return self._rng.random() < self.get_buy_probability()

    def reset(self, initial_value: float | None = None) -> None:
        """Reset to initial state."""
        self._value = initial_value if initial_value is not None else 0.0


def compute_expected_buy_fraction(cfg: ImbalanceConfig, regime: Regime) -> float:
    """
    Compute expected buy fraction for a given regime.

    In stationary state:
        E[imbalance] = μ_regime (since ρ < 1)
        E[P(buy)] ≈ 0.5 + μ_regime

    Args:
        cfg: ImbalanceConfig
        regime: Current regime

    Returns:
        Expected buy fraction
    """
    mu = cfg.mu_calm if regime == Regime.CALM else cfg.mu_stressed
    return np.clip(0.5 + mu, cfg.clip_low, cfg.clip_high)
