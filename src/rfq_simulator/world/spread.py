"""
Log-normal regime-dependent spread distribution.

Base spread sampled from:
    log(spread) ~ Normal(mu_regime, sigma_regime)
    spread = exp(log(spread))

Size adjustment (concave, per Edwards et al. 2007):
    spread_adj = spread_base * (1 + gamma * log(1 + size))

References:
- Bao, Pan, Wang (2011) "The Illiquidity of Corporate Bonds"
- Dick-Nielsen, Feldhutter, Lando (2012) "Corporate Bond Liquidity"
- Edwards, Harris, Piwowar (2007) "Corporate Bond Transaction Costs"
"""

from typing import Tuple
import numpy as np
from numpy.random import Generator

from ..config import SpreadConfig
from .regime import Regime


def get_regime_spread_params(regime: Regime, cfg: SpreadConfig) -> Tuple[float, float]:
    """
    Get log-normal parameters for the given regime.

    Args:
        regime: Current market regime
        cfg: SpreadConfig with mu and sigma for each regime

    Returns:
        (mu, sigma) tuple for log-normal distribution
    """
    if regime == Regime.CALM:
        return cfg.mu_calm, cfg.sigma_calm
    else:
        return cfg.mu_stressed, cfg.sigma_stressed


def sample_base_spread(regime: Regime, cfg: SpreadConfig, rng: Generator) -> float:
    """
    Sample base spread from regime-dependent log-normal.

    Args:
        regime: Current market regime
        cfg: SpreadConfig
        rng: Random generator

    Returns:
        Base spread in basis points
    """
    mu, sigma = get_regime_spread_params(regime, cfg)
    log_spread = mu + sigma * rng.standard_normal()
    return np.exp(log_spread)


def apply_size_adjustment(spread: float, size: int, cfg: SpreadConfig) -> float:
    """
    Apply concave size adjustment to spread.

    Eq: spread_adj = spread * (1 + gamma * log(1 + size))

    Args:
        spread: Base spread
        size: Trade size in lots
        cfg: SpreadConfig with size_gamma

    Returns:
        Size-adjusted spread
    """
    size_mult = 1 + cfg.size_gamma * np.log(1 + size)
    return spread * size_mult


def sample_dealer_spread(
    regime: Regime,
    size: int,
    cfg: SpreadConfig,
    rng: Generator,
) -> float:
    """
    Sample a complete dealer spread with size adjustment.

    Main entry point for spread sampling.

    Args:
        regime: Current market regime
        size: Trade size in lots
        cfg: SpreadConfig
        rng: Random generator

    Returns:
        Complete spread in basis points
    """
    base = sample_base_spread(regime, cfg, rng)
    return apply_size_adjustment(base, size, cfg)


def compute_expected_spread(regime: Regime, cfg: SpreadConfig) -> float:
    """
    Compute expected spread for a given regime.

    For log-normal: E[X] = exp(mu + sigma^2/2)

    Args:
        regime: Market regime
        cfg: SpreadConfig

    Returns:
        Expected spread in basis points
    """
    mu, sigma = get_regime_spread_params(regime, cfg)
    return np.exp(mu + sigma**2 / 2)


def compute_median_spread(regime: Regime, cfg: SpreadConfig) -> float:
    """
    Compute median spread for a given regime.

    For log-normal: median = exp(mu)

    Args:
        regime: Market regime
        cfg: SpreadConfig

    Returns:
        Median spread in basis points
    """
    mu, _ = get_regime_spread_params(regime, cfg)
    return np.exp(mu)
