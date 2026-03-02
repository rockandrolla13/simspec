"""
Observable mid and skew: What the trader actually sees.

Implements Eq 11-12 from spec:
    mid_obs = p_{t-lag} + σ_obs * ξ       (lagged + noisy)
    skew = ρ_s * (p_t - mid_obs) + √(1-ρ_s²) * σ_stale * ζ

The trader doesn't see the true price p_t directly. They observe:
1. A lagged composite mid (stale by Δt_lag minutes)
2. A skew signal that partially corrects the staleness

Their theoretical price is: theo = mid_obs + skew + lean
"""

import numpy as np
from numpy.random import Generator

from dataclasses import dataclass

from ..config import SimConfig
from ..world.clock import TimeGrid


@dataclass
class TheoResult:
    """Result of theoretical price computation."""
    theo: float
    mid_obs: float
    skew: float


def compute_observable_mid(
    prices: np.ndarray,
    current_minute: float,
    time_grid: TimeGrid,
    cfg: SimConfig,
    rng: Generator,
) -> float:
    """
    Compute the observable mid price (lagged + noisy).

    Eq 11: mid_obs = p_{t-lag} + σ_obs * ξ

    Args:
        prices: Full price path
        current_minute: Current time in minutes
        time_grid: TimeGrid for conversions
        cfg: SimConfig with obs_lag_minutes, obs_noise_bps
        rng: Random generator

    Returns:
        Observable mid price in dollars
    """
    # Lagged time
    lag_minute = max(0, current_minute - cfg.obs_lag_minutes)
    lag_step = time_grid.minute_to_step(lag_minute)

    # Lagged price
    p_lagged = prices[lag_step]

    # Add observation noise
    noise_bps = cfg.obs_noise_bps * rng.standard_normal()
    noise_dollars = noise_bps / 10000.0 * cfg.p0

    return p_lagged + noise_dollars


def compute_true_staleness(
    prices: np.ndarray,
    current_minute: float,
    time_grid: TimeGrid,
    cfg: SimConfig,
) -> float:
    """
    Compute the true staleness (p_t - p_{t-lag}).

    This is what the skew model tries to estimate.

    Args:
        prices: Full price path
        current_minute: Current time
        time_grid: TimeGrid
        cfg: SimConfig

    Returns:
        True staleness in dollars
    """
    current_step = time_grid.minute_to_step(current_minute)
    p_true = prices[current_step]

    lag_minute = max(0, current_minute - cfg.obs_lag_minutes)
    lag_step = time_grid.minute_to_step(lag_minute)
    p_lagged = prices[lag_step]

    return p_true - p_lagged


def compute_skew(
    prices: np.ndarray,
    mid_obs: float,
    current_minute: float,
    time_grid: TimeGrid,
    cfg: SimConfig,
    rng: Generator,
) -> float:
    """
    Compute the skew signal (staleness correction).

    Eq 12: skew = ρ_s * (p_t - mid_obs) + √(1-ρ_s²) * σ_stale * ζ

    The skew model has accuracy ρ_s in predicting the true staleness.

    Args:
        prices: Full price path
        mid_obs: Observable mid price
        current_minute: Current time
        time_grid: TimeGrid
        cfg: SimConfig with skew_accuracy
        rng: Random generator

    Returns:
        Skew signal in dollars
    """
    # Get true current price
    current_step = time_grid.minute_to_step(current_minute)
    p_true = prices[current_step]

    # True staleness
    true_staleness = p_true - mid_obs

    # Estimate typical staleness magnitude
    # Use daily vol * sqrt(lag/day) as proxy
    sigma_stale = (cfg.sigma_daily_bps / 10000.0) * cfg.p0 * np.sqrt(
        cfg.obs_lag_minutes / cfg.minutes_per_day
    )

    # Noisy skew signal
    rho_s = cfg.skew_accuracy
    noise_scale = np.sqrt(1 - rho_s ** 2) * sigma_stale
    noise = noise_scale * rng.standard_normal()

    skew = rho_s * true_staleness + noise

    return skew


def compute_theo_price(
    prices: np.ndarray,
    current_minute: float,
    current_q: float,
    target_q: float,
    time_grid: TimeGrid,
    lean: float,
    cfg: SimConfig,
    rng: Generator,
) -> TheoResult:
    """
    Compute the trader's theoretical price.

    theo = mid_obs + skew + lean

    Args:
        prices: Full price path
        current_minute: Current time
        current_q: Current inventory
        target_q: Target inventory
        time_grid: TimeGrid
        lean: Pre-computed lean adjustment
        cfg: SimConfig
        rng: Random generator

    Returns:
        TheoResult with theo, mid_obs, skew fields
    """
    mid_obs = compute_observable_mid(prices, current_minute, time_grid, cfg, rng)
    skew = compute_skew(prices, mid_obs, current_minute, time_grid, cfg, rng)
    theo = mid_obs + skew + lean

    return TheoResult(theo=theo, mid_obs=mid_obs, skew=skew)


def compute_theo_error(
    theo_price: float,
    prices: np.ndarray,
    current_minute: float,
    time_grid: TimeGrid,
) -> float:
    """
    Compute the error between theo and true price (for diagnostics).

    Args:
        theo_price: Trader's theoretical price
        prices: Full price path
        current_minute: Current time
        time_grid: TimeGrid

    Returns:
        Theo error in dollars (theo - true)
    """
    current_step = time_grid.minute_to_step(current_minute)
    p_true = prices[current_step]
    return theo_price - p_true


def estimate_skew_from_recent_moves(
    prices: np.ndarray,
    current_minute: float,
    time_grid: TimeGrid,
    cfg: SimConfig,
    lookback_minutes: float = 30.0,
) -> float:
    """
    Alternative skew estimation using momentum.

    Estimates staleness by extrapolating recent price moves.
    Can be used as an alternative to the information-based skew.

    Args:
        prices: Full price path
        current_minute: Current time
        time_grid: TimeGrid
        cfg: SimConfig
        lookback_minutes: Lookback for momentum calculation

    Returns:
        Momentum-based skew estimate in dollars
    """
    current_step = time_grid.minute_to_step(current_minute)
    lookback_step = time_grid.minute_to_step(
        max(0, current_minute - lookback_minutes)
    )

    if current_step == lookback_step:
        return 0.0

    # Recent momentum
    recent_move = prices[current_step] - prices[lookback_step]

    # Extrapolate for the lag period
    extrapolation_factor = cfg.obs_lag_minutes / lookback_minutes

    return recent_move * extrapolation_factor
