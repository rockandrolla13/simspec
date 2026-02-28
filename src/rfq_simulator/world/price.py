"""
Price path generation: Arithmetic Brownian motion with mean reversion and intraday volatility.

Implements Eq 1-3 from spec:
    p_{t+dt} = p_t + κ(p̄ - p_t)dt + σ(h)√dt ε_t
    σ(h) = σ_base * (1 + v_open*e^{-(h-h0)²/τ²} + v_close*e^{-(h-hc)²/τ²})
    σ_per_step = σ_daily_bps/10000 * p0 * sqrt(dt_min/480)

Also handles adverse move mutations (Eq 25):
    p_{t+} = p_t + direction * τ_c * σ_adverse * |Z|
"""

import numpy as np
from numpy.random import Generator

from ..config import SimConfig
from .clock import TimeGrid


def compute_intraday_vol_multiplier(hour: float, cfg: SimConfig) -> float:
    """
    Compute the intraday volatility multiplier for a given hour of day.

    Eq 2: σ(h) = σ_base * (1 + v_open*e^{-(h-h0)²/τ²} + v_close*e^{-(h-hc)²/τ²})

    This produces a U-shaped pattern with elevated volatility at open and close.

    Args:
        hour: Hour within the trading day (0 to trading_hours)
        cfg: SimConfig with v_open, v_close, tau_v_hours, trading_hours

    Returns:
        Multiplier >= 1.0
    """
    h0 = 0.0  # Market open
    hc = cfg.trading_hours  # Market close
    tau_sq = cfg.tau_v_hours ** 2

    open_bump = cfg.v_open * np.exp(-(hour - h0) ** 2 / tau_sq)
    close_bump = cfg.v_close * np.exp(-(hour - hc) ** 2 / tau_sq)

    return 1.0 + open_bump + close_bump


def generate_price_path(cfg: SimConfig, rng: Generator) -> np.ndarray:
    """
    Generate the true bond mid-price path.

    Implements Eq 1-3: ABM with mean reversion and intraday vol seasonality.

    Args:
        cfg: SimConfig with all price process parameters
        rng: NumPy random generator for reproducibility

    Returns:
        np.ndarray of shape (n_steps + 1,) with prices[0] = p0
    """
    n_steps = cfg.n_steps
    time_grid = TimeGrid(cfg)

    # Pre-allocate price array (n_steps + 1 to include initial price)
    prices = np.zeros(n_steps + 1)
    prices[0] = cfg.p0

    # Base volatility per step (Eq 3)
    sigma_base = cfg.sigma_per_step

    # Mean reversion per step
    kappa = cfg.kappa_per_step

    # Pre-compute intraday vol multipliers for each step
    vol_multipliers = np.array([
        compute_intraday_vol_multiplier(time_grid.step_to_hour_of_day(i), cfg)
        for i in range(n_steps)
    ])

    # Draw all noise at once for efficiency
    eps = rng.standard_normal(n_steps)

    # Sequential price evolution (mean reversion prevents full vectorization)
    for i in range(n_steps):
        drift = kappa * (cfg.p_bar - prices[i])
        diffusion = sigma_base * vol_multipliers[i] * eps[i]
        prices[i + 1] = prices[i] + drift + diffusion

    return prices


def apply_adverse_move(
    prices: np.ndarray,
    step_idx: int,
    direction: int,
    toxicity: float,
    cfg: SimConfig,
    rng: Generator,
) -> float:
    """
    Apply a post-fill adverse move to the price path (mutation).

    Implements Eq 25: p_{t+} = p_t + direction * τ_c * σ_adverse * |Z|

    The jump is applied as a permanent level shift to all future prices.

    Args:
        prices: Price path array to mutate (modified in-place!)
        step_idx: Price step index at which fill occurred
        direction: +1 if client bought (price moves up), -1 if client sold
        toxicity: Client toxicity τ_c ∈ [0, 1]
        cfg: SimConfig with adverse_move_bps, p0
        rng: Random generator

    Returns:
        The jump magnitude applied (for logging)
    """
    # Convert adverse_move_bps to dollar terms
    sigma_adverse = cfg.adverse_move_bps / 10000.0 * cfg.p0

    # Draw absolute noise (half-normal)
    z = abs(rng.standard_normal())

    # Compute jump
    jump = direction * toxicity * sigma_adverse * z

    # Apply as permanent level shift to all future prices
    if step_idx + 1 < len(prices):
        prices[step_idx + 1:] += jump

    return jump


def get_price_at_minute(prices: np.ndarray, minute: float, time_grid: TimeGrid) -> float:
    """
    Get the price at a given continuous time (uses most recent step).

    Args:
        prices: Price path array
        minute: Time in minutes
        time_grid: TimeGrid for conversion

    Returns:
        Price at the most recent step before/at minute
    """
    step = time_grid.minute_to_step(minute)
    return prices[step]


def compute_realized_volatility(prices: np.ndarray, cfg: SimConfig) -> float:
    """
    Compute annualized realized volatility from a price path.

    Useful for validation against target sigma_daily_bps.

    Args:
        prices: Price path array
        cfg: SimConfig with dt_minutes, trading_hours

    Returns:
        Daily volatility in bps
    """
    log_returns = np.diff(np.log(prices))

    # Steps per day
    steps_per_day = cfg.n_steps_per_day

    # Daily return variance (sum of step variances)
    daily_var = np.var(log_returns) * steps_per_day

    # Convert to bps
    daily_vol_bps = np.sqrt(daily_var) * 10000

    return daily_vol_bps
