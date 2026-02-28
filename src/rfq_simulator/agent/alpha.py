"""
Alpha signal: Noisy forecast of future price changes with regime-dependent IC.

Implements Eq 4-7 from spec:
    α* = p_{t+H} - p_t                              (perfect foresight)
    α = ρ(r) * α* + √(1-ρ²) * σ_α * η              (noisy observation)
    α_rem(t) = α * max(0, (t_signal + H - t) / H)   (linear decay)

The signal refreshes every signal_refresh_hours.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.random import Generator

from ..config import SimConfig
from ..world.clock import TimeGrid
from ..world.regime import Regime, get_effective_ic


@dataclass
class AlphaSignal:
    """
    A single alpha signal instance.

    Attributes:
        alpha: The noisy alpha value (in dollars)
        alpha_star: The perfect-foresight alpha (for diagnostics)
        t_signal: Time (minutes) when signal was generated
        horizon_minutes: Signal horizon H in minutes
        regime: Regime when signal was generated
    """

    alpha: float
    alpha_star: float
    t_signal: float
    horizon_minutes: float
    regime: Regime

    def remaining_alpha(self, current_minute: float) -> float:
        """
        Compute remaining alpha at current time (linear decay).

        Eq: α_rem(t) = α * max(0, (t_signal + H - t) / H)

        Args:
            current_minute: Current time in minutes

        Returns:
            Remaining alpha value (decays to 0 at expiry)
        """
        time_remaining = self.t_signal + self.horizon_minutes - current_minute
        decay_fraction = max(0.0, time_remaining / self.horizon_minutes)
        return self.alpha * decay_fraction

    def is_expired(self, current_minute: float) -> bool:
        """Check if signal has expired."""
        return current_minute >= self.t_signal + self.horizon_minutes

    def time_to_expiry(self, current_minute: float) -> float:
        """Get time remaining until signal expires (can be negative)."""
        return self.t_signal + self.horizon_minutes - current_minute


class AlphaSignalManager:
    """
    Manages alpha signal generation, refresh, and decay.

    The manager generates new signals at the start of each refresh period,
    using a lookahead into the (known) price path.
    """

    def __init__(self, cfg: SimConfig, time_grid: TimeGrid):
        """
        Initialize the alpha signal manager.

        Args:
            cfg: SimConfig with IC, alpha_horizon_days, signal_refresh_hours
            time_grid: TimeGrid for time conversions
        """
        self.cfg = cfg
        self.time_grid = time_grid
        self.current_signal: Optional[AlphaSignal] = None
        self.signal_history: list[AlphaSignal] = []

        # Precompute refresh interval in minutes
        self.refresh_minutes = cfg.signal_refresh_hours * 60.0
        self.horizon_minutes = cfg.alpha_horizon_days * cfg.minutes_per_day

    def should_refresh(self, current_minute: float) -> bool:
        """
        Check if a new signal should be generated.

        Refresh happens:
        1. At simulation start (no current signal)
        2. When refresh interval has passed since last signal

        Args:
            current_minute: Current time in minutes

        Returns:
            True if should generate new signal
        """
        if self.current_signal is None:
            return True

        time_since_signal = current_minute - self.current_signal.t_signal
        return time_since_signal >= self.refresh_minutes

    def generate_signal(
        self,
        current_minute: float,
        prices: np.ndarray,
        regime: Regime,
        rng: Generator,
    ) -> AlphaSignal:
        """
        Generate a new alpha signal using perfect foresight + noise.

        Eq 4-5:
            α* = p_{t+H} - p_t
            α = ρ(r) * α* + √(1-ρ²) * σ_α * η

        Args:
            current_minute: Time at which signal is generated
            prices: Full price path (simulator knows this)
            regime: Current market regime
            rng: Random generator

        Returns:
            New AlphaSignal
        """
        # Get price step indices
        current_step = self.time_grid.minute_to_step(current_minute)
        horizon_step = self.time_grid.minute_to_step(
            current_minute + self.horizon_minutes
        )

        # Clamp to valid range (at end of simulation, horizon may exceed path)
        horizon_step = min(horizon_step, len(prices) - 1)

        # Perfect foresight alpha (Eq 4)
        alpha_star = prices[horizon_step] - prices[current_step]

        # Compute σ_α from the path (std of forward returns at this horizon)
        # Simplified: use the expected std based on daily vol
        sigma_alpha = self._estimate_sigma_alpha(prices)

        # Get effective IC for regime (Eq 7)
        ic = get_effective_ic(regime, self.cfg)

        # Noisy alpha (Eq 5)
        eta = rng.standard_normal()
        noise_scale = np.sqrt(1 - ic ** 2) * sigma_alpha
        alpha = ic * alpha_star + noise_scale * eta

        # Create signal
        signal = AlphaSignal(
            alpha=alpha,
            alpha_star=alpha_star,
            t_signal=current_minute,
            horizon_minutes=self.horizon_minutes,
            regime=regime,
        )

        # Store
        self.current_signal = signal
        self.signal_history.append(signal)

        return signal

    def _estimate_sigma_alpha(self, prices: np.ndarray) -> float:
        """
        Estimate σ_α = Std(α*) from the price path.

        Uses the standard deviation of forward returns at the signal horizon.

        Args:
            prices: Price path

        Returns:
            Estimated standard deviation of alpha
        """
        # Number of steps in horizon
        horizon_steps = int(self.horizon_minutes / self.cfg.dt_minutes)

        if horizon_steps >= len(prices):
            # Fall back to daily vol scaling
            return self.cfg.sigma_daily_bps / 10000 * self.cfg.p0 * np.sqrt(
                self.cfg.alpha_horizon_days
            )

        # Compute forward returns
        forward_returns = prices[horizon_steps:] - prices[:-horizon_steps]

        return np.std(forward_returns) if len(forward_returns) > 1 else 0.0

    def get_remaining_alpha(self, current_minute: float) -> float:
        """
        Get the remaining alpha at current time.

        Args:
            current_minute: Current time in minutes

        Returns:
            Remaining alpha (0 if no signal or expired)
        """
        if self.current_signal is None:
            return 0.0
        return self.current_signal.remaining_alpha(current_minute)

    def get_signal_age(self, current_minute: float) -> float:
        """
        Get time since current signal was generated.

        Args:
            current_minute: Current time in minutes

        Returns:
            Age in minutes (0 if no signal)
        """
        if self.current_signal is None:
            return 0.0
        return current_minute - self.current_signal.t_signal

    def get_time_to_expiry(self, current_minute: float) -> float:
        """
        Get time until current signal expires.

        Args:
            current_minute: Current time in minutes

        Returns:
            Time to expiry in minutes (0 if no signal or expired)
        """
        if self.current_signal is None:
            return 0.0
        return max(0.0, self.current_signal.time_to_expiry(current_minute))
