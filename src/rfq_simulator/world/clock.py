"""
TimeGrid: Utilities for converting between continuous time (minutes) and discrete steps.

The simulation operates on two time scales:
1. Continuous minutes: RFQ arrivals, signal refresh, exit timing
2. Discrete steps: Price path indexing (every dt_minutes)

This module provides consistent conversion between them.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..config import SimConfig


@dataclass
class TimeGrid:
    """
    Handles time conversion between continuous minutes and discrete price steps.

    Attributes:
        cfg: SimConfig with dt_minutes, trading_hours, T_days
    """

    cfg: SimConfig

    @property
    def dt(self) -> float:
        """Price step size in minutes."""
        return self.cfg.dt_minutes

    @property
    def minutes_per_day(self) -> float:
        """Trading minutes per day."""
        return self.cfg.minutes_per_day

    @property
    def steps_per_day(self) -> int:
        """Price steps per trading day."""
        return self.cfg.n_steps_per_day

    @property
    def total_steps(self) -> int:
        """Total price steps in simulation."""
        return self.cfg.n_steps

    @property
    def total_minutes(self) -> float:
        """Total simulation time in minutes."""
        return self.cfg.total_minutes

    def minute_to_step(self, minute: float) -> int:
        """
        Convert continuous minute to discrete price step index.

        Uses floor to get the most recent price step.

        Args:
            minute: Time in minutes from simulation start

        Returns:
            Price step index (0 to n_steps-1), clamped to valid range
        """
        step = int(minute / self.dt)
        return max(0, min(step, self.total_steps - 1))

    def step_to_minute(self, step: int) -> float:
        """
        Convert discrete price step to continuous minute (start of step).

        Args:
            step: Price step index

        Returns:
            Time in minutes from simulation start
        """
        return step * self.dt

    def minute_to_day(self, minute: float) -> int:
        """
        Get the trading day number (0-indexed) for a given minute.

        Args:
            minute: Time in minutes from simulation start

        Returns:
            Trading day (0 to T_days-1)
        """
        return int(minute / self.minutes_per_day)

    def minute_to_hour_of_day(self, minute: float) -> float:
        """
        Get the hour within the trading day for a given minute.

        Args:
            minute: Time in minutes from simulation start

        Returns:
            Hour within day (0.0 to trading_hours)
        """
        minute_of_day = minute % self.minutes_per_day
        return minute_of_day / 60.0

    def step_to_hour_of_day(self, step: int) -> float:
        """
        Get the hour within the trading day for a given price step.

        Args:
            step: Price step index

        Returns:
            Hour within day (0.0 to trading_hours)
        """
        minute = self.step_to_minute(step)
        return self.minute_to_hour_of_day(minute)

    def step_to_day(self, step: int) -> int:
        """
        Get the trading day number for a given price step.

        Args:
            step: Price step index

        Returns:
            Trading day (0 to T_days-1)
        """
        return step // self.steps_per_day

    def is_new_day(self, minute: float, prev_minute: float) -> bool:
        """
        Check if we've crossed into a new trading day.

        Args:
            minute: Current time in minutes
            prev_minute: Previous time in minutes

        Returns:
            True if day changed
        """
        return self.minute_to_day(minute) > self.minute_to_day(prev_minute)

    def day_start_minute(self, day: int) -> float:
        """
        Get the starting minute of a trading day.

        Args:
            day: Trading day (0-indexed)

        Returns:
            Minute at start of day
        """
        return day * self.minutes_per_day

    def day_end_minute(self, day: int) -> float:
        """
        Get the ending minute of a trading day.

        Args:
            day: Trading day (0-indexed)

        Returns:
            Minute at end of day
        """
        return (day + 1) * self.minutes_per_day

    def day_range(self, day: int) -> Tuple[float, float]:
        """
        Get the minute range [start, end) for a trading day.

        Args:
            day: Trading day (0-indexed)

        Returns:
            (start_minute, end_minute) tuple
        """
        return self.day_start_minute(day), self.day_end_minute(day)

    def is_valid_minute(self, minute: float) -> bool:
        """Check if minute is within simulation bounds."""
        return 0 <= minute < self.total_minutes

    def is_valid_step(self, step: int) -> bool:
        """Check if step is within price path bounds."""
        return 0 <= step < self.total_steps


def compute_intraday_intensity(hour: float, cfg: SimConfig) -> float:
    """
    Compute the intraday RFQ intensity multiplier.

    Eq 10: f(h) = 1 + A_open * e^{-(h-h0)^2/tau^2} + A_close * e^{-(h-hc)^2/tau^2}

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
