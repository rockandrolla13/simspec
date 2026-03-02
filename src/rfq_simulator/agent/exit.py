"""
Hybrid exit logic: Patient RFQ-based trading transitioning to aggressive crossing.

Implements Eq 26-27 from spec:

Phase 1 (Patient): t < t_signal + H - Δt_aggress
    - Normal RFQ trading with natural lean
    - Lean increases with urgency factor
    - Try to unwind via favorable RFQ fills

Phase 2 (Aggressive): t >= t_signal + H - Δt_aggress
    - Unwind remaining position via aggression
    - Constant unwind rate over remaining time
    - Pay half-spread + market impact

The transition point Δt_aggress is configurable (default: 8 hours = 1 day).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..config import SimConfig
from ..core.state import ExitMode


@dataclass
class ExitDecision:
    """
    Decision from the exit manager.

    Attributes:
        mode: Current exit mode
        should_aggress: Whether to execute an aggressive trade NOW
        aggress_size: Size to aggress (signed: + buy, - sell)
        reason: Why this decision was made
    """

    mode: ExitMode
    should_aggress: bool
    aggress_size: float
    reason: str


class HybridExitManager:
    """
    Manages the hybrid exit logic.

    Tracks time relative to signal expiry and determines when/how much
    to switch from patient RFQ-based exit to aggressive crossing.
    """

    def __init__(self, cfg: SimConfig):
        """
        Initialize the exit manager.

        Args:
            cfg: SimConfig with aggress_window_hours, aggress_halfspread_bps, etc.
        """
        self.cfg = cfg

        # Precompute aggress window in minutes
        self.aggress_window_minutes = cfg.aggress_window_hours * 60.0

        # Track state
        self.mode = ExitMode.PATIENT
        self.last_aggress_time: Optional[float] = None
        self.aggress_interval_minutes = 5.0  # How often to aggress in aggressive mode

    def check_exit_mode(
        self,
        current_minute: float,
        t_signal: float,
        horizon_minutes: float,
    ) -> ExitMode:
        """
        Check and update the current exit mode.

        Args:
            current_minute: Current time
            t_signal: When current signal was generated
            horizon_minutes: Signal horizon H in minutes

        Returns:
            Current ExitMode
        """
        signal_expiry = t_signal + horizon_minutes
        time_to_expiry = signal_expiry - current_minute

        if time_to_expiry <= self.aggress_window_minutes:
            self.mode = ExitMode.AGGRESSIVE
        else:
            self.mode = ExitMode.PATIENT

        return self.mode

    def get_exit_decision(
        self,
        current_minute: float,
        current_q: float,
        t_signal: float,
        horizon_minutes: float,
    ) -> ExitDecision:
        """
        Get the exit decision for current time.

        Args:
            current_minute: Current time
            current_q: Current inventory in lots
            t_signal: Signal generation time
            horizon_minutes: Signal horizon

        Returns:
            ExitDecision with action to take
        """
        mode = self.check_exit_mode(current_minute, t_signal, horizon_minutes)

        # In patient mode, no aggressive action
        if mode == ExitMode.PATIENT:
            return ExitDecision(
                mode=mode,
                should_aggress=False,
                aggress_size=0.0,
                reason="patient_mode",
            )

        # In aggressive mode, check if we need to unwind
        if abs(current_q) < 0.01:
            return ExitDecision(
                mode=mode,
                should_aggress=False,
                aggress_size=0.0,
                reason="position_flat",
            )

        # Check if it's time for an aggressive trade
        if self.last_aggress_time is not None:
            time_since_last = current_minute - self.last_aggress_time
            if time_since_last < self.aggress_interval_minutes:
                return ExitDecision(
                    mode=mode,
                    should_aggress=False,
                    aggress_size=0.0,
                    reason="too_soon",
                )

        # Compute size to aggress
        signal_expiry = t_signal + horizon_minutes
        time_remaining = max(0.1, signal_expiry - current_minute)

        # Unwind at constant rate over remaining time
        unwind_rate = abs(current_q) / (time_remaining / self.aggress_interval_minutes)

        # Clip to reasonable bounds
        unwind_rate = min(unwind_rate, abs(current_q))  # Don't overshoot
        unwind_rate = max(unwind_rate, 0.5)  # Minimum trade size

        # Direction: if long (q > 0), sell (negative); if short, buy
        aggress_size = -np.sign(current_q) * min(unwind_rate, abs(current_q))

        self.last_aggress_time = current_minute

        return ExitDecision(
            mode=mode,
            should_aggress=True,
            aggress_size=aggress_size,
            reason="aggressive_unwind",
        )

    def compute_aggress_cost(self, size: float) -> float:
        """
        Compute the cost of an aggressive trade.

        Cost = |size| * lot_size * (c_aggress + c_impact * √|size|) / 10000 * p0

        Args:
            size: Trade size in lots (signed)

        Returns:
            Total cost in dollars (always positive)
        """
        abs_size = abs(size)
        if abs_size < 0.01:
            return 0.0

        # Half-spread cost
        spread_cost = abs_size * self.cfg.aggress_halfspread_bps / 10000.0 * self.cfg.p0

        # Market impact (scales with sqrt of size)
        impact_cost = self.cfg.aggress_impact_bps * np.sqrt(abs_size) / 10000.0 * self.cfg.p0

        # Scale by lot size
        total = (spread_cost + impact_cost) * self.cfg.lot_size_mm * 10000

        return total

    def reset(self) -> None:
        """Reset the exit manager for a new signal."""
        self.mode = ExitMode.PATIENT
        self.last_aggress_time = None


def compute_urgency_adjusted_lean(
    base_lean: float,
    current_minute: float,
    t_signal: float,
    horizon_minutes: float,
    cfg: SimConfig,
) -> float:
    """
    Adjust lean based on urgency (time to signal expiry).

    As we approach expiry, increase lean to encourage faster convergence.

    Args:
        base_lean: Lean from compute_lean()
        current_minute: Current time
        t_signal: Signal generation time
        horizon_minutes: Signal horizon
        cfg: SimConfig

    Returns:
        Urgency-adjusted lean
    """
    signal_expiry = t_signal + horizon_minutes
    time_to_expiry = signal_expiry - current_minute
    time_elapsed = current_minute - t_signal

    if horizon_minutes <= 0:
        return base_lean

    # Urgency increases linearly from 1.0 to (1 + kappa_urgency)
    urgency_factor = 1.0 + cfg.kappa_urgency * (time_elapsed / horizon_minutes)
    urgency_factor = min(urgency_factor, 1.0 + cfg.kappa_urgency)  # Cap at max

    return base_lean * urgency_factor


def estimate_time_to_unwind(
    current_q: float,
    favorable_rfq_rate: float,
    win_rate: float,
    avg_rfq_size: float,
) -> float:
    """
    Estimate time needed to unwind position via RFQs.

    Useful for deciding when to switch from patient to aggressive.

    Args:
        current_q: Current position in lots
        favorable_rfq_rate: Rate of favorable-direction RFQs per minute
        win_rate: Estimated win rate on favorable RFQs
        avg_rfq_size: Average RFQ size in lots

    Returns:
        Estimated unwind time in minutes
    """
    if favorable_rfq_rate <= 0 or win_rate <= 0 or avg_rfq_size <= 0:
        return float("inf")

    # Expected fills per minute
    fills_per_minute = favorable_rfq_rate * win_rate

    # Expected volume per minute
    volume_per_minute = fills_per_minute * avg_rfq_size

    if volume_per_minute <= 0:
        return float("inf")

    return abs(current_q) / volume_per_minute
