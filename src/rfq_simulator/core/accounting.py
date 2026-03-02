"""
P&L Accounting: Decomposition into alpha, spread, carry, and aggressor components.

Implements Eq 35-40 from spec:
    PnL_total = PnL_alpha + PnL_spread + PnL_carry + PnL_hedge - Cost_aggress

Components:
    PnL_alpha = Σ q_{t-1} * (p_t - p_{t-1}) * lot_size
    PnL_spread = Σ (p_fill - p_true) * signed_size * lot_size
    PnL_carry = Σ_days q * lot_size * coupon_bps / 360
    Cost_aggress = Σ spread_paid + impact_cost
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..config import SimConfig


@dataclass
class PnLDecomposition:
    """
    P&L decomposition at a point in time.

    All values in dollars unless otherwise noted.
    """

    # Core components
    alpha_pnl: float = 0.0
    """P&L from being correctly positioned (mark-to-market)."""

    spread_pnl: float = 0.0
    """Edge earned at execution (fill_price - true_price)."""

    carry_pnl: float = 0.0
    """Coupon accrual on inventory."""

    hedge_pnl: float = 0.0
    """P&L from hedge instruments."""

    aggress_cost: float = 0.0
    """Cost of aggressive execution (spread + impact)."""

    # Total
    @property
    def total_pnl(self) -> float:
        """Total P&L."""
        return (
            self.alpha_pnl + self.spread_pnl + self.carry_pnl + self.hedge_pnl
            - self.aggress_cost
        )

    # Normalized metrics
    lot_size_mm: float = 1.0
    """Lot size for normalization."""

    p0: float = 100.0
    """Reference price (per $100 face) for bps conversion."""

    @property
    def total_pnl_bps(self) -> float:
        """Total P&L in bps of one-lot notional at p0."""
        notional = self.p0 * self.lot_size_mm * 10000  # dollar value of 1 lot
        if notional <= 0:
            return 0.0
        return self.total_pnl / notional * 10000

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            "alpha_pnl": self.alpha_pnl,
            "spread_pnl": self.spread_pnl,
            "carry_pnl": self.carry_pnl,
            "hedge_pnl": self.hedge_pnl,
            "aggress_cost": self.aggress_cost,
            "total_pnl": self.total_pnl,
        }


@dataclass
class PnLTracker:
    """
    Tracks P&L components over the simulation.

    Updates incrementally as events occur.
    """

    cfg: SimConfig

    # Running totals
    alpha_pnl: float = 0.0
    spread_pnl: float = 0.0
    carry_pnl: float = 0.0
    hedge_pnl: float = 0.0
    aggress_cost: float = 0.0

    # Time series for analysis
    alpha_pnl_series: List[float] = field(default_factory=list)
    spread_pnl_series: List[float] = field(default_factory=list)
    carry_pnl_series: List[float] = field(default_factory=list)
    total_pnl_series: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    # Previous state for delta calculations
    _last_q: float = 0.0
    _last_price: float = 0.0

    def record_price_move(
        self, current_q: float, new_price: float, time: float
    ) -> float:
        """
        Record alpha P&L from a price move.

        PnL_alpha = q_{t-1} * (p_t - p_{t-1}) * lot_size * 10000

        Args:
            current_q: Inventory BEFORE this price step
            new_price: New price after the move
            time: Current time in minutes

        Returns:
            Incremental alpha P&L
        """
        if self._last_price == 0:
            self._last_price = new_price
            return 0.0

        price_change = new_price - self._last_price
        alpha_increment = (
            self._last_q * price_change * self.cfg.lot_size_mm * 10000
        )

        self.alpha_pnl += alpha_increment
        self._last_price = new_price
        self._last_q = current_q

        return alpha_increment

    def record_spread(
        self, fill_price: float, true_price: float, signed_size: float
    ) -> float:
        """
        Record spread P&L from a fill.

        PnL_spread = (p_fill - p_true) * signed_size * lot_size * 10000

        For client buy (we sell): signed_size > 0
        For client sell (we buy): signed_size < 0

        Edge is positive when we fill better than true price.

        Args:
            fill_price: Price we traded at
            true_price: True market price at fill time
            signed_size: Signed trade size (+ if we sell, - if we buy)

        Returns:
            Spread P&L from this fill
        """
        spread_increment = (
            (fill_price - true_price) * signed_size * self.cfg.lot_size_mm * 10000
        )
        self.spread_pnl += spread_increment
        return spread_increment

    def record_carry(self, avg_q: float, days: float = 1.0) -> float:
        """
        Record carry P&L from holding inventory.

        PnL_carry = q * lot_size * coupon_bps / 10000 * (days / 360)

        Simplified: assume accrual on average daily position.

        Args:
            avg_q: Average inventory over the period
            days: Number of days

        Returns:
            Carry P&L
        """
        # Daily accrual rate
        daily_carry = (
            self.cfg.coupon_bps / 10000.0 / 360.0 * self.cfg.lot_size_mm * 10000
        )
        carry_increment = avg_q * daily_carry * days
        self.carry_pnl += carry_increment
        return carry_increment

    def record_hedge_pnl(self, hedge_pnl: float) -> None:
        """Record P&L from hedge instruments."""
        self.hedge_pnl += hedge_pnl

    def record_aggress_cost(
        self, size: float, half_spread_bps: float, impact_bps: float
    ) -> float:
        """
        Record cost of aggressive execution.

        Cost = size * lot_size * (half_spread + impact * sqrt(size)) / 10000 * price

        Args:
            size: Trade size in lots
            half_spread_bps: Half spread cost in bps
            impact_bps: Impact cost per sqrt(lot) in bps

        Returns:
            Total aggression cost
        """
        # Half spread cost
        spread_cost = size * half_spread_bps / 10000.0 * self.cfg.p0
        spread_cost *= self.cfg.lot_size_mm * 10000

        # Impact cost (scales with sqrt of size)
        impact_cost = impact_bps * np.sqrt(size) / 10000.0 * self.cfg.p0
        impact_cost *= self.cfg.lot_size_mm * 10000

        total_cost = spread_cost + impact_cost
        self.aggress_cost += total_cost

        return total_cost

    def snapshot(self, time: float) -> None:
        """
        Take a snapshot of current P&L for time series.

        Args:
            time: Current time in minutes
        """
        self.timestamps.append(time)
        self.alpha_pnl_series.append(self.alpha_pnl)
        self.spread_pnl_series.append(self.spread_pnl)
        self.carry_pnl_series.append(self.carry_pnl)
        self.total_pnl_series.append(self.total_pnl)

    @property
    def total_pnl(self) -> float:
        """Current total P&L."""
        return (
            self.alpha_pnl + self.spread_pnl + self.carry_pnl + self.hedge_pnl
            - self.aggress_cost
        )

    def get_decomposition(self) -> PnLDecomposition:
        """Get current P&L decomposition."""
        return PnLDecomposition(
            alpha_pnl=self.alpha_pnl,
            spread_pnl=self.spread_pnl,
            carry_pnl=self.carry_pnl,
            hedge_pnl=self.hedge_pnl,
            aggress_cost=self.aggress_cost,
            lot_size_mm=self.cfg.lot_size_mm,
            p0=self.cfg.p0,
        )

    def get_time_series(self) -> dict:
        """Get P&L time series as arrays."""
        return {
            "time": np.array(self.timestamps),
            "alpha_pnl": np.array(self.alpha_pnl_series),
            "spread_pnl": np.array(self.spread_pnl_series),
            "carry_pnl": np.array(self.carry_pnl_series),
            "total_pnl": np.array(self.total_pnl_series),
        }

    def reset(self) -> None:
        """Reset all P&L tracking."""
        self.alpha_pnl = 0.0
        self.spread_pnl = 0.0
        self.carry_pnl = 0.0
        self.hedge_pnl = 0.0
        self.aggress_cost = 0.0
        self.alpha_pnl_series.clear()
        self.spread_pnl_series.clear()
        self.carry_pnl_series.clear()
        self.total_pnl_series.clear()
        self.timestamps.clear()
        self._last_q = 0.0
        self._last_price = 0.0


def compute_sharpe_ratio(pnl_series: np.ndarray, annualization: float = 252.0) -> float:
    """
    Compute annualized Sharpe ratio from daily P&L.

    Args:
        pnl_series: Array of daily P&L values
        annualization: Annualization factor (252 for daily)

    Returns:
        Annualized Sharpe ratio
    """
    if len(pnl_series) < 2:
        return 0.0

    daily_returns = np.diff(pnl_series)
    if np.std(daily_returns) == 0:
        return 0.0

    return np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(annualization)


def compute_max_drawdown(pnl_series: np.ndarray) -> float:
    """
    Compute maximum drawdown from P&L series.

    Args:
        pnl_series: Cumulative P&L series

    Returns:
        Maximum drawdown (positive value)
    """
    if len(pnl_series) < 2:
        return 0.0

    running_max = np.maximum.accumulate(pnl_series)
    drawdowns = running_max - pnl_series

    return np.max(drawdowns)


def compute_alpha_capture_ratio(alpha_pnl: float, theoretical_alpha: float) -> float:
    """
    Compute alpha capture ratio (ACR).

    ACR = actual alpha P&L / theoretical alpha P&L

    Args:
        alpha_pnl: Actual alpha P&L achieved
        theoretical_alpha: P&L from perfect execution at target position

    Returns:
        Alpha capture ratio (0 to 1+)
    """
    if theoretical_alpha == 0:
        return 0.0

    return alpha_pnl / theoretical_alpha
