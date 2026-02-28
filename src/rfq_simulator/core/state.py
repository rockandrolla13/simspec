"""
SimulationState: Central state tracking for the simulator.

Maintains:
- Inventory position
- Cash balance
- Alpha signal state
- RFQ event log
- Various running statistics
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

from ..config import SimConfig
from ..world.regime import Regime
from ..world.rfq_stream import RFQEvent


class ExitMode(Enum):
    """Current exit mode for hybrid exit logic."""

    PATIENT = "patient"  # Normal RFQ-based trading
    AGGRESSIVE = "aggressive"  # Time to unwind via crossing


@dataclass
class RFQLog:
    """
    Log entry for a single RFQ event.

    Contains full details for post-simulation analysis.
    """

    # RFQ attributes
    time: float
    is_client_buy: bool
    size: int
    n_dealers: int
    toxicity: float

    # Prices
    p_true: float
    mid_obs: float
    skew: float
    lean: float
    theo: float

    # State at RFQ time
    q_before: float
    q_target: float
    alpha_remaining: float

    # Quote details
    quote_price: Optional[float]
    markup_bps: float
    win_prob_est: float
    expected_value: float
    declined: bool
    decline_reason: Optional[str]

    # Competition result
    filled: bool
    n_competitors_responded: int
    best_competitor_price: Optional[float]

    # Post-fill
    q_after: float
    spread_pnl: float  # Edge captured if filled
    adverse_move: float  # Price move after fill

    # Regime
    regime: Regime


@dataclass
class SimulationState:
    """
    Complete simulation state at any point in time.

    This is the central mutable state object passed through the event loop.
    """

    cfg: SimConfig

    # Position and cash
    q: float = 0.0
    """Current inventory in lots (positive = long)."""

    cash: float = 0.0
    """Cumulative cash from trades (price * size, signed)."""

    # Alpha signal state
    alpha: float = 0.0
    """Current alpha signal value."""

    alpha_star: float = 0.0
    """Perfect-foresight alpha (for diagnostics)."""

    alpha_remaining: float = 0.0
    """Remaining alpha after decay."""

    t_signal: float = 0.0
    """Time when current signal was generated."""

    q_target: float = 0.0
    """Current target position from alpha."""

    # Exit mode
    exit_mode: ExitMode = ExitMode.PATIENT
    """Current exit mode."""

    # Regime
    current_day: int = 0
    """Current trading day (0-indexed)."""

    current_regime: Regime = Regime.CALM
    """Current market regime."""

    # Statistics
    n_rfqs_seen: int = 0
    """Total RFQs processed."""

    n_rfqs_quoted: int = 0
    """RFQs where we submitted a quote."""

    n_rfqs_filled: int = 0
    """RFQs where we won the trade."""

    n_rfqs_declined: int = 0
    """RFQs we declined to quote."""

    total_volume: float = 0.0
    """Total traded volume in lots."""

    total_spread_pnl: float = 0.0
    """Cumulative spread P&L in dollars."""

    total_adverse_move: float = 0.0
    """Cumulative adverse selection cost."""

    # Event log
    rfq_log: List[RFQLog] = field(default_factory=list)
    """Full log of all RFQ events."""

    # Price tracking for MTM
    last_price: float = 0.0
    """Most recent observed true price (for MTM)."""

    def update_position(self, delta_q: float, price: float, size: int) -> float:
        """
        Update position from a fill.

        Args:
            delta_q: Signed inventory change (+1 if we buy, -1 if we sell)
            price: Fill price per lot
            size: Trade size in lots

        Returns:
            Cash flow from trade (negative if buying)
        """
        self.q += delta_q * size
        cash_flow = -delta_q * size * price * self.cfg.lot_size_mm * 10000
        self.cash += cash_flow
        self.total_volume += abs(size)
        return cash_flow

    def record_spread_pnl(self, spread: float) -> None:
        """Record spread P&L from a fill."""
        self.total_spread_pnl += spread

    def record_adverse_move(self, move: float) -> None:
        """Record adverse price move after fill."""
        self.total_adverse_move += abs(move)

    def mark_to_market(self, current_price: float) -> float:
        """
        Compute mark-to-market portfolio value.

        MTM = cash + q * price * lot_size * 10000

        Args:
            current_price: Current true price

        Returns:
            Total portfolio value in dollars
        """
        position_value = self.q * current_price * self.cfg.lot_size_mm * 10000
        return self.cash + position_value

    def get_fill_rate(self) -> float:
        """Get fill rate (fills / quotes submitted)."""
        if self.n_rfqs_quoted == 0:
            return 0.0
        return self.n_rfqs_filled / self.n_rfqs_quoted

    def get_quote_rate(self) -> float:
        """Get quote rate (quotes submitted / RFQs seen)."""
        if self.n_rfqs_seen == 0:
            return 0.0
        return self.n_rfqs_quoted / self.n_rfqs_seen

    def get_average_spread(self) -> float:
        """Get average spread earned per fill (in bps)."""
        if self.n_rfqs_filled == 0:
            return 0.0
        avg_spread_dollars = self.total_spread_pnl / self.n_rfqs_filled
        return avg_spread_dollars / (self.cfg.p0 * self.cfg.lot_size_mm * 10000) * 10000

    def to_summary_dict(self) -> dict:
        """Export state summary as dictionary."""
        return {
            "q": self.q,
            "cash": self.cash,
            "alpha": self.alpha,
            "alpha_remaining": self.alpha_remaining,
            "q_target": self.q_target,
            "exit_mode": self.exit_mode.value,
            "current_day": self.current_day,
            "current_regime": self.current_regime.name,
            "n_rfqs_seen": self.n_rfqs_seen,
            "n_rfqs_quoted": self.n_rfqs_quoted,
            "n_rfqs_filled": self.n_rfqs_filled,
            "fill_rate": self.get_fill_rate(),
            "quote_rate": self.get_quote_rate(),
            "total_volume": self.total_volume,
            "total_spread_pnl": self.total_spread_pnl,
            "avg_spread_bps": self.get_average_spread(),
        }


def create_initial_state(cfg: SimConfig) -> SimulationState:
    """
    Create initial simulation state.

    Args:
        cfg: SimConfig

    Returns:
        Fresh SimulationState
    """
    return SimulationState(
        cfg=cfg,
        last_price=cfg.p0,
    )
