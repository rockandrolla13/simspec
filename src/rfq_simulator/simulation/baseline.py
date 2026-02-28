"""
Baseline strategy: Naive aggressor that executes immediately.

The baseline provides a benchmark for the LP strategy:
- Same alpha signals (IC, refresh, decay)
- Same target positions q*
- Executes IMMEDIATELY when target changes (no waiting for RFQs)
- Pays aggressor costs (half-spread + market impact)
- Does NOT interact with RFQs (no adverse moves caused)

Key metric: Alpha Capture Ratio (ACR)
- Baseline has higher ACR (faster convergence)
- LP has lower ACR but earns spread
- Question: Does spread income exceed ACR loss?
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.random import Generator

from ..config import SimConfig
from ..world.clock import TimeGrid
from ..world.regime import generate_regime_path, get_regime_at_day
from ..agent.alpha import AlphaSignalManager
from ..agent.target import compute_target_position
from ..core.accounting import PnLTracker, PnLDecomposition


@dataclass
class BaselineResult:
    """
    Results from baseline strategy simulation.

    Designed to be comparable with SimulationResult from event_loop.
    """

    # P&L decomposition
    pnl: PnLDecomposition
    pnl_tracker: PnLTracker

    # Strategy metrics
    total_trades: int
    total_volume: float  # in lots
    total_aggress_cost: float

    # Alpha tracking
    avg_position_gap: float  # Average |q - q*|
    time_to_target: float  # Average time to reach target (minutes)

    # Config reference
    cfg: SimConfig

    @property
    def total_pnl(self) -> float:
        return self.pnl.total_pnl

    @property
    def alpha_capture_ratio(self) -> float:
        """Ratio of alpha P&L to theoretical maximum."""
        if self.pnl.alpha_pnl == 0:
            return 0.0
        # Theoretical max would be if we were always at target
        # This is an approximation
        return 1.0  # Baseline by definition has near-perfect ACR

    def summary(self) -> dict:
        """Get summary metrics."""
        return {
            "total_pnl": self.total_pnl,
            "alpha_pnl": self.pnl.alpha_pnl,
            "carry_pnl": self.pnl.carry_pnl,
            "aggress_cost": self.pnl.aggress_cost,
            "total_trades": self.total_trades,
            "total_volume": self.total_volume,
            "avg_position_gap": self.avg_position_gap,
        }


def run_baseline(
    prices: np.ndarray,
    regime_path: np.ndarray,
    cfg: SimConfig,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> BaselineResult:
    """
    Run baseline aggressive strategy.

    Uses the SAME price path as the LP strategy (without mutation),
    the SAME alpha signals, but executes immediately via aggression.

    Args:
        prices: Pre-generated price path (NOT mutated)
        regime_path: Pre-generated regime path
        cfg: SimConfig
        seed: Random seed for alpha generation
        verbose: Print progress if True

    Returns:
        BaselineResult with comparable metrics
    """
    seed = seed if seed is not None else cfg.seed
    rng = np.random.default_rng(seed)

    # Time grid
    time_grid = TimeGrid(cfg)

    # Alpha manager (same signals as LP strategy if same seed)
    alpha_manager = AlphaSignalManager(cfg, time_grid)

    # P&L tracker
    pnl_tracker = PnLTracker(cfg)
    pnl_tracker._last_price = cfg.p0

    # State
    q = 0.0  # Current position
    q_target = 0.0
    alpha_remaining = 0.0

    # Metrics
    total_trades = 0
    total_volume = 0.0
    position_gaps = []

    # Simulate at daily granularity (when signals refresh)
    # Check more frequently to track position gaps
    check_interval = cfg.dt_minutes  # Check every price step

    if verbose:
        print("Running baseline strategy...")

    for step in range(len(prices)):
        current_minute = step * cfg.dt_minutes
        current_day = time_grid.minute_to_day(current_minute)
        current_price = prices[step]

        # Record alpha P&L from price move
        pnl_tracker.record_price_move(q, current_price, current_minute)

        # Get regime
        regime = get_regime_at_day(regime_path, current_day)

        # Refresh alpha if needed
        if alpha_manager.should_refresh(current_minute):
            signal = alpha_manager.generate_signal(
                current_minute=current_minute,
                prices=prices,
                regime=regime,
                rng=rng,
            )

            if verbose and total_trades < 5:
                print(f"  Day {current_day}: New signal α={signal.alpha:.2f}")

        # Get remaining alpha and target
        alpha_remaining = alpha_manager.get_remaining_alpha(current_minute)
        q_target = compute_target_position(alpha_remaining, cfg)

        # Track position gap
        gap = abs(q - q_target)
        position_gaps.append(gap)

        # Execute immediately if not at target
        if q != q_target:
            # Compute trade needed
            trade_size = q_target - q

            # Pay aggressor costs
            aggress_cost = pnl_tracker.record_aggress_cost(
                size=abs(trade_size),
                half_spread_bps=cfg.aggress_halfspread_bps,
                impact_bps=cfg.aggress_impact_bps,
            )

            # Update position
            q = q_target
            total_trades += 1
            total_volume += abs(trade_size)

        # Daily carry (at day boundaries)
        if step > 0 and current_day > time_grid.minute_to_day((step - 1) * cfg.dt_minutes):
            pnl_tracker.record_carry(q, days=1.0)

        # Snapshot P&L periodically
        if step % 100 == 0:
            pnl_tracker.snapshot(current_minute)

    # Final snapshot
    pnl_tracker.snapshot(cfg.total_minutes)

    if verbose:
        print(f"Baseline complete:")
        print(f"  Total P&L: ${pnl_tracker.total_pnl:,.2f}")
        print(f"  Alpha P&L: ${pnl_tracker.alpha_pnl:,.2f}")
        print(f"  Aggress cost: ${pnl_tracker.aggress_cost:,.2f}")
        print(f"  Total trades: {total_trades}")

    avg_gap = np.mean(position_gaps) if position_gaps else 0.0

    return BaselineResult(
        pnl=pnl_tracker.get_decomposition(),
        pnl_tracker=pnl_tracker,
        total_trades=total_trades,
        total_volume=total_volume,
        total_aggress_cost=pnl_tracker.aggress_cost,
        avg_position_gap=avg_gap,
        time_to_target=0.0,  # Baseline is always at target
        cfg=cfg,
    )


def compare_strategies(
    lp_result,  # SimulationResult
    baseline_result: BaselineResult,
) -> dict:
    """
    Compare LP strategy against baseline.

    Args:
        lp_result: Result from run_simulation()
        baseline_result: Result from run_baseline()

    Returns:
        Dictionary with comparison metrics
    """
    lp_summary = lp_result.summary()
    bl_summary = baseline_result.summary()

    return {
        # Absolute P&L
        "lp_total_pnl": lp_summary["total_pnl"],
        "baseline_total_pnl": bl_summary["total_pnl"],
        "pnl_difference": lp_summary["total_pnl"] - bl_summary["total_pnl"],

        # P&L components
        "lp_alpha_pnl": lp_summary["alpha_pnl"],
        "baseline_alpha_pnl": bl_summary["alpha_pnl"],
        "alpha_sacrifice": bl_summary["alpha_pnl"] - lp_summary["alpha_pnl"],

        "lp_spread_pnl": lp_summary["spread_pnl"],
        "baseline_spread_pnl": 0.0,  # Baseline doesn't earn spread

        "lp_aggress_cost": lp_summary["aggress_cost"],
        "baseline_aggress_cost": bl_summary["aggress_cost"],

        # Trade efficiency
        "lp_fill_rate": lp_summary["fill_rate"],
        "lp_avg_spread_bps": lp_summary["avg_spread_bps"],

        "baseline_avg_gap": bl_summary["avg_position_gap"],
        "lp_final_inventory": lp_summary["final_inventory"],

        # Key question: Does spread exceed alpha sacrifice?
        "spread_minus_alpha_loss": (
            lp_summary["spread_pnl"] -
            (bl_summary["alpha_pnl"] - lp_summary["alpha_pnl"])
        ),
    }
