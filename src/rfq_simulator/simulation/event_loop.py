"""
Main simulation event loop: Process RFQs, update state, track P&L.

This is the core simulation driver that:
1. Generates all WORLD processes upfront (price, regime, RFQs)
2. Iterates through RFQs in time order
3. At each RFQ: compute quote, simulate competition, update state
4. Returns full simulation results for analysis
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numpy.random import Generator

from ..config import SimConfig
from ..world.clock import TimeGrid
from ..world.price import generate_price_path, apply_adverse_move
from ..world.regime import generate_regime_path, get_regime_at_day, Regime
from ..world.rfq_stream import generate_rfq_stream, RFQEvent
from ..world.competitors import DealerPool, simulate_competition
from ..world.street_lean import generate_street_lean_path, get_street_lean_at_step

from ..agent.alpha import AlphaSignalManager
from ..agent.exit import HybridExitManager, ExitMode
from ..agent.target import compute_target_position
from ..agent.observable import compute_observable_mid, compute_skew
from ..agent.lean import compute_lean
from ..agent.quoting import compute_optimal_quote

from ..core.state import SimulationState, RFQLog, ExitMode, create_initial_state
from ..core.accounting import PnLTracker, PnLDecomposition


@dataclass
class SimulationResult:
    """
    Complete results from a simulation run.

    Contains final state, P&L decomposition, and full event log.
    """

    # Final state
    final_state: SimulationState

    # P&L decomposition
    pnl: PnLDecomposition
    pnl_tracker: PnLTracker

    # World data (for replay/analysis)
    prices: np.ndarray
    regime_path: np.ndarray
    street_lean_path: np.ndarray
    rfq_events: List[RFQEvent]

    # Config for reference
    cfg: SimConfig

    # Summary metrics
    @property
    def total_pnl(self) -> float:
        return self.pnl.total_pnl

    @property
    def sharpe_ratio(self) -> float:
        from ..core.accounting import compute_sharpe_ratio
        ts = self.pnl_tracker.get_time_series()
        if len(ts["total_pnl"]) < 2:
            return 0.0
        return compute_sharpe_ratio(ts["total_pnl"])

    @property
    def max_drawdown(self) -> float:
        from ..core.accounting import compute_max_drawdown
        ts = self.pnl_tracker.get_time_series()
        return compute_max_drawdown(ts["total_pnl"])

    def summary(self) -> dict:
        """Get summary metrics."""
        state = self.final_state
        return {
            "total_pnl": self.total_pnl,
            "alpha_pnl": self.pnl.alpha_pnl,
            "spread_pnl": self.pnl.spread_pnl,
            "carry_pnl": self.pnl.carry_pnl,
            "aggress_cost": self.pnl.aggress_cost,
            "n_rfqs": state.n_rfqs_seen,
            "n_fills": state.n_rfqs_filled,
            "fill_rate": state.get_fill_rate(),
            "avg_spread_bps": state.get_average_spread(),
            "final_inventory": state.q,
            "sharpe": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
        }


def run_simulation(
    cfg: SimConfig,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> SimulationResult:
    """
    Run a complete simulation.

    Args:
        cfg: SimConfig with all parameters
        seed: Random seed (uses cfg.seed if None)
        verbose: Print progress if True

    Returns:
        SimulationResult with all outputs
    """
    # Initialize RNG
    seed = seed if seed is not None else cfg.seed
    rng = np.random.default_rng(seed)

    # Create separate RNG streams for reproducibility
    price_rng = np.random.default_rng(rng.integers(2**32))
    rfq_rng = np.random.default_rng(rng.integers(2**32))
    competitor_rng = np.random.default_rng(rng.integers(2**32))
    agent_rng = np.random.default_rng(rng.integers(2**32))

    # Validate config
    cfg.validate()

    # Create time grid
    time_grid = TimeGrid(cfg)

    # =========================================================================
    # STEP 1: Generate WORLD processes upfront
    # =========================================================================

    if verbose:
        print("Generating world processes...")

    # Price path
    prices = generate_price_path(cfg, price_rng)

    # Regime path
    regime_path = generate_regime_path(cfg, rng)

    # RFQ stream
    rfq_events = generate_rfq_stream(cfg, rfq_rng)

    # Street lean (OU process)
    street_lean_rng = np.random.default_rng(rng.integers(2**32))
    street_lean_path = generate_street_lean_path(cfg, street_lean_rng)

    if verbose:
        print(f"  Price path: {len(prices)} steps")
        print(f"  RFQ events: {len(rfq_events)}")
        stress_pct = 100 * np.mean(regime_path == Regime.STRESSED)
        print(f"  Stress regime: {stress_pct:.1f}% of days")
        print(f"  Street lean range: [{street_lean_path.min():.2f}, {street_lean_path.max():.2f}] bps")

    # =========================================================================
    # STEP 2: Initialize state
    # =========================================================================

    state = create_initial_state(cfg)
    pnl_tracker = PnLTracker(cfg)
    pnl_tracker._last_price = cfg.p0

    # Alpha signal manager
    alpha_manager = AlphaSignalManager(cfg, time_grid)

    # Dealer pool for competitor simulation
    dealer_pool = DealerPool(cfg.n_dealers_max, cfg, competitor_rng)

    # Hybrid exit manager
    exit_manager = HybridExitManager(cfg)

    # =========================================================================
    # STEP 3: Process RFQ events
    # =========================================================================

    if verbose:
        print("Processing RFQ events...")

    for rfq_idx, rfq in enumerate(rfq_events):
        # Get current time and day
        current_minute = rfq.time
        current_day = time_grid.minute_to_day(current_minute)
        current_step = time_grid.minute_to_step(current_minute)

        # Get true price at RFQ time
        p_true = prices[current_step]

        # ---------------------------------------------------------------------
        # Update time-dependent state
        # ---------------------------------------------------------------------

        # Get street lean at current time
        street_lean = get_street_lean_at_step(street_lean_path, current_step)

        # Refresh dealer biases at start of new day (centered on street lean)
        if current_day > state.current_day:
            dealer_pool.refresh_biases(current_day, street_lean)
            state.current_day = current_day

            # Record daily carry
            pnl_tracker.record_carry(state.q, days=1.0)

            # Reset exit manager for new signal if applicable
            if state.t_signal is not None and alpha_manager.should_refresh(current_minute):
                exit_manager.reset()

        # Update regime
        state.current_regime = get_regime_at_day(regime_path, current_day)

        # Refresh alpha signal if needed
        if alpha_manager.should_refresh(current_minute):
            signal = alpha_manager.generate_signal(
                current_minute=current_minute,
                prices=prices,
                regime=state.current_regime,
                rng=agent_rng,
            )
            state.alpha = signal.alpha
            state.alpha_star = signal.alpha_star
            state.t_signal = current_minute

            if verbose and rfq_idx < 10:
                print(f"  New alpha signal: α={signal.alpha:.2f}, α*={signal.alpha_star:.2f}")

        # Compute remaining alpha
        state.alpha_remaining = alpha_manager.get_remaining_alpha(current_minute)

        # Compute target position
        state.q_target = compute_target_position(state.alpha_remaining, cfg)

        # Record alpha P&L from price moves since last RFQ
        pnl_tracker.record_price_move(state.q, p_true, current_minute)

        # ---------------------------------------------------------------------
        # Compute agent decisions
        # ---------------------------------------------------------------------

        # Observable mid (lagged + noisy)
        mid_obs = compute_observable_mid(
            prices, current_minute, time_grid, cfg, agent_rng
        )

        # Skew correction
        skew = compute_skew(
            prices, mid_obs, current_minute, time_grid, cfg, agent_rng
        )

        # Lean
        signal_age = alpha_manager.get_signal_age(current_minute)
        horizon_minutes = cfg.alpha_horizon_days * cfg.minutes_per_day
        lean = compute_lean(
            state.q, state.q_target, signal_age, horizon_minutes, cfg
        )

        # Theoretical price
        theo = mid_obs + skew + lean

        # ---------------------------------------------------------------------
        # Check exit mode and handle aggressive unwind if needed
        # ---------------------------------------------------------------------

        if alpha_manager.current_signal is not None:
            exit_decision = exit_manager.get_exit_decision(
                current_minute=current_minute,
                current_q=state.q,
                t_signal=state.t_signal,
                horizon_minutes=horizon_minutes,
            )

            if exit_decision.should_aggress:
                # Execute aggressive trade
                aggress_size = exit_decision.aggress_size
                aggress_cost = exit_manager.compute_aggress_cost(aggress_size)

                # Record cost
                pnl_tracker.aggress_cost += aggress_cost

                # Update position
                state.q += aggress_size

                if verbose and rfq_idx < 5:
                    print(f"  Aggressive exit: size={aggress_size:.1f}, cost=${aggress_cost:.2f}")

                # Skip RFQ processing when in aggressive mode
                continue

        # ---------------------------------------------------------------------
        # Compute optimal quote
        # ---------------------------------------------------------------------

        quote_result = compute_optimal_quote(
            rfq=rfq,
            theo_price=theo,
            current_q=state.q,
            target_q=state.q_target,
            alpha_remaining=state.alpha_remaining,
            cfg=cfg,
        )

        state.n_rfqs_seen += 1

        if quote_result.declined:
            state.n_rfqs_declined += 1
            # Log declined RFQ
            log_entry = _create_rfq_log(
                rfq=rfq,
                p_true=p_true,
                mid_obs=mid_obs,
                skew=skew,
                lean=lean,
                theo=theo,
                state=state,
                quote_result=quote_result,
                filled=False,
                n_resp=0,
                best_comp=None,
                spread_pnl=0.0,
                adverse=0.0,
            )
            state.rfq_log.append(log_entry)
            continue

        state.n_rfqs_quoted += 1

        # ---------------------------------------------------------------------
        # Simulate competition
        # ---------------------------------------------------------------------

        competition = simulate_competition(
            rfq=rfq,
            p_true=p_true,
            trader_price=quote_result.quote_price,
            dealer_pool=dealer_pool,
            street_lean=street_lean,
            rng=competitor_rng,
        )

        # ---------------------------------------------------------------------
        # Process fill if we won
        # ---------------------------------------------------------------------

        if competition.trader_won:
            state.n_rfqs_filled += 1

            # Inventory change: client buy = we sell = negative delta_q
            delta_q = -1 if rfq.is_client_buy else +1

            # Signed size for spread calculation
            # Client buy (we sell): signed_size > 0
            # Client sell (we buy): signed_size < 0
            signed_size = rfq.size if rfq.is_client_buy else -rfq.size

            # Record spread P&L
            spread_pnl = pnl_tracker.record_spread(
                fill_price=quote_result.quote_price,
                true_price=p_true,
                signed_size=signed_size,
            )
            state.record_spread_pnl(spread_pnl)

            # Update position
            state.update_position(delta_q, quote_result.quote_price, rfq.size)

            # Apply adverse move to price path
            adverse_move = apply_adverse_move(
                prices=prices,
                step_idx=current_step,
                direction=1 if rfq.is_client_buy else -1,
                toxicity=rfq.toxicity,
                cfg=cfg,
                rng=price_rng,
            )
            state.record_adverse_move(adverse_move)

            # Log filled RFQ
            log_entry = _create_rfq_log(
                rfq=rfq,
                p_true=p_true,
                mid_obs=mid_obs,
                skew=skew,
                lean=lean,
                theo=theo,
                state=state,
                quote_result=quote_result,
                filled=True,
                n_resp=competition.n_competitors_responded,
                best_comp=competition.best_competitor_price,
                spread_pnl=spread_pnl,
                adverse=adverse_move,
            )
            state.rfq_log.append(log_entry)

        else:
            # Log missed RFQ (quoted but lost)
            log_entry = _create_rfq_log(
                rfq=rfq,
                p_true=p_true,
                mid_obs=mid_obs,
                skew=skew,
                lean=lean,
                theo=theo,
                state=state,
                quote_result=quote_result,
                filled=False,
                n_resp=competition.n_competitors_responded,
                best_comp=competition.best_competitor_price,
                spread_pnl=0.0,
                adverse=0.0,
            )
            state.rfq_log.append(log_entry)

        # Snapshot P&L periodically
        if rfq_idx % 10 == 0:
            pnl_tracker.snapshot(current_minute)

    # =========================================================================
    # STEP 4: Finalize
    # =========================================================================

    # Final P&L snapshot
    final_price = prices[-1]
    pnl_tracker.record_price_move(state.q, final_price, cfg.total_minutes)
    pnl_tracker.snapshot(cfg.total_minutes)

    state.last_price = final_price

    if verbose:
        print(f"Simulation complete:")
        print(f"  Total P&L: ${pnl_tracker.total_pnl:,.2f}")
        print(f"  Alpha P&L: ${pnl_tracker.alpha_pnl:,.2f}")
        print(f"  Spread P&L: ${pnl_tracker.spread_pnl:,.2f}")
        print(f"  Fill rate: {state.get_fill_rate():.1%}")
        print(f"  Final inventory: {state.q:.1f} lots")

    return SimulationResult(
        final_state=state,
        pnl=pnl_tracker.get_decomposition(),
        pnl_tracker=pnl_tracker,
        prices=prices,
        regime_path=regime_path,
        street_lean_path=street_lean_path,
        rfq_events=rfq_events,
        cfg=cfg,
    )


def _create_rfq_log(
    rfq: RFQEvent,
    p_true: float,
    mid_obs: float,
    skew: float,
    lean: float,
    theo: float,
    state: SimulationState,
    quote_result,
    filled: bool,
    n_resp: int,
    best_comp: Optional[float],
    spread_pnl: float,
    adverse: float,
) -> RFQLog:
    """Create an RFQ log entry."""
    return RFQLog(
        time=rfq.time,
        is_client_buy=rfq.is_client_buy,
        size=rfq.size,
        n_dealers=rfq.n_dealers,
        toxicity=rfq.toxicity,
        p_true=p_true,
        mid_obs=mid_obs,
        skew=skew,
        lean=lean,
        theo=theo,
        q_before=state.q - ((-1 if rfq.is_client_buy else 1) * rfq.size if filled else 0),
        q_target=state.q_target,
        alpha_remaining=state.alpha_remaining,
        quote_price=quote_result.quote_price,
        markup_bps=quote_result.markup_bps,
        win_prob_est=quote_result.win_probability,
        expected_value=quote_result.expected_value,
        declined=quote_result.declined,
        decline_reason=quote_result.decline_reason,
        filled=filled,
        n_competitors_responded=n_resp,
        best_competitor_price=best_comp,
        q_after=state.q,
        spread_pnl=spread_pnl,
        adverse_move=adverse,
        regime=state.current_regime,
    )
