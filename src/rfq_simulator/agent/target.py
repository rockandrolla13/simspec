"""
Target position: Maps alpha signal to desired inventory level.

Implements Eq 8 from spec:
    q* = clip(α_rem / (γ * σ²_daily), -q_max, q_max)

where γ is risk aversion and σ²_daily is daily price variance.
"""

import numpy as np

from ..config import SimConfig


def compute_target_position(alpha_remaining: float, cfg: SimConfig) -> float:
    """
    Compute target inventory from remaining alpha.

    Eq 8: q* = clip(α_rem / (γ * σ²_daily), -q_max, q_max)

    Args:
        alpha_remaining: Remaining alpha in dollars
        cfg: SimConfig with gamma, sigma_daily_bps, p0, q_max

    Returns:
        Target position in lots (float, not rounded)
    """
    # Convert sigma from bps to dollar variance
    sigma_daily_dollars = cfg.sigma_daily_bps / 10000.0 * cfg.p0
    sigma_squared = sigma_daily_dollars ** 2

    # Avoid division by zero
    if sigma_squared < 1e-12 or cfg.gamma < 1e-12:
        return 0.0

    # Raw target (Eq 8 before clip)
    q_star_raw = alpha_remaining / (cfg.gamma * sigma_squared)

    # Clip to position limits
    q_star = np.clip(q_star_raw, -cfg.q_max, cfg.q_max)

    return q_star


def compute_feasibility_bound(
    favorable_rfq_rate: float,
    avg_win_prob: float,
    time_remaining_minutes: float,
    cfg: SimConfig,
) -> float:
    """
    Compute the feasible position given flow constraints.

    Eq 9: q*_feasible ≤ μ_fav * P̄(win) * (T - t)

    This is a diagnostic, not a constraint. If q* exceeds this bound,
    the trader should expect underdelivery.

    Args:
        favorable_rfq_rate: Rate of favorable RFQs per minute
        avg_win_prob: Average win probability on favorable side
        time_remaining_minutes: Time until signal expiry
        cfg: SimConfig

    Returns:
        Maximum feasible position in lots
    """
    expected_fills = favorable_rfq_rate * avg_win_prob * time_remaining_minutes
    return expected_fills


def is_favorable_direction(
    rfq_is_client_buy: bool, current_q: float, target_q: float
) -> bool:
    """
    Check if an RFQ direction moves inventory toward target.

    Args:
        rfq_is_client_buy: True if client is buying (we sell)
        current_q: Current inventory in lots
        target_q: Target inventory in lots

    Returns:
        True if filling this RFQ moves us closer to target
    """
    # If client buys, we sell (inventory decreases)
    # If client sells, we buy (inventory increases)
    inventory_change = -1 if rfq_is_client_buy else +1

    # Favorable if it reduces the gap |q - q*|
    current_gap = abs(current_q - target_q)
    new_gap = abs((current_q + inventory_change) - target_q)

    return new_gap < current_gap


def compute_continuation_value(
    alpha_remaining: float,
    current_q: float,
    target_q: float,
    delta_q: int,
    cfg: SimConfig,
) -> float:
    """
    Compute the continuation value of an inventory change.

    Eq 23:
        ΔV = +α_rem * |Δq|           if favorable (toward q*)
           = -γ * σ² * |q - q*| * |Δq| / q_max   if adverse

    Args:
        alpha_remaining: Remaining alpha
        current_q: Current inventory
        target_q: Target inventory
        delta_q: Signed inventory change (+1 if we buy, -1 if we sell)
        cfg: SimConfig

    Returns:
        Continuation value in dollars
    """
    sigma_daily_dollars = cfg.sigma_daily_bps / 10000.0 * cfg.p0
    sigma_squared = sigma_daily_dollars ** 2

    # Check if favorable
    current_gap = abs(current_q - target_q)
    new_gap = abs((current_q + delta_q) - target_q)
    is_favorable = new_gap < current_gap

    abs_delta_q = abs(delta_q)

    if is_favorable:
        # Benefit: capture more alpha
        return alpha_remaining * abs_delta_q
    else:
        # Cost: moving away from target
        risk_penalty = cfg.gamma * sigma_squared * current_gap * abs_delta_q / cfg.q_max
        return -risk_penalty
