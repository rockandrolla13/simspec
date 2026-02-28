"""
Optimal quoting: Grid search for the best markup.

Implements Eq 21-24 from spec:
    m* = argmax_m P̂(win|m) * [edge(m) + ΔV]

where:
    edge(m) = m * direction  (positive = trader earns spread)
    ΔV = continuation value from inventory change

Quote bounds:
    1. Alpha bound: edge + ΔV >= -|α_rem| * |Δq|
    2. Position bound: |q + Δq| <= q_max
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import SimConfig
from ..world.rfq_stream import RFQEvent
from .winrate import estimate_win_probability
from .target import compute_continuation_value


@dataclass
class QuoteResult:
    """
    Result of the optimal quote computation.

    Attributes:
        quote_price: Optimal quote price in dollars (None if declined)
        markup_bps: Optimal markup over theo in bps
        expected_value: Expected value at optimal markup
        win_probability: Estimated win probability at optimal markup
        edge_bps: Edge captured if filled (markup * direction)
        continuation_value: ΔV from inventory change
        declined: True if no feasible quote exists
        decline_reason: Reason for declining (if applicable)
    """

    quote_price: Optional[float]
    markup_bps: float
    expected_value: float
    win_probability: float
    edge_bps: float
    continuation_value: float
    declined: bool
    decline_reason: Optional[str]


def compute_edge(markup_bps: float, is_client_buy: bool) -> float:
    """
    Compute edge earned at a given markup.

    For offers (client buy): edge = +markup (higher price = more edge)
    For bids (client sell): edge = +markup (with our convention that positive
        markup means quoting "wider" i.e., lower bid = more edge for us)

    Convention: positive markup always means we're earning more edge.
    The quote formula handles the directional translation.

    Args:
        markup_bps: Markup over theo in bps (positive = wider quote)
        is_client_buy: True if client is buying

    Returns:
        Edge in bps (positive = we earn)
    """
    # Positive markup = earning edge on both sides
    return markup_bps


def check_position_bound(
    current_q: float,
    rfq: RFQEvent,
    cfg: SimConfig,
) -> bool:
    """
    Check if filling this RFQ would violate position limits.

    Args:
        current_q: Current inventory in lots
        rfq: RFQ event
        cfg: SimConfig

    Returns:
        True if position is feasible after fill
    """
    # Inventory change: client buy = we sell = -size
    delta_q = -rfq.size if rfq.is_client_buy else +rfq.size
    new_q = current_q + delta_q

    return abs(new_q) <= cfg.q_max


def check_alpha_bound(
    edge_bps: float,
    continuation_value: float,
    alpha_remaining: float,
    size: int,
    cfg: SimConfig,
) -> bool:
    """
    Check if quote satisfies the alpha bound.

    Alpha bound: edge + ΔV >= -|α_rem| * |Δq|

    We never pay more than the remaining alpha to acquire position.

    Args:
        edge_bps: Edge in bps
        continuation_value: ΔV in dollars
        alpha_remaining: Remaining alpha in dollars
        size: Trade size in lots
        cfg: SimConfig

    Returns:
        True if alpha bound is satisfied
    """
    # Convert edge from bps to dollars
    edge_dollars = edge_bps / 10000.0 * cfg.p0 * size * cfg.lot_size_mm * 10000

    # Total value
    total_value = edge_dollars + continuation_value

    # Maximum we're willing to pay
    max_cost = abs(alpha_remaining) * size

    return total_value >= -max_cost


def compute_objective(
    markup_bps: float,
    rfq: RFQEvent,
    current_q: float,
    target_q: float,
    alpha_remaining: float,
    cfg: SimConfig,
) -> tuple:
    """
    Compute the objective function at a given markup.

    Objective: P̂(win|m) * [edge(m) + ΔV]

    Args:
        markup_bps: Candidate markup in bps
        rfq: RFQ event
        current_q: Current inventory
        target_q: Target inventory
        alpha_remaining: Remaining alpha
        cfg: SimConfig

    Returns:
        (objective_value, win_prob, edge_bps, delta_v)
    """
    # Win probability estimate
    win_prob = estimate_win_probability(markup_bps, rfq, cfg)

    # Edge
    edge_bps = compute_edge(markup_bps, rfq.is_client_buy)

    # Inventory change
    delta_q = -rfq.size if rfq.is_client_buy else +rfq.size

    # Continuation value
    delta_v = compute_continuation_value(
        alpha_remaining=alpha_remaining,
        current_q=current_q,
        target_q=target_q,
        delta_q=delta_q,
        cfg=cfg,
    )

    # Convert edge to dollars for objective
    edge_dollars = edge_bps / 10000.0 * cfg.p0 * rfq.size * cfg.lot_size_mm * 10000

    # Objective
    objective = win_prob * (edge_dollars + delta_v)

    return objective, win_prob, edge_bps, delta_v


def compute_optimal_quote(
    rfq: RFQEvent,
    theo_price: float,
    current_q: float,
    target_q: float,
    alpha_remaining: float,
    cfg: SimConfig,
) -> QuoteResult:
    """
    Compute the optimal quote via grid search.

    Searches m ∈ [-m_max, +m_max] at m_grid resolution.
    Applies position and alpha bounds.

    Args:
        rfq: RFQ event
        theo_price: Trader's theoretical price (mid + skew + lean)
        current_q: Current inventory in lots
        target_q: Target inventory in lots
        alpha_remaining: Remaining alpha in dollars
        cfg: SimConfig

    Returns:
        QuoteResult with optimal quote or decline
    """
    # Check position bound first
    if not check_position_bound(current_q, rfq, cfg):
        return QuoteResult(
            quote_price=None,
            markup_bps=0.0,
            expected_value=0.0,
            win_probability=0.0,
            edge_bps=0.0,
            continuation_value=0.0,
            declined=True,
            decline_reason="position_limit",
        )

    # Build markup grid
    markups = np.arange(-cfg.m_max_bps, cfg.m_max_bps + cfg.m_grid_bps, cfg.m_grid_bps)

    # Inventory change for continuation value
    delta_q = -rfq.size if rfq.is_client_buy else +rfq.size

    # Evaluate objective at each markup
    best_markup = 0.0
    best_objective = float("-inf")
    best_win_prob = 0.0
    best_edge = 0.0
    best_delta_v = 0.0

    for m in markups:
        obj, win_prob, edge, delta_v = compute_objective(
            markup_bps=m,
            rfq=rfq,
            current_q=current_q,
            target_q=target_q,
            alpha_remaining=alpha_remaining,
            cfg=cfg,
        )

        # Check alpha bound
        if not check_alpha_bound(edge, delta_v, alpha_remaining, rfq.size, cfg):
            continue

        if obj > best_objective:
            best_objective = obj
            best_markup = m
            best_win_prob = win_prob
            best_edge = edge
            best_delta_v = delta_v

    # Check if we found any feasible markup
    if best_objective == float("-inf"):
        return QuoteResult(
            quote_price=None,
            markup_bps=0.0,
            expected_value=0.0,
            win_probability=0.0,
            edge_bps=0.0,
            continuation_value=0.0,
            declined=True,
            decline_reason="alpha_bound",
        )

    # Check if expected value is positive
    if best_objective <= 0:
        return QuoteResult(
            quote_price=None,
            markup_bps=best_markup,
            expected_value=best_objective,
            win_probability=best_win_prob,
            edge_bps=best_edge,
            continuation_value=best_delta_v,
            declined=True,
            decline_reason="negative_ev",
        )

    # Compute final quote price
    # Direction: client buy = offer (add markup), client sell = bid (subtract)
    direction = 1 if rfq.is_client_buy else -1
    bps_to_dollars = cfg.p0 / 10000.0
    quote_price = theo_price + direction * best_markup * bps_to_dollars

    return QuoteResult(
        quote_price=quote_price,
        markup_bps=best_markup,
        expected_value=best_objective,
        win_probability=best_win_prob,
        edge_bps=best_edge,
        continuation_value=best_delta_v,
        declined=False,
        decline_reason=None,
    )


def compute_quote_surface(
    rfq: RFQEvent,
    current_q: float,
    target_q: float,
    alpha_remaining: float,
    cfg: SimConfig,
    markup_range: tuple = (-15, 25),
    n_points: int = 41,
) -> tuple:
    """
    Compute the objective surface for visualization.

    Args:
        rfq: RFQ event
        current_q: Current inventory
        target_q: Target inventory
        alpha_remaining: Remaining alpha
        cfg: SimConfig
        markup_range: (min, max) markup range
        n_points: Number of points

    Returns:
        (markups, objectives, win_probs, edges, delta_vs)
    """
    markups = np.linspace(markup_range[0], markup_range[1], n_points)

    objectives = []
    win_probs = []
    edges = []
    delta_vs = []

    for m in markups:
        obj, wp, e, dv = compute_objective(
            markup_bps=m,
            rfq=rfq,
            current_q=current_q,
            target_q=target_q,
            alpha_remaining=alpha_remaining,
            cfg=cfg,
        )
        objectives.append(obj)
        win_probs.append(wp)
        edges.append(e)
        delta_vs.append(dv)

    return (
        markups,
        np.array(objectives),
        np.array(win_probs),
        np.array(edges),
        np.array(delta_vs),
    )
