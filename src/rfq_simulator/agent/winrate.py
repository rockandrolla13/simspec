"""
Win-rate estimation: Trader's ESTIMATED win probability model.

This is the model the trader uses for quote optimization, NOT the
actual win probability (which emerges from competitors.py simulation).

The trader's model can be miscalibrated via winrate_est_error param,
creating a realistic gap between beliefs and reality.

Implements logistic win-rate (Eq 21 from V1 spec, used as estimation):
    P̂(win|m, z) = 1 / (1 + exp(-β̂(z) * (ĉ(z) - m)))

where:
    ĉ(z) = c_base + c_N*(N-N̄) + c_size*log(size)
    β̂(z) = β_base + β_N*N
"""

import numpy as np

from ..config import SimConfig
from ..world.rfq_stream import RFQEvent


def compute_estimated_center(rfq: RFQEvent, cfg: SimConfig) -> float:
    """
    Compute the trader's estimated competitive center.

    The center ĉ(z) is the markup at which win probability = 50%.

    Eq 21 (center): ĉ(z) = c_base + c_N*(N-N̄) + c_size*log(size)

    More dealers → tighter center (negative c_N)
    Larger size → wider center (positive c_size)

    Args:
        rfq: RFQ event
        cfg: SimConfig

    Returns:
        Estimated competitive center in bps
    """
    center = cfg.markup_base_bps
    center += cfg.markup_N_bps * (rfq.n_dealers - cfg.n_dealers_mean)
    center += cfg.markup_size_bps * np.log(max(1, rfq.size))

    # Apply estimation error if configured
    # Positive error = trader thinks market is wider than it is
    center *= (1.0 + cfg.winrate_est_error)

    return center


def compute_estimated_steepness(rfq: RFQEvent, cfg: SimConfig) -> float:
    """
    Compute the trader's estimated curve steepness.

    The steepness β̂(z) determines how fast win probability changes
    with markup. Higher β = more competitive, steeper curve.

    From V1 spec: β̂(z) = β_base + β_N*N

    Note: We derive β from the competitor model parameters.
    The logistic slope relates to the dispersion of competitor quotes.

    Args:
        rfq: RFQ event
        cfg: SimConfig

    Returns:
        Estimated steepness parameter (inverse bps)
    """
    # Approximate steepness from quote noise
    # If quote_noise ~ N(0, σ), then logistic β ≈ π / (sqrt(3) * σ)
    # This creates the right shape for the win-rate curve
    total_noise_bps = np.sqrt(
        cfg.quote_noise_bps ** 2 +
        cfg.markup_noise_bps ** 2 +
        cfg.dealer_bias_std_bps ** 2
    )

    # Logistic approximation to normal CDF
    beta = np.pi / (np.sqrt(3) * max(0.5, total_noise_bps))

    # Steepness increases with more dealers (more competition)
    beta *= (1.0 + 0.05 * (rfq.n_dealers - cfg.n_dealers_mean))

    # Apply estimation error
    # Positive error = trader underestimates competition (thinks less steep)
    beta *= (1.0 - cfg.winrate_est_error)

    return beta


def estimate_win_probability(
    markup_bps: float,
    rfq: RFQEvent,
    cfg: SimConfig,
) -> float:
    """
    Estimate win probability at a given markup using logistic model.

    This is the trader's BELIEF about win probability, which may
    differ from reality (computed via competitors.py).

    P̂(win|m) = 1 / (1 + exp(-β * (c - m)))

    At markup = c, P(win) = 50%
    Below center (more aggressive), P(win) > 50%
    Above center (wider), P(win) < 50%

    Args:
        markup_bps: Trader's markup over theo in bps
        rfq: RFQ event
        cfg: SimConfig

    Returns:
        Estimated win probability in [0, 1]
    """
    center = compute_estimated_center(rfq, cfg)
    steepness = compute_estimated_steepness(rfq, cfg)

    # Logistic function
    # Note: (c - m) means aggressive quotes (low m) have high win prob
    z = steepness * (center - markup_bps)

    # Stable computation avoiding overflow
    if z > 20:
        return 1.0
    elif z < -20:
        return 0.0
    else:
        return 1.0 / (1.0 + np.exp(-z))


def compute_win_probability_gradient(
    markup_bps: float,
    rfq: RFQEvent,
    cfg: SimConfig,
) -> float:
    """
    Compute the gradient dP/dm of win probability w.r.t. markup.

    Useful for gradient-based optimization (though we use grid search).

    dP/dm = -β * P * (1 - P)

    Args:
        markup_bps: Current markup in bps
        rfq: RFQ event
        cfg: SimConfig

    Returns:
        Gradient (always negative - higher markup = lower win prob)
    """
    p = estimate_win_probability(markup_bps, rfq, cfg)
    beta = compute_estimated_steepness(rfq, cfg)

    return -beta * p * (1 - p)


def find_breakeven_markup(
    rfq: RFQEvent,
    cfg: SimConfig,
    target_win_prob: float = 0.5,
) -> float:
    """
    Find the markup that achieves a target win probability.

    Inverts the logistic function: m = c - ln(p/(1-p)) / β

    Args:
        rfq: RFQ event
        cfg: SimConfig
        target_win_prob: Desired win probability

    Returns:
        Markup in bps that achieves the target
    """
    center = compute_estimated_center(rfq, cfg)
    steepness = compute_estimated_steepness(rfq, cfg)

    # Clamp to avoid log(0)
    p = np.clip(target_win_prob, 0.001, 0.999)

    # Invert logistic
    markup = center - np.log(p / (1 - p)) / steepness

    return markup


def generate_estimated_win_curve(
    rfq: RFQEvent,
    cfg: SimConfig,
    markup_range_bps: tuple = (-10, 30),
    n_points: int = 41,
) -> tuple:
    """
    Generate the trader's estimated win-rate curve.

    Useful for visualization and comparison with empirical curve.

    Args:
        rfq: RFQ event
        cfg: SimConfig
        markup_range_bps: (min, max) markup range
        n_points: Number of points

    Returns:
        (markups, win_probs) arrays
    """
    markups = np.linspace(markup_range_bps[0], markup_range_bps[1], n_points)
    win_probs = np.array([
        estimate_win_probability(m, rfq, cfg) for m in markups
    ])

    return markups, win_probs
