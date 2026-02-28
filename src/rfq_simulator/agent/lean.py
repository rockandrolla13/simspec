"""
Lean computation: Adjusts theo toward target position.

Implements Eq 13-15 from spec:
    lean = -λ_eff * (q - q*)

where:
    λ_eff = λ_base * vol_scaling * urgency * convexity * soft_limit

Components:
    - vol_scaling: (σ_daily / σ_ref)² - lean harder in volatile markets
    - urgency: (1 + κ_urg * (t - t_signal) / H) - lean harder as signal ages
    - convexity: (1 + κ_conv * |q - q*|) - lean harder when far from target
    - soft_limit: exponential penalty near position limits
"""

import numpy as np

from ..config import SimConfig


def compute_vol_scaling(cfg: SimConfig) -> float:
    """
    Compute volatility scaling factor.

    Eq 13: vol_scale = (σ_daily / σ_ref)²

    Lean harder in volatile markets to manage risk.

    Args:
        cfg: SimConfig with sigma_daily_bps, sigma_ref_bps

    Returns:
        Volatility scaling factor (≥ 0)
    """
    return (cfg.sigma_daily_bps / cfg.sigma_ref_bps) ** 2


def compute_urgency_factor(
    signal_age_minutes: float,
    horizon_minutes: float,
    cfg: SimConfig,
) -> float:
    """
    Compute urgency factor based on signal age.

    Eq 14: urgency = 1 + κ_urg * (t - t_signal) / H

    As the signal approaches expiry, urgency increases linearly.
    At signal generation: urgency = 1
    At signal expiry: urgency = 1 + κ_urg

    Args:
        signal_age_minutes: Time since signal was generated
        horizon_minutes: Total signal horizon H
        cfg: SimConfig with kappa_urgency

    Returns:
        Urgency factor (≥ 1)
    """
    if horizon_minutes <= 0:
        return 1.0

    # Clamp age to [0, H]
    age_fraction = np.clip(signal_age_minutes / horizon_minutes, 0, 1)

    return 1.0 + cfg.kappa_urgency * age_fraction


def compute_convexity_factor(
    inventory_gap: float,
    cfg: SimConfig,
) -> float:
    """
    Compute convexity factor based on inventory deviation.

    Eq 15: convexity = 1 + κ_conv * |q - q*|

    Lean harder when further from target position.

    Args:
        inventory_gap: |q - q*| in lots
        cfg: SimConfig with kappa_convexity

    Returns:
        Convexity factor (≥ 1)
    """
    return 1.0 + cfg.kappa_convexity * abs(inventory_gap)


def compute_soft_limit_penalty(
    current_q: float,
    cfg: SimConfig,
) -> float:
    """
    Compute soft limit penalty near position limits.

    When |q| > θ * q_max, apply exponential penalty to encourage
    reducing position before hitting hard limits.

    Penalty = exp(κ_limit * (|q|/q_max - θ))  if |q| > θ * q_max
            = 1.0                               otherwise

    Args:
        current_q: Current inventory in lots
        cfg: SimConfig with theta_limit, kappa_limit, q_max

    Returns:
        Soft limit penalty (≥ 1)
    """
    position_frac = abs(current_q) / cfg.q_max
    threshold = cfg.theta_limit

    if position_frac <= threshold:
        return 1.0

    # Exponential penalty for positions beyond threshold
    excess = position_frac - threshold
    return np.exp(cfg.kappa_limit * excess)


def compute_effective_lambda(
    current_q: float,
    target_q: float,
    signal_age_minutes: float,
    horizon_minutes: float,
    cfg: SimConfig,
) -> float:
    """
    Compute effective lean coefficient.

    λ_eff = λ_base * vol_scale * urgency * convexity * soft_limit

    All factors combine multiplicatively to determine how aggressively
    to lean toward the target position.

    Args:
        current_q: Current inventory
        target_q: Target inventory
        signal_age_minutes: Time since signal was generated
        horizon_minutes: Signal horizon H
        cfg: SimConfig

    Returns:
        Effective lean coefficient in bps per lot
    """
    inventory_gap = abs(current_q - target_q)

    vol_scale = compute_vol_scaling(cfg)
    urgency = compute_urgency_factor(signal_age_minutes, horizon_minutes, cfg)
    convexity = compute_convexity_factor(inventory_gap, cfg)
    soft_limit = compute_soft_limit_penalty(current_q, cfg)

    return cfg.lambda_base_bps * vol_scale * urgency * convexity * soft_limit


def compute_lean(
    current_q: float,
    target_q: float,
    signal_age_minutes: float,
    horizon_minutes: float,
    cfg: SimConfig,
) -> float:
    """
    Compute the lean adjustment in dollars.

    lean = -λ_eff * (q - q*)

    Positive lean: theo increases (better for selling/offering)
    Negative lean: theo decreases (better for buying/bidding)

    When q > q* (too long): negative lean → lower theo → buy cheaper
    When q < q* (too short): positive lean → higher theo → sell higher

    Args:
        current_q: Current inventory in lots
        target_q: Target inventory in lots
        signal_age_minutes: Time since signal generation
        horizon_minutes: Signal horizon H
        cfg: SimConfig

    Returns:
        Lean adjustment in dollars
    """
    lambda_eff = compute_effective_lambda(
        current_q, target_q, signal_age_minutes, horizon_minutes, cfg
    )

    # Inventory gap: positive if we're long relative to target
    gap = current_q - target_q

    # Lean in bps
    lean_bps = -lambda_eff * gap

    # Convert to dollars
    lean_dollars = lean_bps / 10000.0 * cfg.p0

    return lean_dollars


def compute_lean_decomposition(
    current_q: float,
    target_q: float,
    signal_age_minutes: float,
    horizon_minutes: float,
    cfg: SimConfig,
) -> dict:
    """
    Compute lean with full decomposition of factors.

    Useful for diagnostics and understanding lean behavior.

    Args:
        current_q: Current inventory
        target_q: Target inventory
        signal_age_minutes: Signal age
        horizon_minutes: Signal horizon
        cfg: SimConfig

    Returns:
        Dictionary with all lean components
    """
    gap = current_q - target_q

    vol_scale = compute_vol_scaling(cfg)
    urgency = compute_urgency_factor(signal_age_minutes, horizon_minutes, cfg)
    convexity = compute_convexity_factor(abs(gap), cfg)
    soft_limit = compute_soft_limit_penalty(current_q, cfg)

    lambda_eff = cfg.lambda_base_bps * vol_scale * urgency * convexity * soft_limit
    lean_bps = -lambda_eff * gap
    lean_dollars = lean_bps / 10000.0 * cfg.p0

    return {
        "lean_dollars": lean_dollars,
        "lean_bps": lean_bps,
        "lambda_eff": lambda_eff,
        "lambda_base": cfg.lambda_base_bps,
        "vol_scale": vol_scale,
        "urgency": urgency,
        "convexity": convexity,
        "soft_limit": soft_limit,
        "inventory_gap": gap,
    }


def estimate_lean_for_target_deviation(
    target_deviation_lots: float,
    signal_age_fraction: float,
    cfg: SimConfig,
) -> float:
    """
    Estimate lean for a hypothetical position deviation.

    Useful for sensitivity analysis.

    Args:
        target_deviation_lots: Hypothetical |q - q*|
        signal_age_fraction: Fraction of signal life elapsed (0 to 1)
        cfg: SimConfig

    Returns:
        Approximate lean in bps
    """
    vol_scale = compute_vol_scaling(cfg)
    urgency = 1.0 + cfg.kappa_urgency * signal_age_fraction
    convexity = 1.0 + cfg.kappa_convexity * abs(target_deviation_lots)

    lambda_eff = cfg.lambda_base_bps * vol_scale * urgency * convexity

    return lambda_eff * target_deviation_lots
