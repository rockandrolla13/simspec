"""
Regime process: 2-state Markov chain for calm/stressed market conditions.

Implements Eq 6-7 from spec:
    P = [[1 - p_cs, p_cs],
         [p_sc, 1 - p_sc]]

    ρ(r_t) = IC           if r_t = calm
           = IC * mult    if r_t = stressed

With default params (p_cs=0.05, p_sc=0.15):
- Stationary probability of stress: π_stress = p_cs/(p_cs + p_sc) = 0.05/0.20 = 25%
- Average stress duration: 1/p_sc ≈ 6.7 days
- Average calm duration: 1/p_cs = 20 days
"""

from enum import IntEnum
from typing import Tuple

import numpy as np
from numpy.random import Generator

from ..config import SimConfig


class Regime(IntEnum):
    """Market regime states."""

    CALM = 0
    STRESSED = 1


def generate_regime_path(cfg: SimConfig, rng: Generator) -> np.ndarray:
    """
    Generate a regime path with daily transitions.

    Args:
        cfg: SimConfig with p_calm_to_stress, p_stress_to_calm, T_days
        rng: Random generator

    Returns:
        np.ndarray of shape (T_days,) with Regime values
    """
    n_days = cfg.T_days

    # Transition matrix
    p_cs = cfg.p_calm_to_stress
    p_sc = cfg.p_stress_to_calm

    # Initialize from stationary distribution
    pi_stress = p_cs / (p_cs + p_sc)
    current_regime = Regime.STRESSED if rng.random() < pi_stress else Regime.CALM

    regimes = np.zeros(n_days, dtype=np.int32)
    regimes[0] = current_regime

    # Simulate transitions
    for day in range(1, n_days):
        u = rng.random()
        if current_regime == Regime.CALM:
            if u < p_cs:
                current_regime = Regime.STRESSED
        else:  # STRESSED
            if u < p_sc:
                current_regime = Regime.CALM
        regimes[day] = current_regime

    return regimes


def get_regime_at_day(regime_path: np.ndarray, day: int) -> Regime:
    """
    Get the regime for a given day.

    Args:
        regime_path: Array from generate_regime_path
        day: Day index (0 to T_days-1)

    Returns:
        Regime enum value
    """
    day = max(0, min(day, len(regime_path) - 1))
    return Regime(regime_path[day])


def get_effective_ic(regime: Regime, cfg: SimConfig) -> float:
    """
    Get the effective IC for a given regime.

    Eq 7: ρ(r_t) = IC if calm, IC * IC_stress_mult if stressed

    Args:
        regime: Current regime
        cfg: SimConfig with IC, IC_stress_mult

    Returns:
        Effective information coefficient
    """
    if regime == Regime.CALM:
        return cfg.IC
    else:
        return cfg.IC * cfg.IC_stress_mult


def compute_stationary_distribution(cfg: SimConfig) -> Tuple[float, float]:
    """
    Compute the stationary distribution of the regime chain.

    Args:
        cfg: SimConfig with transition probabilities

    Returns:
        (π_calm, π_stress) tuple
    """
    p_cs = cfg.p_calm_to_stress
    p_sc = cfg.p_stress_to_calm

    pi_stress = p_cs / (p_cs + p_sc)
    pi_calm = 1.0 - pi_stress

    return pi_calm, pi_stress


def compute_average_durations(cfg: SimConfig) -> Tuple[float, float]:
    """
    Compute average duration in each regime state.

    Args:
        cfg: SimConfig with transition probabilities

    Returns:
        (avg_calm_days, avg_stress_days) tuple
    """
    avg_calm = 1.0 / cfg.p_calm_to_stress
    avg_stress = 1.0 / cfg.p_stress_to_calm

    return avg_calm, avg_stress
