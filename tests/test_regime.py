"""Tests for regime system (2-state Markov chain)."""

import numpy as np
import pytest

import sys
sys.path.insert(0, 'src')

from rfq_simulator.config import SimConfig
from rfq_simulator.world.regime import (
    Regime,
    generate_regime_path,
    get_regime_at_day,
    get_effective_ic,
    compute_stationary_distribution,
    compute_average_durations,
)


class TestRegimeGeneration:
    """Tests for regime path generation."""

    def test_regime_path_length(self):
        """Regime path should have one value per day."""
        cfg = SimConfig(T_days=60)
        rng = np.random.default_rng(42)

        path = generate_regime_path(cfg, rng)

        assert len(path) == cfg.T_days

    def test_regime_values_valid(self):
        """All regime values should be CALM (0) or STRESSED (1)."""
        cfg = SimConfig(T_days=100)
        rng = np.random.default_rng(42)

        path = generate_regime_path(cfg, rng)

        assert all(r in [Regime.CALM, Regime.STRESSED] for r in path)

    def test_stationary_distribution(self):
        """Long-run stress fraction should match stationary distribution."""
        cfg = SimConfig(
            T_days=10000,
            p_calm_to_stress=0.05,
            p_stress_to_calm=0.15,
        )
        rng = np.random.default_rng(42)

        path = generate_regime_path(cfg, rng)
        empirical_stress = np.mean(path == Regime.STRESSED)

        # Theoretical: π_stress = p_cs / (p_cs + p_sc) = 0.05 / 0.20 = 0.25
        expected_stress = cfg.p_calm_to_stress / (cfg.p_calm_to_stress + cfg.p_stress_to_calm)

        # Allow 2% tolerance for Monte Carlo
        assert abs(empirical_stress - expected_stress) < 0.02

    def test_reproducibility(self):
        """Same seed should produce same path."""
        cfg = SimConfig(T_days=60)

        path1 = generate_regime_path(cfg, np.random.default_rng(123))
        path2 = generate_regime_path(cfg, np.random.default_rng(123))

        assert np.array_equal(path1, path2)


class TestEffectiveIC:
    """Tests for regime-dependent IC."""

    def test_calm_ic(self):
        """IC in calm regime should equal base IC."""
        cfg = SimConfig(IC=0.10, IC_stress_mult=0.4)

        ic = get_effective_ic(Regime.CALM, cfg)

        assert ic == cfg.IC

    def test_stressed_ic(self):
        """IC in stressed regime should be reduced by multiplier."""
        cfg = SimConfig(IC=0.10, IC_stress_mult=0.4)

        ic = get_effective_ic(Regime.STRESSED, cfg)

        assert ic == cfg.IC * cfg.IC_stress_mult
        assert abs(ic - 0.04) < 1e-10  # Float comparison


class TestStationaryDistribution:
    """Tests for analytical distribution functions."""

    def test_stationary_probabilities_sum_to_one(self):
        """Stationary probabilities should sum to 1."""
        cfg = SimConfig(p_calm_to_stress=0.05, p_stress_to_calm=0.15)

        pi_calm, pi_stress = compute_stationary_distribution(cfg)

        assert abs(pi_calm + pi_stress - 1.0) < 1e-10

    def test_average_durations(self):
        """Average durations should be reciprocals of transition probs."""
        cfg = SimConfig(p_calm_to_stress=0.05, p_stress_to_calm=0.15)

        avg_calm, avg_stress = compute_average_durations(cfg)

        assert avg_calm == 1.0 / cfg.p_calm_to_stress  # 20 days
        assert avg_stress == 1.0 / cfg.p_stress_to_calm  # ~6.67 days


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
