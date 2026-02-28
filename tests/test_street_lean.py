"""Tests for street lean OU process."""

import numpy as np
import pytest

import sys
sys.path.insert(0, 'src')

from rfq_simulator.config import SimConfig
from rfq_simulator.world.street_lean import (
    generate_street_lean_path,
    get_street_lean_at_step,
    simulate_noisy_street_lean_observation,
)


class TestStreetLeanGeneration:
    """Tests for street lean path generation."""

    def test_path_length(self):
        """Street lean path should have one value per price step."""
        cfg = SimConfig(T_days=10)
        rng = np.random.default_rng(42)

        path = generate_street_lean_path(cfg, rng)

        assert len(path) == cfg.n_steps

    def test_mean_reversion(self):
        """Street lean should mean-revert toward equilibrium."""
        cfg = SimConfig(
            T_days=100,
            street_lean_mean_rev=0.5,  # Stronger mean reversion
            street_lean_vol_bps=1.0,   # Lower vol
            street_lean_eq=0.0,
        )
        rng = np.random.default_rng(42)

        path = generate_street_lean_path(cfg, rng)

        # Long-run average should be near equilibrium (allow wider tolerance)
        assert abs(np.mean(path) - cfg.street_lean_eq) < 2.0

    def test_volatility_reasonable(self):
        """Street lean volatility should be bounded."""
        cfg = SimConfig(T_days=60, street_lean_vol_bps=2.0)
        rng = np.random.default_rng(42)

        path = generate_street_lean_path(cfg, rng)

        # Std should be within reasonable range of target vol
        assert np.std(path) < 10.0  # bps

    def test_reproducibility(self):
        """Same seed should produce same path."""
        cfg = SimConfig(T_days=30)

        path1 = generate_street_lean_path(cfg, np.random.default_rng(123))
        path2 = generate_street_lean_path(cfg, np.random.default_rng(123))

        assert np.array_equal(path1, path2)


class TestStreetLeanAccess:
    """Tests for accessing street lean values."""

    def test_get_lean_at_valid_step(self):
        """Should return correct value at valid step."""
        cfg = SimConfig(T_days=10)
        rng = np.random.default_rng(42)

        path = generate_street_lean_path(cfg, rng)

        for step in [0, 100, len(path) - 1]:
            lean = get_street_lean_at_step(path, step)
            assert lean == path[step]

    def test_get_lean_clamps_negative(self):
        """Negative step should return first value."""
        cfg = SimConfig(T_days=10)
        rng = np.random.default_rng(42)

        path = generate_street_lean_path(cfg, rng)
        lean = get_street_lean_at_step(path, -5)

        assert lean == path[0]

    def test_get_lean_clamps_overflow(self):
        """Step beyond path should return last value."""
        cfg = SimConfig(T_days=10)
        rng = np.random.default_rng(42)

        path = generate_street_lean_path(cfg, rng)
        lean = get_street_lean_at_step(path, len(path) + 100)

        assert lean == path[-1]


class TestNoisyObservation:
    """Tests for noisy street lean observation."""

    def test_observation_centered_on_true(self):
        """Noisy observations should be centered on true value."""
        cfg = SimConfig(street_obs_noise=0.5, street_lean_vol_bps=2.0)
        rng = np.random.default_rng(42)
        true_lean = 5.0

        observations = [
            simulate_noisy_street_lean_observation(true_lean, cfg, rng)
            for _ in range(1000)
        ]

        # Mean should be close to true value
        assert abs(np.mean(observations) - true_lean) < 0.5

    def test_observation_has_noise(self):
        """Noisy observations should have non-zero variance."""
        cfg = SimConfig(street_obs_noise=0.5, street_lean_vol_bps=2.0)
        rng = np.random.default_rng(42)
        true_lean = 5.0

        observations = [
            simulate_noisy_street_lean_observation(true_lean, cfg, rng)
            for _ in range(100)
        ]

        # Should have variance
        assert np.std(observations) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
