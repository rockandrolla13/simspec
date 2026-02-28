"""Tests for log-normal spread distribution."""
import sys
sys.path.insert(0, 'src')

import numpy as np
import pytest
from rfq_simulator.config import SpreadConfig
from rfq_simulator.world.spread import (
    sample_base_spread,
    sample_dealer_spread,
    apply_size_adjustment,
    compute_expected_spread,
    compute_median_spread,
)
from rfq_simulator.world.regime import Regime


class TestSpreadSampling:
    def test_spread_always_positive(self):
        cfg = SpreadConfig()
        rng = np.random.default_rng(42)
        for _ in range(100):
            spread = sample_base_spread(Regime.CALM, cfg, rng)
            assert spread > 0

    def test_stressed_spreads_wider(self):
        cfg = SpreadConfig(mu_calm=2.0, mu_stressed=3.2)
        rng = np.random.default_rng(42)

        calm_spreads = [sample_base_spread(Regime.CALM, cfg, rng) for _ in range(500)]
        stressed_spreads = [sample_base_spread(Regime.STRESSED, cfg, rng) for _ in range(500)]

        assert np.mean(stressed_spreads) > np.mean(calm_spreads)

    def test_size_adjustment_increases_spread(self):
        cfg = SpreadConfig(size_gamma=0.15)
        base = 10.0
        adjusted = apply_size_adjustment(base, size=100, cfg=cfg)
        assert adjusted > base

    def test_sample_dealer_spread_combines_both(self):
        cfg = SpreadConfig()
        rng = np.random.default_rng(42)
        spread = sample_dealer_spread(Regime.CALM, size=50, cfg=cfg, rng=rng)
        assert spread > 0


class TestSpreadMoments:
    def test_expected_spread_formula(self):
        cfg = SpreadConfig(mu_calm=2.0, sigma_calm=0.5)
        expected = compute_expected_spread(Regime.CALM, cfg)
        # E[X] = exp(mu + sigma^2/2) = exp(2.0 + 0.125) = exp(2.125)
        assert expected == pytest.approx(np.exp(2.125))

    def test_median_spread_formula(self):
        cfg = SpreadConfig(mu_calm=2.0)
        median = compute_median_spread(Regime.CALM, cfg)
        # Median = exp(mu) = exp(2.0)
        assert median == pytest.approx(np.exp(2.0))

    def test_stressed_moments_higher(self):
        cfg = SpreadConfig(mu_calm=2.0, mu_stressed=3.2)
        calm_expected = compute_expected_spread(Regime.CALM, cfg)
        stressed_expected = compute_expected_spread(Regime.STRESSED, cfg)
        assert stressed_expected > 2 * calm_expected  # Stressed should be >2x calm
