"""Tests for AR(1) buy/sell imbalance process."""
import sys
sys.path.insert(0, 'src')

import numpy as np
import pytest
from rfq_simulator.config import ImbalanceConfig
from rfq_simulator.world.imbalance import ImbalanceProcess, compute_expected_buy_fraction
from rfq_simulator.world.regime import Regime


class TestImbalanceProcess:
    def test_initial_value_is_zero(self):
        cfg = ImbalanceConfig()
        rng = np.random.default_rng(42)
        proc = ImbalanceProcess(cfg, rng)
        assert proc.value == 0.0

    def test_step_changes_value(self):
        cfg = ImbalanceConfig(rho=0.5, sigma=0.3)
        rng = np.random.default_rng(42)
        proc = ImbalanceProcess(cfg, rng)
        proc.step()
        assert proc.value != 0.0

    def test_buy_probability_clipped(self):
        cfg = ImbalanceConfig(clip_low=0.2, clip_high=0.8)
        rng = np.random.default_rng(42)
        proc = ImbalanceProcess(cfg, rng)
        for _ in range(100):
            proc.step()
            p = proc.get_buy_probability()
            assert 0.2 <= p <= 0.8

    def test_regime_affects_mean(self):
        cfg = ImbalanceConfig(mu_calm=0.0, mu_stressed=-0.3, rho=0.0, sigma=0.01)
        rng = np.random.default_rng(42)

        # Calm regime
        proc_calm = ImbalanceProcess(cfg, rng, initial_regime=Regime.CALM)
        calm_values = [proc_calm.step() for _ in range(100)]

        # Stressed regime
        rng2 = np.random.default_rng(42)
        proc_stressed = ImbalanceProcess(cfg, rng2, initial_regime=Regime.STRESSED)
        stressed_values = [proc_stressed.step() for _ in range(100)]

        assert np.mean(calm_values) > np.mean(stressed_values)

    def test_reset_clears_state(self):
        cfg = ImbalanceConfig()
        rng = np.random.default_rng(42)
        proc = ImbalanceProcess(cfg, rng)
        proc.step()
        proc.reset()
        assert proc.value == 0.0


class TestExpectedBuyFraction:
    def test_calm_regime_balanced(self):
        cfg = ImbalanceConfig(mu_calm=0.0)
        expected = compute_expected_buy_fraction(cfg, Regime.CALM)
        assert expected == pytest.approx(0.5)

    def test_stressed_regime_sell_bias(self):
        cfg = ImbalanceConfig(mu_stressed=-0.25)
        expected = compute_expected_buy_fraction(cfg, Regime.STRESSED)
        assert expected < 0.5
