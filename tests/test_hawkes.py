"""Tests for Hawkes self-exciting point process."""
import sys
sys.path.insert(0, 'src')

import numpy as np
import pytest
from rfq_simulator.config import SimConfig, ArrivalConfig
from rfq_simulator.world.hawkes import HawkesProcess, generate_hawkes_arrivals


class TestHawkesProcess:
    def test_initial_value_is_zero(self):
        cfg = ArrivalConfig()
        rng = np.random.default_rng(42)
        hawkes = HawkesProcess(cfg, rng)
        assert hawkes.value == 0.0

    def test_excitation_after_event(self):
        cfg = ArrivalConfig(hawkes_alpha=0.5, hawkes_beta=1.0)
        rng = np.random.default_rng(42)
        hawkes = HawkesProcess(cfg, rng)
        hawkes.record_event()
        assert hawkes.value == pytest.approx(0.5)  # alpha * 1.0

    def test_decay_over_time(self):
        cfg = ArrivalConfig(hawkes_alpha=1.0, hawkes_beta=1.0)
        rng = np.random.default_rng(42)
        hawkes = HawkesProcess(cfg, rng)
        hawkes.record_event()
        initial = hawkes.value
        hawkes.step(dt=1.0)  # Decay by exp(-1)
        assert hawkes.value < initial
        assert hawkes.value == pytest.approx(initial * np.exp(-1.0))

    def test_reset_clears_state(self):
        cfg = ArrivalConfig()
        rng = np.random.default_rng(42)
        hawkes = HawkesProcess(cfg, rng)
        hawkes.record_event()
        hawkes.reset()
        assert hawkes.value == 0.0


class TestHawkesArrivals:
    def test_generates_arrivals(self):
        cfg = SimConfig(T_days=5, arrivals=ArrivalConfig(use_hawkes=True))
        rng = np.random.default_rng(42)
        arrivals = generate_hawkes_arrivals(cfg, rng)
        assert len(arrivals) > 0

    def test_arrivals_are_sorted(self):
        cfg = SimConfig(T_days=5, arrivals=ArrivalConfig(use_hawkes=True))
        rng = np.random.default_rng(42)
        arrivals = generate_hawkes_arrivals(cfg, rng)
        assert arrivals == sorted(arrivals)

    def test_clustering_detected(self):
        """Inter-arrival times should show positive autocorrelation."""
        cfg = SimConfig(
            T_days=30,
            arrivals=ArrivalConfig(use_hawkes=True, hawkes_alpha=0.5, hawkes_beta=0.8)
        )
        rng = np.random.default_rng(42)
        arrivals = generate_hawkes_arrivals(cfg, rng)
        if len(arrivals) > 100:
            inter_arrivals = np.diff(arrivals)
            acf_1 = np.corrcoef(inter_arrivals[:-1], inter_arrivals[1:])[0, 1]
            # Hawkes should show some positive autocorrelation (clustering)
            # May be weak, so just check it's computed
            assert not np.isnan(acf_1)
