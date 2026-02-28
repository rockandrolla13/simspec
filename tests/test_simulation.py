"""Integration tests for the full simulation."""

import numpy as np
import pytest

import sys
sys.path.insert(0, 'src')

from rfq_simulator.config import SimConfig
from rfq_simulator.simulation.event_loop import run_simulation, SimulationResult
from rfq_simulator.simulation.baseline import run_baseline, compare_strategies
from rfq_simulator.world.regime import Regime


class TestSimulationRuns:
    """Tests that simulation runs successfully."""

    def test_basic_simulation(self):
        """Simulation should run with default config."""
        cfg = SimConfig(T_days=10, seed=42)
        cfg.validate()

        result = run_simulation(cfg)

        assert isinstance(result, SimulationResult)
        assert result.final_state is not None
        assert result.pnl is not None

    def test_simulation_with_v2_features(self):
        """Simulation should work with V2 features enabled."""
        cfg = SimConfig(
            T_days=15,
            alpha_horizon_days=5.0,
            aggress_window_hours=8.0,
            seed=42,
        )
        cfg.validate()

        result = run_simulation(cfg)

        # Street lean should be generated
        assert result.street_lean_path is not None
        assert len(result.street_lean_path) == cfg.n_steps

    def test_reproducibility(self):
        """Same seed should produce same results."""
        cfg = SimConfig(T_days=10, seed=123)

        result1 = run_simulation(cfg)
        result2 = run_simulation(cfg)

        assert result1.total_pnl == result2.total_pnl
        assert result1.final_state.n_rfqs_filled == result2.final_state.n_rfqs_filled


class TestPnLDecomposition:
    """Tests for P&L tracking."""

    def test_pnl_components_exist(self):
        """All P&L components should be tracked."""
        cfg = SimConfig(T_days=10, seed=42)
        result = run_simulation(cfg)

        pnl = result.pnl
        assert hasattr(pnl, 'alpha_pnl')
        assert hasattr(pnl, 'spread_pnl')
        assert hasattr(pnl, 'carry_pnl')
        assert hasattr(pnl, 'aggress_cost')
        assert hasattr(pnl, 'total_pnl')

    def test_pnl_decomposition_adds_up(self):
        """Total P&L should equal sum of components."""
        cfg = SimConfig(T_days=10, seed=42)
        result = run_simulation(cfg)

        pnl = result.pnl
        expected_total = (
            pnl.alpha_pnl + pnl.spread_pnl + pnl.carry_pnl + pnl.hedge_pnl
            - pnl.aggress_cost
        )

        assert abs(pnl.total_pnl - expected_total) < 0.01

    def test_aggress_cost_tracked(self):
        """Aggressive exit cost should be tracked when triggered."""
        # Use short horizon to trigger aggressive exit
        cfg = SimConfig(
            T_days=10,
            alpha_horizon_days=3.0,
            aggress_window_hours=8.0,
            seed=42,
        )
        result = run_simulation(cfg)

        # Should have some aggressive cost (may be 0 if no position at expiry)
        assert result.pnl.aggress_cost >= 0


class TestWorldGeneration:
    """Tests for world process generation."""

    def test_price_path_generated(self):
        """Price path should be generated with correct length."""
        cfg = SimConfig(T_days=10, seed=42)
        result = run_simulation(cfg)

        assert len(result.prices) == cfg.n_steps + 1

    def test_regime_path_generated(self):
        """Regime path should be generated with correct length."""
        cfg = SimConfig(T_days=10, seed=42)
        result = run_simulation(cfg)

        assert len(result.regime_path) == cfg.T_days

    def test_street_lean_path_generated(self):
        """Street lean path should be generated."""
        cfg = SimConfig(T_days=10, seed=42)
        result = run_simulation(cfg)

        assert result.street_lean_path is not None
        assert len(result.street_lean_path) == cfg.n_steps

    def test_rfq_events_generated(self):
        """RFQ events should be generated."""
        cfg = SimConfig(T_days=10, rfq_rate_per_day=15.0, seed=42)
        result = run_simulation(cfg)

        # Should have some RFQs
        assert len(result.rfq_events) > 0

        # Approximate expected count
        expected = cfg.rfq_rate_per_day * cfg.T_days
        assert 0.5 * expected < len(result.rfq_events) < 2.0 * expected


class TestRFQProcessing:
    """Tests for RFQ processing logic."""

    def test_rfqs_processed(self):
        """All RFQs should be processed."""
        cfg = SimConfig(T_days=10, seed=42)
        result = run_simulation(cfg)

        state = result.final_state
        assert state.n_rfqs_seen == len(result.rfq_events)

    def test_fill_rate_reasonable(self):
        """Fill rate should be in reasonable range."""
        cfg = SimConfig(T_days=10, seed=42)
        result = run_simulation(cfg)

        fill_rate = result.final_state.get_fill_rate()

        # Fill rate should be between 20% and 80% typically
        assert 0.1 < fill_rate < 0.9

    def test_rfq_log_populated(self):
        """RFQ log should have entries for all RFQs."""
        cfg = SimConfig(T_days=10, seed=42)
        result = run_simulation(cfg)

        assert len(result.final_state.rfq_log) == len(result.rfq_events)


class TestBaseline:
    """Tests for baseline strategy."""

    def test_baseline_runs(self):
        """Baseline should run successfully."""
        cfg = SimConfig(T_days=10, seed=42)
        lp_result = run_simulation(cfg)

        baseline = run_baseline(
            prices=lp_result.prices,
            regime_path=lp_result.regime_path,
            cfg=cfg,
            seed=cfg.seed,
        )

        assert baseline is not None
        assert baseline.total_pnl is not None

    def test_baseline_uses_original_prices(self):
        """Baseline should use unmutated price path."""
        cfg = SimConfig(T_days=10, seed=42)

        # Generate a fresh price path
        rng = np.random.default_rng(cfg.seed)
        from rfq_simulator.world.price import generate_price_path
        original_prices = generate_price_path(cfg, rng)
        prices_for_baseline = original_prices.copy()

        # Run baseline
        from rfq_simulator.world.regime import generate_regime_path
        regime_path = generate_regime_path(cfg, rng)

        baseline = run_baseline(
            prices=prices_for_baseline,
            regime_path=regime_path,
            cfg=cfg,
            seed=cfg.seed,
        )

        # Baseline should not modify the prices it receives
        assert np.array_equal(prices_for_baseline, original_prices)
        assert baseline is not None

    def test_compare_strategies(self):
        """Strategy comparison should return valid metrics."""
        cfg = SimConfig(T_days=10, seed=42)
        lp_result = run_simulation(cfg)

        baseline = run_baseline(
            prices=lp_result.prices.copy(),  # Use copy to avoid mutation issues
            regime_path=lp_result.regime_path,
            cfg=cfg,
            seed=cfg.seed,
        )

        comparison = compare_strategies(lp_result, baseline)

        assert 'lp_total_pnl' in comparison
        assert 'baseline_total_pnl' in comparison
        assert 'spread_minus_alpha_loss' in comparison


class TestSummaryMetrics:
    """Tests for summary metrics."""

    def test_summary_dict(self):
        """Summary should contain all expected fields."""
        cfg = SimConfig(T_days=10, seed=42)
        result = run_simulation(cfg)

        summary = result.summary()

        expected_keys = [
            'total_pnl', 'alpha_pnl', 'spread_pnl', 'carry_pnl',
            'aggress_cost', 'n_rfqs', 'n_fills', 'fill_rate',
            'avg_spread_bps', 'final_inventory', 'sharpe', 'max_drawdown'
        ]

        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
