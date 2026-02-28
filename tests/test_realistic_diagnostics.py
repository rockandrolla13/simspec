"""Tests for realistic distribution diagnostics."""
import pytest
import numpy as np
from rfq_simulator.config import SimConfig, ArrivalConfig, SpreadConfig
from rfq_simulator.simulation import run_simulation
from rfq_simulator.output.realistic_diagnostics import DiagnosticResult, HawkesDiagnostics, SpreadDiagnostics


class TestDiagnosticResult:
    def test_diagnostic_result_creation(self):
        result = DiagnosticResult(
            name="Test Diagnostic",
            passed=True,
            stats={"acf_lag1": 0.12, "p_value": 0.03},
            figures=[],
            narrative="Test narrative",
            warnings=[],
        )
        assert result.name == "Test Diagnostic"
        assert result.passed is True
        assert result.stats["acf_lag1"] == 0.12

    def test_diagnostic_result_with_warnings(self):
        result = DiagnosticResult(
            name="Test",
            passed=False,
            stats={},
            figures=[],
            narrative="",
            warnings=["Warning 1", "Warning 2"],
        )
        assert result.passed is False
        assert len(result.warnings) == 2


class TestHawkesDiagnostics:
    @pytest.fixture
    def hawkes_result(self):
        cfg = SimConfig(
            T_days=30,
            seed=42,
            arrivals=ArrivalConfig(use_hawkes=True, hawkes_alpha=0.4, hawkes_beta=0.8),
        )
        return run_simulation(cfg)

    def test_hawkes_diagnostics_creation(self, hawkes_result):
        diag = HawkesDiagnostics(hawkes_result)
        assert diag is not None

    def test_hawkes_analyze_returns_result(self, hawkes_result):
        diag = HawkesDiagnostics(hawkes_result)
        result = diag.analyze()
        assert isinstance(result, DiagnosticResult)
        assert result.name == "Hawkes Arrivals"

    def test_hawkes_stats_include_acf(self, hawkes_result):
        diag = HawkesDiagnostics(hawkes_result)
        result = diag.analyze()
        assert "acf_lag1" in result.stats
        assert "ljung_box_p" in result.stats

    def test_hawkes_narrative_not_empty(self, hawkes_result):
        diag = HawkesDiagnostics(hawkes_result)
        result = diag.analyze()
        assert len(result.narrative) > 0

    def test_hawkes_generates_figures(self, hawkes_result):
        diag = HawkesDiagnostics(hawkes_result)
        result = diag.analyze(generate_plots=True)
        assert len(result.figures) == 4  # histogram, ACF, heatmap, QQ

    def test_hawkes_plot_titles(self, hawkes_result):
        diag = HawkesDiagnostics(hawkes_result)
        result = diag.analyze(generate_plots=True)
        # Check that figures have expected titles
        titles = []
        for fig in result.figures:
            if fig.axes:
                titles.append(fig.axes[0].get_title())
        assert any("Inter-Arrival" in t for t in titles)
        assert any("ACF" in t for t in titles)


class TestSpreadDiagnostics:
    @pytest.fixture
    def spread_result(self):
        cfg = SimConfig(
            T_days=30,
            seed=42,
            spreads=SpreadConfig(use_lognormal=True),
        )
        return run_simulation(cfg)

    def test_spread_diagnostics_creation(self, spread_result):
        diag = SpreadDiagnostics(spread_result)
        assert diag is not None

    def test_spread_analyze_returns_result(self, spread_result):
        diag = SpreadDiagnostics(spread_result)
        result = diag.analyze()
        assert isinstance(result, DiagnosticResult)
        assert result.name == "LogNormal Spreads"

    def test_spread_stats_include_shapiro(self, spread_result):
        diag = SpreadDiagnostics(spread_result)
        result = diag.analyze()
        assert "shapiro_p" in result.stats
        assert "calm_median" in result.stats
        assert "stressed_median" in result.stats

    def test_spread_regime_difference(self, spread_result):
        diag = SpreadDiagnostics(spread_result)
        result = diag.analyze()
        # Stressed should be wider than calm (from sampled spread distribution)
        assert result.stats["stressed_median"] > result.stats["calm_median"]
        # Verify regime ratio is significant (config has mu_stressed=3.2 vs mu_calm=2.0)
        assert result.stats["regime_ratio"] > 2.0
