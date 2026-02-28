"""Tests for realistic distribution diagnostics."""
import pytest
import numpy as np
from rfq_simulator.config import SimConfig, ArrivalConfig, SpreadConfig, ImbalanceConfig
from rfq_simulator.simulation import run_simulation
from rfq_simulator.output.realistic_diagnostics import DiagnosticResult, HawkesDiagnostics, SpreadDiagnostics, ImbalanceDiagnostics, StreetLeanDiagnostics, ValidationReport


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


class TestImbalanceDiagnostics:
    @pytest.fixture
    def imbalance_result(self):
        cfg = SimConfig(
            T_days=30,
            seed=42,
            imbalance=ImbalanceConfig(use_ar1=True, rho=0.4, mu_stressed=-0.25),
        )
        return run_simulation(cfg)

    def test_imbalance_diagnostics_creation(self, imbalance_result):
        diag = ImbalanceDiagnostics(imbalance_result)
        assert diag is not None

    def test_imbalance_analyze_returns_result(self, imbalance_result):
        diag = ImbalanceDiagnostics(imbalance_result)
        result = diag.analyze()
        assert isinstance(result, DiagnosticResult)
        assert result.name == "AR(1) Imbalance"

    def test_imbalance_stats_include_acf(self, imbalance_result):
        diag = ImbalanceDiagnostics(imbalance_result)
        result = diag.analyze()
        assert "direction_acf1" in result.stats
        assert "buy_frac_calm" in result.stats
        assert "buy_frac_stressed" in result.stats

    def test_imbalance_regime_bias(self, imbalance_result):
        diag = ImbalanceDiagnostics(imbalance_result)
        result = diag.analyze()
        # Stressed should have lower buy fraction (sell bias)
        assert result.stats["buy_frac_stressed"] < result.stats["buy_frac_calm"]


class TestStreetLeanDiagnostics:
    @pytest.fixture
    def sim_result(self):
        cfg = SimConfig(
            T_days=30,
            seed=42,
            street_lean_vol_bps=3.0,
        )
        return run_simulation(cfg)

    def test_street_lean_diagnostics_creation(self, sim_result):
        diag = StreetLeanDiagnostics(sim_result)
        assert diag is not None

    def test_street_lean_analyze_returns_result(self, sim_result):
        diag = StreetLeanDiagnostics(sim_result)
        result = diag.analyze()
        assert isinstance(result, DiagnosticResult)
        assert result.name == "Street Lean"

    def test_street_lean_stats_include_volatility(self, sim_result):
        diag = StreetLeanDiagnostics(sim_result)
        result = diag.analyze()
        assert "observed_vol" in result.stats
        assert "configured_vol" in result.stats
        assert "mean_reversion_half_life" in result.stats

    def test_street_lean_generates_plots(self, sim_result):
        diag = StreetLeanDiagnostics(sim_result)
        result = diag.analyze(generate_plots=True)
        assert len(result.figures) >= 2


class TestValidationReport:
    @pytest.fixture
    def full_result(self):
        cfg = SimConfig(
            T_days=30,
            seed=42,
            arrivals=ArrivalConfig(use_hawkes=True),
            spreads=SpreadConfig(use_lognormal=True),
            imbalance=ImbalanceConfig(use_ar1=True),
        )
        return run_simulation(cfg)

    def test_validation_report_creation(self, full_result):
        report = ValidationReport(full_result)
        assert report is not None

    def test_run_all_returns_three_results(self, full_result):
        report = ValidationReport(full_result)
        results = report.run_all()
        assert len(results) == 3

    def test_summary_not_empty(self, full_result):
        report = ValidationReport(full_result)
        report.run_all()
        summary = report.summary()
        assert len(summary) > 50

    def test_all_passed_property(self, full_result):
        report = ValidationReport(full_result)
        report.run_all()
        assert isinstance(report.all_passed, bool)
