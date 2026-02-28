"""Integration tests for realistic diagnostics."""
import pytest
import tempfile
import os

from rfq_simulator.config import SimConfig, ArrivalConfig, SpreadConfig, ImbalanceConfig
from rfq_simulator.simulation import run_simulation
from rfq_simulator.output import ValidationReport


class TestDiagnosticsIntegration:
    def test_full_validation_with_all_features(self):
        """Run full validation report with all features enabled."""
        cfg = SimConfig(
            T_days=30,
            seed=42,
            arrivals=ArrivalConfig(use_hawkes=True),
            spreads=SpreadConfig(use_lognormal=True),
            imbalance=ImbalanceConfig(use_ar1=True),
        )
        result = run_simulation(cfg)

        report = ValidationReport(result)
        results = report.run_all(generate_plots=False)

        assert len(results) == 3
        assert report.summary() != ""

        # Check all three diagnostics ran
        names = [r.name for r in results]
        assert "Hawkes Arrivals" in names
        assert "LogNormal Spreads" in names
        assert "AR(1) Imbalance" in names

    def test_partial_features_hawkes_only(self):
        """Run validation with only Hawkes enabled."""
        cfg = SimConfig(
            T_days=30,
            seed=42,
            arrivals=ArrivalConfig(use_hawkes=True),
            # spreads and imbalance use defaults (disabled)
        )
        result = run_simulation(cfg)

        report = ValidationReport(result)
        results = report.run_all()

        assert len(results) == 1
        assert results[0].name == "Hawkes Arrivals"

    def test_html_export(self):
        """Test HTML export produces valid file."""
        cfg = SimConfig(
            T_days=30,
            seed=42,
            arrivals=ArrivalConfig(use_hawkes=True),
            spreads=SpreadConfig(use_lognormal=True),
            imbalance=ImbalanceConfig(use_ar1=True),
        )
        result = run_simulation(cfg)

        report = ValidationReport(result)

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            html_path = f.name

        try:
            report.to_html(html_path)

            # Verify file exists and has content
            assert os.path.exists(html_path)
            with open(html_path, 'r') as f:
                content = f.read()
            assert len(content) > 1000  # Should have substantial content
            assert "Validation Report" in content
            assert "Hawkes" in content or "Spread" in content or "Imbalance" in content
        finally:
            if os.path.exists(html_path):
                os.unlink(html_path)

    def test_no_features_enabled(self):
        """Run validation with no realistic features enabled."""
        cfg = SimConfig(
            T_days=30,
            seed=42,
            # All features default to disabled
        )
        result = run_simulation(cfg)

        report = ValidationReport(result)
        results = report.run_all()

        # Should return empty list when no features enabled
        assert len(results) == 0
        assert report.summary() == "No diagnostics run. Call run_all() first." or "Validated:" not in report.summary()
