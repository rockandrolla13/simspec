"""Tests for realistic distribution diagnostics."""
import pytest
from rfq_simulator.output.realistic_diagnostics import DiagnosticResult


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
