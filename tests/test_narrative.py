"""Tests for narrative generation helpers."""
import pytest
from rfq_simulator.output.narrative import format_narrative, format_warning


class TestNarrativeFormatting:
    def test_format_narrative_substitution(self):
        template = "ACF(1)={acf:.3f}, p-value={p:.4f}"
        result = format_narrative(template, acf=0.123, p=0.0456)
        assert result == "ACF(1)=0.123, p-value=0.0456"

    def test_format_warning(self):
        result = format_warning("ACF", 0.02, 0.05, "below threshold")
        assert "ACF" in result
        assert "0.02" in result
        assert "0.05" in result

    def test_format_narrative_missing_key(self):
        template = "Value={value:.2f}"
        result = format_narrative(template, other=1.0)
        assert "{value" in result  # Unformatted placeholder remains
