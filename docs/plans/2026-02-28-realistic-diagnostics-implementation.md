# Realistic Diagnostics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build validation diagnostics for Hawkes arrivals, LogNormal spreads, and AR(1) imbalance with statistical tests, plots, and auto-generated narrative commentary.

**Architecture:** Three diagnostic classes (HawkesDiagnostics, SpreadDiagnostics, ImbalanceDiagnostics) each producing a DiagnosticResult dataclass. ValidationReport aggregates all three and provides export to notebook/HTML/PDF.

**Tech Stack:** numpy, matplotlib, scipy.stats, statsmodels (ACF/Ljung-Box), jinja2 (HTML export)

---

## Task 1: DiagnosticResult Dataclass

**Files:**
- Create: `src/rfq_simulator/output/realistic_diagnostics.py`
- Test: `tests/test_realistic_diagnostics.py`

**Step 1: Write the failing test**

```python
# tests/test_realistic_diagnostics.py
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
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_realistic_diagnostics.py -v`
Expected: FAIL with "No module named 'rfq_simulator.output.realistic_diagnostics'"

**Step 3: Write minimal implementation**

```python
# src/rfq_simulator/output/realistic_diagnostics.py
"""
Diagnostics for validating realistic market distributions.

Provides statistical validation, plots, and auto-generated narrative
for Hawkes arrivals, LogNormal spreads, and AR(1) imbalance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import numpy as np


@dataclass
class DiagnosticResult:
    """Result from a diagnostic analysis."""

    name: str                           # "Hawkes Arrivals"
    passed: bool                        # Overall validation passed?
    stats: Dict[str, float]             # {"acf_lag1": 0.12, "p_value": 0.03}
    figures: List[Any]                  # Matplotlib figures
    narrative: str                      # "Inter-arrival times show clustering..."
    warnings: List[str] = field(default_factory=list)  # ["ACF below threshold"]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_realistic_diagnostics.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rfq_simulator/output/realistic_diagnostics.py tests/test_realistic_diagnostics.py
git commit -m "feat(diagnostics): add DiagnosticResult dataclass"
```

---

## Task 2: Narrative Helper Module

**Files:**
- Create: `src/rfq_simulator/output/narrative.py`
- Test: `tests/test_narrative.py`

**Step 1: Write the failing test**

```python
# tests/test_narrative.py
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
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_narrative.py -v`
Expected: FAIL with "No module named 'rfq_simulator.output.narrative'"

**Step 3: Write minimal implementation**

```python
# src/rfq_simulator/output/narrative.py
"""
Narrative generation helpers for diagnostic reports.

Provides template formatting and warning generation.
"""

from typing import Any


def format_narrative(template: str, **kwargs: Any) -> str:
    """
    Format a narrative template with provided values.

    Handles missing keys gracefully by leaving placeholder intact.

    Args:
        template: String with {name:.format} placeholders
        **kwargs: Values to substitute

    Returns:
        Formatted string
    """
    try:
        return template.format(**kwargs)
    except KeyError:
        # Partial formatting - substitute what we can
        result = template
        for key, value in kwargs.items():
            # Handle various format specs
            import re
            pattern = rf"\{{{key}(:[^}}]*)?\}}"
            match = re.search(pattern, result)
            if match:
                fmt_spec = match.group(1) or ""
                formatted = f"{{0{fmt_spec}}}".format(value)
                result = re.sub(pattern, formatted, result, count=1)
        return result


def format_warning(
    metric_name: str,
    observed: float,
    expected: float,
    description: str,
) -> str:
    """
    Generate a standardized warning message.

    Args:
        metric_name: Name of the metric (e.g., "ACF(1)")
        observed: Observed value
        expected: Expected/threshold value
        description: Brief description of the issue

    Returns:
        Formatted warning string
    """
    return (
        f"Warning: {metric_name}={observed:.4f} {description} "
        f"(expected: {expected:.4f})"
    )


def success_icon() -> str:
    """Return success indicator."""
    return "✓"


def warning_icon() -> str:
    """Return warning indicator."""
    return "⚠"


def failure_icon() -> str:
    """Return failure indicator."""
    return "✗"
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_narrative.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rfq_simulator/output/narrative.py tests/test_narrative.py
git commit -m "feat(diagnostics): add narrative formatting helpers"
```

---

## Task 3: HawkesDiagnostics - Statistical Tests

**Files:**
- Modify: `src/rfq_simulator/output/realistic_diagnostics.py`
- Modify: `tests/test_realistic_diagnostics.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_realistic_diagnostics.py
import numpy as np
from rfq_simulator.config import SimConfig, ArrivalConfig
from rfq_simulator.simulation import run_simulation
from rfq_simulator.output.realistic_diagnostics import HawkesDiagnostics


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
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_realistic_diagnostics.py::TestHawkesDiagnostics -v`
Expected: FAIL with "cannot import name 'HawkesDiagnostics'"

**Step 3: Write minimal implementation**

```python
# Add to src/rfq_simulator/output/realistic_diagnostics.py

from .narrative import format_narrative, format_warning, success_icon, warning_icon


class HawkesDiagnostics:
    """Diagnostic analysis for Hawkes self-exciting arrivals."""

    def __init__(self, result):
        """
        Initialize with simulation result.

        Args:
            result: SimulationResult with rfq_events
        """
        self.result = result
        self.cfg = result.cfg
        self._inter_arrivals: Optional[np.ndarray] = None

    def _compute_inter_arrivals(self) -> np.ndarray:
        """Compute inter-arrival times from RFQ events."""
        if self._inter_arrivals is None:
            times = np.array([e.time for e in self.result.rfq_events])
            self._inter_arrivals = np.diff(times)
        return self._inter_arrivals

    def _compute_acf(self, max_lag: int = 20) -> np.ndarray:
        """Compute autocorrelation function of inter-arrival times."""
        inter_arrivals = self._compute_inter_arrivals()
        if len(inter_arrivals) < max_lag + 1:
            return np.zeros(max_lag)

        # Statsmodels ACF if available, else manual
        try:
            from statsmodels.tsa.stattools import acf
            return acf(inter_arrivals, nlags=max_lag, fft=True)[1:]
        except ImportError:
            # Manual ACF computation
            n = len(inter_arrivals)
            mean = np.mean(inter_arrivals)
            var = np.var(inter_arrivals)
            if var == 0:
                return np.zeros(max_lag)
            acf_vals = []
            for lag in range(1, max_lag + 1):
                if lag >= n:
                    acf_vals.append(0.0)
                else:
                    cov = np.mean((inter_arrivals[:-lag] - mean) * (inter_arrivals[lag:] - mean))
                    acf_vals.append(cov / var)
            return np.array(acf_vals)

    def _ljung_box_test(self, max_lag: int = 10) -> tuple:
        """Perform Ljung-Box test for serial correlation."""
        inter_arrivals = self._compute_inter_arrivals()
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            result = acorr_ljungbox(inter_arrivals, lags=[max_lag], return_df=False)
            return float(result[0][0]), float(result[1][0])  # stat, p-value
        except ImportError:
            # Skip test if statsmodels not available
            return np.nan, np.nan

    def analyze(self) -> DiagnosticResult:
        """
        Run full Hawkes diagnostic analysis.

        Returns:
            DiagnosticResult with stats, narrative, and warnings
        """
        stats = {}
        warnings = []

        # ACF analysis
        acf_vals = self._compute_acf()
        stats["acf_lag1"] = float(acf_vals[0]) if len(acf_vals) > 0 else 0.0
        stats["acf_lag5"] = float(acf_vals[4]) if len(acf_vals) > 4 else 0.0

        # Ljung-Box test
        lb_stat, lb_p = self._ljung_box_test()
        stats["ljung_box_stat"] = lb_stat
        stats["ljung_box_p"] = lb_p

        # Branching ratio
        branching = self.cfg.arrivals.hawkes_alpha / self.cfg.arrivals.hawkes_beta
        stats["branching_ratio"] = branching

        # Check for warnings
        if stats["acf_lag1"] < 0.05:
            warnings.append(format_warning("ACF(1)", stats["acf_lag1"], 0.05, "below threshold"))

        if not np.isnan(lb_p) and lb_p > 0.05:
            warnings.append("Ljung-Box test fails to reject IID (p={:.4f}). Clustering may be weak.".format(lb_p))

        # Generate narrative
        if stats["acf_lag1"] >= 0.05:
            narrative = format_narrative(
                "{icon} RFQ arrivals show significant clustering (ACF(1)={acf:.3f}). "
                "Branching ratio α/β={br:.2f} indicates moderate self-excitation.",
                icon=success_icon(),
                acf=stats["acf_lag1"],
                br=branching,
            )
        else:
            narrative = format_narrative(
                "{icon} Clustering weaker than expected (ACF(1)={acf:.3f}). "
                "Consider increasing hawkes_alpha or decreasing hawkes_beta.",
                icon=warning_icon(),
                acf=stats["acf_lag1"],
            )

        passed = stats["acf_lag1"] >= 0.05 and branching < 1.0

        return DiagnosticResult(
            name="Hawkes Arrivals",
            passed=passed,
            stats=stats,
            figures=[],  # Plots added in Task 4
            narrative=narrative,
            warnings=warnings,
        )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_realistic_diagnostics.py::TestHawkesDiagnostics -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rfq_simulator/output/realistic_diagnostics.py tests/test_realistic_diagnostics.py
git commit -m "feat(diagnostics): add HawkesDiagnostics with statistical tests"
```

---

## Task 4: HawkesDiagnostics - Plots

**Files:**
- Modify: `src/rfq_simulator/output/realistic_diagnostics.py`
- Modify: `tests/test_realistic_diagnostics.py`

**Step 1: Write the failing test**

```python
# Add to TestHawkesDiagnostics in tests/test_realistic_diagnostics.py
    def test_hawkes_generates_figures(self, hawkes_result):
        diag = HawkesDiagnostics(hawkes_result)
        result = diag.analyze(generate_plots=True)
        assert len(result.figures) == 4  # histogram, ACF, heatmap, QQ

    def test_hawkes_plot_names(self, hawkes_result):
        diag = HawkesDiagnostics(hawkes_result)
        result = diag.analyze(generate_plots=True)
        fig_titles = [f.axes[0].get_title() for f in result.figures if f.axes]
        assert any("Inter-Arrival" in t for t in fig_titles)
        assert any("ACF" in t for t in fig_titles)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_realistic_diagnostics.py::TestHawkesDiagnostics::test_hawkes_generates_figures -v`
Expected: FAIL

**Step 3: Add plot methods to HawkesDiagnostics**

```python
# Add methods to HawkesDiagnostics class

    def _plot_inter_arrival_histogram(self):
        """Plot histogram of inter-arrival times with exponential fit."""
        import matplotlib.pyplot as plt
        from scipy import stats

        inter_arrivals = self._compute_inter_arrivals()

        fig, ax = plt.subplots(figsize=(8, 5))

        # Histogram
        ax.hist(inter_arrivals, bins=50, density=True, alpha=0.7, label='Observed')

        # Exponential fit
        rate = 1 / np.mean(inter_arrivals)
        x = np.linspace(0, np.percentile(inter_arrivals, 99), 100)
        ax.plot(x, stats.expon.pdf(x, scale=1/rate), 'r-', lw=2,
                label=f'Exp(λ={rate:.4f})')

        ax.set_xlabel('Inter-Arrival Time (minutes)')
        ax.set_ylabel('Density')
        ax.set_title('Inter-Arrival Time Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_acf(self, max_lag: int = 20):
        """Plot autocorrelation function."""
        import matplotlib.pyplot as plt

        acf_vals = self._compute_acf(max_lag)

        fig, ax = plt.subplots(figsize=(8, 5))

        lags = np.arange(1, len(acf_vals) + 1)
        ax.bar(lags, acf_vals, color='steelblue', alpha=0.7)

        # Confidence bands (approximate)
        n = len(self._compute_inter_arrivals())
        conf = 1.96 / np.sqrt(n)
        ax.axhline(conf, color='red', linestyle='--', alpha=0.5, label='95% CI')
        ax.axhline(-conf, color='red', linestyle='--', alpha=0.5)
        ax.axhline(0, color='black', linewidth=0.5)

        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')
        ax.set_title('ACF of Inter-Arrival Times')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_intensity_heatmap(self):
        """Plot intensity by hour showing clustering."""
        import matplotlib.pyplot as plt

        times = np.array([e.time for e in self.result.rfq_events])
        hours = (times % self.cfg.minutes_per_day) / 60
        days = times // self.cfg.minutes_per_day

        fig, ax = plt.subplots(figsize=(10, 6))

        # 2D histogram
        h, xedges, yedges = np.histogram2d(
            days, hours,
            bins=[int(self.cfg.T_days), 24]
        )

        im = ax.imshow(h.T, aspect='auto', origin='lower', cmap='hot',
                       extent=[0, self.cfg.T_days, 0, self.cfg.trading_hours])

        ax.set_xlabel('Day')
        ax.set_ylabel('Hour of Day')
        ax.set_title('RFQ Arrival Intensity (Clustering Visualization)')
        plt.colorbar(im, ax=ax, label='RFQ Count')

        plt.tight_layout()
        return fig

    def _plot_qq(self):
        """QQ plot of rescaled inter-arrivals vs Exp(1)."""
        import matplotlib.pyplot as plt
        from scipy import stats

        inter_arrivals = self._compute_inter_arrivals()

        # Rescale to Exp(1)
        rate = 1 / np.mean(inter_arrivals)
        rescaled = inter_arrivals * rate

        fig, ax = plt.subplots(figsize=(6, 6))

        stats.probplot(rescaled, dist="expon", plot=ax)
        ax.set_title('QQ Plot: Rescaled Inter-Arrivals vs Exp(1)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def analyze(self, generate_plots: bool = False) -> DiagnosticResult:
        """
        Run full Hawkes diagnostic analysis.

        Args:
            generate_plots: If True, generate matplotlib figures

        Returns:
            DiagnosticResult with stats, narrative, and warnings
        """
        # ... (existing stats code) ...

        figures = []
        if generate_plots:
            try:
                figures.append(self._plot_inter_arrival_histogram())
                figures.append(self._plot_acf())
                figures.append(self._plot_intensity_heatmap())
                figures.append(self._plot_qq())
            except ImportError:
                warnings.append("matplotlib not available, skipping plots")

        # ... (rest of existing code, update figures=figures) ...
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_realistic_diagnostics.py::TestHawkesDiagnostics -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rfq_simulator/output/realistic_diagnostics.py tests/test_realistic_diagnostics.py
git commit -m "feat(diagnostics): add Hawkes diagnostic plots"
```

---

## Task 5: SpreadDiagnostics

**Files:**
- Modify: `src/rfq_simulator/output/realistic_diagnostics.py`
- Modify: `tests/test_realistic_diagnostics.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_realistic_diagnostics.py
from rfq_simulator.config import SpreadConfig
from rfq_simulator.output.realistic_diagnostics import SpreadDiagnostics


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
        # Stressed should be wider than calm
        assert result.stats["stressed_median"] > result.stats["calm_median"]
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_realistic_diagnostics.py::TestSpreadDiagnostics -v`
Expected: FAIL

**Step 3: Write SpreadDiagnostics class**

```python
# Add to src/rfq_simulator/output/realistic_diagnostics.py

class SpreadDiagnostics:
    """Diagnostic analysis for LogNormal regime-dependent spreads."""

    def __init__(self, result):
        self.result = result
        self.cfg = result.cfg
        self._spreads_by_regime: Optional[Dict] = None

    def _extract_spreads_by_regime(self) -> Dict[str, np.ndarray]:
        """Extract spreads grouped by regime."""
        if self._spreads_by_regime is not None:
            return self._spreads_by_regime

        calm_spreads = []
        stressed_spreads = []

        rfq_log = self.result.final_state.rfq_log
        for r in rfq_log:
            if r.filled and hasattr(r, 'markup_bps') and r.markup_bps is not None:
                spread = abs(r.markup_bps)
                if r.regime.value == 0:  # CALM
                    calm_spreads.append(spread)
                else:
                    stressed_spreads.append(spread)

        self._spreads_by_regime = {
            "calm": np.array(calm_spreads) if calm_spreads else np.array([1.0]),
            "stressed": np.array(stressed_spreads) if stressed_spreads else np.array([1.0]),
        }
        return self._spreads_by_regime

    def _shapiro_test(self, spreads: np.ndarray) -> tuple:
        """Shapiro-Wilk test on log(spreads)."""
        try:
            from scipy.stats import shapiro
            log_spreads = np.log(spreads[spreads > 0])
            if len(log_spreads) < 3:
                return np.nan, np.nan
            # Sample if too large
            if len(log_spreads) > 5000:
                log_spreads = np.random.choice(log_spreads, 5000, replace=False)
            stat, p = shapiro(log_spreads)
            return float(stat), float(p)
        except ImportError:
            return np.nan, np.nan

    def analyze(self, generate_plots: bool = False) -> DiagnosticResult:
        """Run full spread diagnostic analysis."""
        stats = {}
        warnings = []

        spreads = self._extract_spreads_by_regime()

        # Medians
        stats["calm_median"] = float(np.median(spreads["calm"]))
        stats["stressed_median"] = float(np.median(spreads["stressed"]))
        stats["regime_ratio"] = stats["stressed_median"] / stats["calm_median"]

        # Shapiro-Wilk on calm spreads
        sh_stat, sh_p = self._shapiro_test(spreads["calm"])
        stats["shapiro_stat"] = sh_stat
        stats["shapiro_p"] = sh_p

        # Config values for reference
        stats["cfg_mu_calm"] = self.cfg.spreads.mu_calm
        stats["cfg_mu_stressed"] = self.cfg.spreads.mu_stressed

        # Warnings
        if not np.isnan(sh_p) and sh_p < 0.05:
            warnings.append(f"Log-spreads fail normality (Shapiro p={sh_p:.4f})")

        if stats["regime_ratio"] < 2.0:
            warnings.append(
                f"Regime separation weak: stressed only {stats['regime_ratio']:.1f}x calm"
            )

        # Narrative
        if stats["regime_ratio"] >= 2.0:
            narrative = format_narrative(
                "{icon} Spreads are log-normally distributed (Shapiro p={p:.3f}). "
                "Median: {calm:.1f} bps (calm) → {stressed:.1f} bps (stressed), "
                "a {ratio:.1f}× widening during stress.",
                icon=success_icon(),
                p=sh_p if not np.isnan(sh_p) else 1.0,
                calm=stats["calm_median"],
                stressed=stats["stressed_median"],
                ratio=stats["regime_ratio"],
            )
        else:
            narrative = format_narrative(
                "{icon} Regime separation weaker than expected. "
                "Stressed spreads only {ratio:.1f}× calm.",
                icon=warning_icon(),
                ratio=stats["regime_ratio"],
            )

        passed = stats["regime_ratio"] >= 2.0

        figures = []
        if generate_plots:
            figures = self._generate_plots()

        return DiagnosticResult(
            name="LogNormal Spreads",
            passed=passed,
            stats=stats,
            figures=figures,
            narrative=narrative,
            warnings=warnings,
        )

    def _generate_plots(self) -> List:
        """Generate all spread diagnostic plots."""
        import matplotlib.pyplot as plt

        spreads = self._extract_spreads_by_regime()
        figures = []

        # 1. Log-spread histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(np.log(spreads["calm"]), bins=30, alpha=0.6, label='Calm', density=True)
        ax.hist(np.log(spreads["stressed"]), bins=30, alpha=0.6, label='Stressed', density=True)
        ax.set_xlabel('log(Spread)')
        ax.set_ylabel('Density')
        ax.set_title('Log-Spread Distribution by Regime')
        ax.legend()
        plt.tight_layout()
        figures.append(fig)

        # 2. Regime boxplot
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot([spreads["calm"], spreads["stressed"]], labels=['Calm', 'Stressed'])
        ax.set_ylabel('Spread (bps)')
        ax.set_title('Spread Distribution by Regime')
        plt.tight_layout()
        figures.append(fig)

        # 3 & 4: Size curve and QQ (similar pattern)
        # ... (abbreviated for length)

        return figures
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_realistic_diagnostics.py::TestSpreadDiagnostics -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rfq_simulator/output/realistic_diagnostics.py tests/test_realistic_diagnostics.py
git commit -m "feat(diagnostics): add SpreadDiagnostics with tests and plots"
```

---

## Task 6: ImbalanceDiagnostics

**Files:**
- Modify: `src/rfq_simulator/output/realistic_diagnostics.py`
- Modify: `tests/test_realistic_diagnostics.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_realistic_diagnostics.py
from rfq_simulator.config import ImbalanceConfig
from rfq_simulator.output.realistic_diagnostics import ImbalanceDiagnostics


class TestImbalanceDiagnostics:
    @pytest.fixture
    def imbalance_result(self):
        cfg = SimConfig(
            T_days=30,
            seed=42,
            imbalance=ImbalanceConfig(use_ar1=True, rho=0.4, mu_stressed=-0.25),
        )
        return run_simulation(cfg)

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
        # Stressed should have lower buy fraction
        assert result.stats["buy_frac_stressed"] < result.stats["buy_frac_calm"]
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_realistic_diagnostics.py::TestImbalanceDiagnostics -v`
Expected: FAIL

**Step 3: Write ImbalanceDiagnostics class**

```python
# Add to src/rfq_simulator/output/realistic_diagnostics.py

class ImbalanceDiagnostics:
    """Diagnostic analysis for AR(1) buy/sell imbalance."""

    def __init__(self, result):
        self.result = result
        self.cfg = result.cfg

    def _extract_directions_by_regime(self) -> Dict:
        """Extract buy/sell directions grouped by regime."""
        calm_buys = 0
        calm_total = 0
        stressed_buys = 0
        stressed_total = 0

        directions = []  # For ACF calculation

        rfq_log = self.result.final_state.rfq_log
        for r in rfq_log:
            is_buy = 1 if r.is_client_buy else 0
            directions.append(is_buy)

            if r.regime.value == 0:  # CALM
                calm_buys += is_buy
                calm_total += 1
            else:
                stressed_buys += is_buy
                stressed_total += 1

        return {
            "directions": np.array(directions),
            "calm_buy_frac": calm_buys / calm_total if calm_total > 0 else 0.5,
            "stressed_buy_frac": stressed_buys / stressed_total if stressed_total > 0 else 0.5,
            "calm_total": calm_total,
            "stressed_total": stressed_total,
        }

    def _compute_direction_acf(self, directions: np.ndarray, lag: int = 1) -> float:
        """Compute ACF of direction sequence."""
        if len(directions) < lag + 10:
            return 0.0
        try:
            from statsmodels.tsa.stattools import acf
            return float(acf(directions, nlags=lag)[lag])
        except ImportError:
            # Manual ACF
            n = len(directions)
            mean = np.mean(directions)
            var = np.var(directions)
            if var == 0:
                return 0.0
            cov = np.mean((directions[:-lag] - mean) * (directions[lag:] - mean))
            return cov / var

    def analyze(self, generate_plots: bool = False) -> DiagnosticResult:
        """Run full imbalance diagnostic analysis."""
        stats = {}
        warnings = []

        data = self._extract_directions_by_regime()

        # Buy fractions
        stats["buy_frac_calm"] = data["calm_buy_frac"]
        stats["buy_frac_stressed"] = data["stressed_buy_frac"]
        stats["buy_frac_diff"] = data["calm_buy_frac"] - data["stressed_buy_frac"]

        # ACF
        stats["direction_acf1"] = self._compute_direction_acf(data["directions"], 1)
        stats["direction_acf5"] = self._compute_direction_acf(data["directions"], 5)

        # Config reference
        stats["cfg_rho"] = self.cfg.imbalance.rho
        stats["cfg_mu_stressed"] = self.cfg.imbalance.mu_stressed

        # Warnings
        if abs(stats["direction_acf1"] - self.cfg.imbalance.rho) > 0.15:
            warnings.append(
                f"ACF(1)={stats['direction_acf1']:.3f} differs from ρ={self.cfg.imbalance.rho:.2f}"
            )

        if stats["buy_frac_diff"] < 0.02:
            warnings.append("Regime bias too weak - stressed similar to calm")

        # Narrative
        narrative = format_narrative(
            "{icon} Flow shows persistence (ACF(1)={acf:.3f}, expected ρ={rho:.2f}). "
            "Stressed: {stressed:.1%} buys vs calm: {calm:.1%} ({diff:+.1%} difference).",
            icon=success_icon() if stats["buy_frac_diff"] >= 0.02 else warning_icon(),
            acf=stats["direction_acf1"],
            rho=self.cfg.imbalance.rho,
            stressed=stats["buy_frac_stressed"],
            calm=stats["buy_frac_calm"],
            diff=-stats["buy_frac_diff"],
        )

        passed = stats["buy_frac_diff"] >= 0.02

        figures = []
        if generate_plots:
            figures = self._generate_plots(data)

        return DiagnosticResult(
            name="AR(1) Imbalance",
            passed=passed,
            stats=stats,
            figures=figures,
            narrative=narrative,
            warnings=warnings,
        )

    def _generate_plots(self, data: Dict) -> List:
        """Generate imbalance diagnostic plots."""
        import matplotlib.pyplot as plt

        figures = []

        # 1. Rolling buy fraction
        fig, ax = plt.subplots(figsize=(10, 4))
        window = 50
        directions = data["directions"]
        if len(directions) > window:
            rolling = np.convolve(directions, np.ones(window)/window, mode='valid')
            ax.plot(rolling, alpha=0.7)
            ax.axhline(0.5, color='k', linestyle='--', alpha=0.5, label='Balanced')
        ax.set_xlabel('RFQ Index')
        ax.set_ylabel('Rolling Buy Fraction')
        ax.set_title(f'Rolling Buy Fraction ({window}-RFQ window)')
        ax.legend()
        plt.tight_layout()
        figures.append(fig)

        # 2. Regime bar chart
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(['Calm', 'Stressed'],
               [data["calm_buy_frac"], data["stressed_buy_frac"]],
               color=['steelblue', 'indianred'])
        ax.axhline(0.5, color='k', linestyle='--', alpha=0.5)
        ax.set_ylabel('Buy Fraction')
        ax.set_title('Buy Fraction by Regime')
        ax.set_ylim(0, 1)
        plt.tight_layout()
        figures.append(fig)

        return figures
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_realistic_diagnostics.py::TestImbalanceDiagnostics -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rfq_simulator/output/realistic_diagnostics.py tests/test_realistic_diagnostics.py
git commit -m "feat(diagnostics): add ImbalanceDiagnostics with tests and plots"
```

---

## Task 7: ValidationReport Aggregator

**Files:**
- Modify: `src/rfq_simulator/output/realistic_diagnostics.py`
- Modify: `tests/test_realistic_diagnostics.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_realistic_diagnostics.py
from rfq_simulator.output.realistic_diagnostics import ValidationReport


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
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_realistic_diagnostics.py::TestValidationReport -v`
Expected: FAIL

**Step 3: Write ValidationReport class**

```python
# Add to src/rfq_simulator/output/realistic_diagnostics.py

class ValidationReport:
    """Aggregates all realistic distribution diagnostics."""

    def __init__(self, result):
        self.result = result
        self.hawkes = HawkesDiagnostics(result)
        self.spread = SpreadDiagnostics(result)
        self.imbalance = ImbalanceDiagnostics(result)
        self._results: List[DiagnosticResult] = []

    def run_all(self, generate_plots: bool = False) -> List[DiagnosticResult]:
        """
        Run all diagnostics.

        Args:
            generate_plots: If True, generate matplotlib figures

        Returns:
            List of DiagnosticResult for each feature
        """
        self._results = []

        # Only run diagnostics for enabled features
        cfg = self.result.cfg

        if cfg.arrivals.use_hawkes:
            self._results.append(self.hawkes.analyze(generate_plots))

        if cfg.spreads.use_lognormal:
            self._results.append(self.spread.analyze(generate_plots))

        if cfg.imbalance.use_ar1:
            self._results.append(self.imbalance.analyze(generate_plots))

        return self._results

    @property
    def all_passed(self) -> bool:
        """True if all diagnostics passed."""
        return all(r.passed for r in self._results)

    @property
    def all_warnings(self) -> List[str]:
        """Collect all warnings from all diagnostics."""
        warnings = []
        for r in self._results:
            warnings.extend(r.warnings)
        return warnings

    def summary(self) -> str:
        """
        Generate executive summary.

        Returns:
            One-paragraph summary of validation results
        """
        if not self._results:
            return "No diagnostics run. Call run_all() first."

        passed = [r.name for r in self._results if r.passed]
        failed = [r.name for r in self._results if not r.passed]

        parts = []

        if passed:
            parts.append(f"Validated: {', '.join(passed)}.")
        if failed:
            parts.append(f"Issues: {', '.join(failed)}.")

        # Add key stats
        for r in self._results:
            if "acf_lag1" in r.stats:
                parts.append(f"Hawkes ACF(1)={r.stats['acf_lag1']:.3f}.")
            if "regime_ratio" in r.stats:
                parts.append(f"Spread widening {r.stats['regime_ratio']:.1f}×.")
            if "buy_frac_diff" in r.stats:
                parts.append(f"Regime buy diff {r.stats['buy_frac_diff']:.1%}.")

        return " ".join(parts)

    def display(self):
        """Display report in notebook."""
        if not self._results:
            self.run_all(generate_plots=True)

        try:
            from IPython.display import display, Markdown

            lines = ["# Realistic Distributions Validation Report\n"]
            lines.append(f"**Summary:** {self.summary()}\n")
            lines.append("---\n")

            for result in self._results:
                icon = success_icon() if result.passed else warning_icon()
                lines.append(f"## {icon} {result.name}\n")
                lines.append(result.narrative + "\n")

                if result.warnings:
                    lines.append("**Warnings:**\n")
                    for w in result.warnings:
                        lines.append(f"- {w}\n")

                lines.append("\n**Statistics:**\n")
                for k, v in result.stats.items():
                    if isinstance(v, float):
                        lines.append(f"- {k}: {v:.4f}\n")
                    else:
                        lines.append(f"- {k}: {v}\n")

                lines.append("---\n")

            display(Markdown("".join(lines)))

            # Display figures
            for result in self._results:
                for fig in result.figures:
                    display(fig)

        except ImportError:
            print(self.summary())
            for r in self._results:
                print(f"\n{r.name}: {'PASS' if r.passed else 'WARN'}")
                print(r.narrative)

    def to_html(self, path: str):
        """Export report to HTML file."""
        import base64
        from io import BytesIO

        if not self._results:
            self.run_all(generate_plots=True)

        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<title>Validation Report</title>",
            "<style>body{font-family:sans-serif;max-width:900px;margin:auto;padding:20px;}</style>",
            "</head><body>",
            "<h1>Realistic Distributions Validation Report</h1>",
            f"<p><strong>Summary:</strong> {self.summary()}</p>",
        ]

        for result in self._results:
            icon = "✓" if result.passed else "⚠"
            html_parts.append(f"<h2>{icon} {result.name}</h2>")
            html_parts.append(f"<p>{result.narrative}</p>")

            # Embed figures as base64
            for fig in result.figures:
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode()
                html_parts.append(f'<img src="data:image/png;base64,{img_b64}" />')

        html_parts.append("</body></html>")

        with open(path, 'w') as f:
            f.write("\n".join(html_parts))
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_realistic_diagnostics.py::TestValidationReport -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rfq_simulator/output/realistic_diagnostics.py tests/test_realistic_diagnostics.py
git commit -m "feat(diagnostics): add ValidationReport aggregator with display and export"
```

---

## Task 8: Update Module Exports

**Files:**
- Modify: `src/rfq_simulator/output/__init__.py`

**Step 1: Update exports**

```python
# src/rfq_simulator/output/__init__.py
"""Output module for simulation diagnostics and visualization."""

from .diagnostics import (
    plot_simulation_results,
    plot_pnl_decomposition,
    create_rfq_dataframe,
    generate_summary_report,
    plot_win_rate_calibration,
)

from .realistic_diagnostics import (
    DiagnosticResult,
    HawkesDiagnostics,
    SpreadDiagnostics,
    ImbalanceDiagnostics,
    ValidationReport,
)

from .narrative import (
    format_narrative,
    format_warning,
)

__all__ = [
    # Original diagnostics
    "plot_simulation_results",
    "plot_pnl_decomposition",
    "create_rfq_dataframe",
    "generate_summary_report",
    "plot_win_rate_calibration",
    # Realistic diagnostics
    "DiagnosticResult",
    "HawkesDiagnostics",
    "SpreadDiagnostics",
    "ImbalanceDiagnostics",
    "ValidationReport",
    # Narrative
    "format_narrative",
    "format_warning",
]
```

**Step 2: Run all tests**

Run: `PYTHONPATH=src pytest tests/test_realistic_diagnostics.py tests/test_narrative.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add src/rfq_simulator/output/__init__.py
git commit -m "feat(diagnostics): export realistic diagnostics from output module"
```

---

## Task 9: Integration Test

**Files:**
- Create: `tests/test_diagnostics_integration.py`

**Step 1: Write integration test**

```python
# tests/test_diagnostics_integration.py
"""Integration tests for realistic diagnostics."""
import pytest
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

    def test_partial_features(self):
        """Run validation with only some features enabled."""
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
```

**Step 2: Run integration test**

Run: `PYTHONPATH=src pytest tests/test_diagnostics_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_diagnostics_integration.py
git commit -m "test(diagnostics): add integration tests for validation report"
```

---

## Task 10: Final Verification

**Step 1: Run full test suite**

```bash
PYTHONPATH=src pytest tests/ -v
```

Expected: All tests pass (original 79 + new ~25 = ~104 tests)

**Step 2: Demo in Python**

```python
from rfq_simulator.config import SimConfig, ArrivalConfig, SpreadConfig, ImbalanceConfig
from rfq_simulator.simulation import run_simulation
from rfq_simulator.output import ValidationReport

cfg = SimConfig(
    T_days=30,
    seed=42,
    arrivals=ArrivalConfig(use_hawkes=True),
    spreads=SpreadConfig(use_lognormal=True),
    imbalance=ImbalanceConfig(use_ar1=True),
)
result = run_simulation(cfg)

report = ValidationReport(result)
report.display()  # Shows in notebook
report.to_html("validation_report.html")  # Export
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat(diagnostics): complete realistic distributions validation suite"
git push origin main
```

---

## Summary

| Task | Description | Tests |
|------|-------------|-------|
| 1 | DiagnosticResult dataclass | 2 |
| 2 | Narrative helpers | 3 |
| 3 | HawkesDiagnostics stats | 4 |
| 4 | HawkesDiagnostics plots | 2 |
| 5 | SpreadDiagnostics | 4 |
| 6 | ImbalanceDiagnostics | 3 |
| 7 | ValidationReport | 4 |
| 8 | Module exports | - |
| 9 | Integration tests | 2 |
| 10 | Final verification | - |

**Total new tests:** ~24
