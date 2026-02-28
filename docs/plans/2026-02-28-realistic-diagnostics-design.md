# Realistic Distributions Diagnostics Design

**Date**: 2026-02-28
**Status**: Approved
**Goal**: Provide comprehensive visualization and auto-generated commentary for validating realistic market distribution features (Hawkes arrivals, LogNormal spreads, AR(1) imbalance).

---

## 1. Overview

### Audience
Quant researchers requiring statistical rigor, distribution fits, hypothesis testing, and calibration validation.

### Scope
Validate all three realistic distribution features equally:
- Hawkes self-exciting arrivals
- LogNormal regime-dependent spreads
- AR(1) buy/sell imbalance

### Commentary Types
1. **Statistical validation** - Goodness-of-fit tests, p-values, confidence intervals
2. **Plain English insights** - Narrative explanations of observed behavior
3. **Anomaly detection** - Warnings when metrics deviate from expectations

### Output Formats
- Primary: Jupyter notebook cells (inline plots + markdown)
- Export: HTML (self-contained, shareable) and PDF (publication-quality)

---

## 2. Architecture

### File Structure

```
src/rfq_simulator/output/
├── diagnostics.py                # Existing (unchanged)
├── realistic_diagnostics.py      # NEW: Main module
│   ├── DiagnosticResult          # Dataclass: stats + narrative + warnings
│   ├── HawkesDiagnostics         # Arrival clustering validation
│   ├── SpreadDiagnostics         # Distribution fitting
│   ├── ImbalanceDiagnostics      # Autocorrelation analysis
│   └── ValidationReport          # Aggregates all three
└── narrative.py                  # NEW: Text generation helpers
```

### Data Flow

```
SimulationResult
    → HawkesDiagnostics.analyze() → DiagnosticResult
    → SpreadDiagnostics.analyze() → DiagnosticResult
    → ImbalanceDiagnostics.analyze() → DiagnosticResult
    → ValidationReport.generate() → Combined plots + markdown
```

### Core Dataclass

```python
@dataclass
class DiagnosticResult:
    name: str                    # "Hawkes Arrivals"
    passed: bool                 # Overall validation passed?
    stats: Dict[str, float]      # {"acf_lag1": 0.12, "p_value": 0.03}
    figures: List[Figure]        # Matplotlib figures
    narrative: str               # "Inter-arrival times show clustering..."
    warnings: List[str]          # ["ACF below expected threshold"]
```

---

## 3. Hawkes Diagnostics

### Plots

| # | Plot | Purpose |
|---|------|---------|
| 1 | Inter-arrival histogram | Show distribution with exponential fit overlay |
| 2 | ACF of inter-arrivals | Visualize clustering (lags 1-20) |
| 3 | Intensity heatmap | Hour-of-day clustering bursts |
| 4 | QQ-plot | Rescaled inter-arrivals vs Exp(1) |

### Statistical Tests

| Test | Purpose | Pass Criterion |
|------|---------|----------------|
| Ljung-Box | Serial correlation in arrivals | p < 0.05 (reject IID) |
| ACF(1) | Clustering strength | > 0.05 |
| Branching ratio | Stationarity check | α/β < 1 |

### Narrative Templates

**Success**:
```
"RFQ arrivals show significant clustering (ACF(1)={acf:.3f}, p={p:.4f}).
Bursts of {burst_size:.1f} RFQs follow large trades on average."
```

**Warning**:
```
"Warning: Clustering weaker than expected (ACF(1)={acf:.3f} < 0.05).
Consider increasing hawkes_alpha or decreasing hawkes_beta."
```

---

## 4. Spread Diagnostics

### Plots

| # | Plot | Purpose |
|---|------|---------|
| 1 | Log-spread histogram | Fitted normal overlay per regime |
| 2 | Regime comparison boxplot | Calm vs stressed side-by-side |
| 3 | Size adjustment curve | Spread vs size with γ·log(1+size) fit |
| 4 | QQ-plot | log(spreads) vs Normal |

### Statistical Tests

| Test | Purpose | Pass Criterion |
|------|---------|----------------|
| Shapiro-Wilk | Log-normality of spreads | p > 0.05 on log(spreads) |
| Welch t-test | Regime separation | stressed_mean > 2× calm_mean |
| Regression | Size coefficient | γ_fitted ≈ γ_config ± 20% |

### Narrative Templates

**Success**:
```
"Spreads are log-normally distributed (Shapiro p={p:.3f}).
Median spread: {calm_median:.1f} bps (calm) → {stressed_median:.1f} bps (stressed),
a {ratio:.1f}× widening during stress."

"Size impact coefficient γ={gamma_fit:.3f} matches config ({gamma_cfg:.3f}).
Large trades (100 lots) pay {large_spread:.1f} bps vs {small_spread:.1f} bps for small."
```

**Warning**:
```
"Warning: Log-spreads fail normality test (p={p:.4f}).
Distribution may be bimodal - consider mixture model."
```

---

## 5. Imbalance Diagnostics

### Plots

| # | Plot | Purpose |
|---|------|---------|
| 1 | Imbalance time series | With regime shading (calm=blue, stressed=red) |
| 2 | ACF plot | Lags 1-20 with confidence bands |
| 3 | Buy fraction by regime | Bar chart calm vs stressed |
| 4 | Rolling buy fraction | 50-RFQ rolling window |

### Statistical Tests

| Test | Purpose | Pass Criterion |
|------|---------|----------------|
| ACF(1) | AR(1) persistence | ≈ ρ_config ± 0.1 |
| Regime t-test | Mean shift in stress | p < 0.05 |
| Stationarity (ADF) | No drift | p < 0.05 (stationary) |

### Narrative Templates

**Success**:
```
"Buy/sell flow shows persistence (ACF(1)={acf:.3f}, expected ρ={rho:.2f}).
Autocorrelation indicates informed flow clustering."

"Stressed periods show sell bias: {stressed_buy:.1f}% buys vs {calm_buy:.1f}% in calm.
This {diff:.1f}pp difference is statistically significant (p={p:.4f})."
```

**Warning**:
```
"Warning: ACF(1)={acf:.3f} deviates from configured ρ={rho:.2f}.
Check imbalance_sigma - high noise can wash out persistence."
```

---

## 6. Unified Report

### ValidationReport Class

```python
class ValidationReport:
    def __init__(self, result: SimulationResult):
        self.hawkes = HawkesDiagnostics(result)
        self.spread = SpreadDiagnostics(result)
        self.imbalance = ImbalanceDiagnostics(result)

    def run_all(self) -> List[DiagnosticResult]:
        """Run all diagnostics, return results."""

    def to_notebook_cells(self) -> str:
        """Generate markdown + code cells for Jupyter."""

    def to_html(self, path: str) -> None:
        """Export self-contained HTML with embedded plots."""

    def to_pdf(self, path: str) -> None:
        """Export publication-quality PDF."""

    def summary(self) -> str:
        """One-paragraph executive summary."""
```

### Report Structure

```
╔════════════════════════════════════════════════════════════╗
║           REALISTIC DISTRIBUTIONS VALIDATION               ║
╠════════════════════════════════════════════════════════════╣
║ Executive Summary                                          ║
║   "All three features validated. Hawkes clustering         ║
║    detected (ACF=0.12), spreads 3.2× wider in stress,      ║
║    sell bias of -4.1pp during stressed periods."           ║
╠════════════════════════════════════════════════════════════╣
║ 1. HAWKES ARRIVALS           ✓ Passed                      ║
║    [4 plots] [stats table] [narrative]                     ║
╠════════════════════════════════════════════════════════════╣
║ 2. LOGNORMAL SPREADS         ✓ Passed                      ║
║    [4 plots] [stats table] [narrative]                     ║
╠════════════════════════════════════════════════════════════╣
║ 3. AR(1) IMBALANCE           ⚠ Warning                     ║
║    [4 plots] [stats table] [narrative]                     ║
║    Warning: ACF weaker than expected                       ║
╠════════════════════════════════════════════════════════════╣
║ Appendix: Raw Statistics                                   ║
╚════════════════════════════════════════════════════════════╝
```

### Export Options

| Format | Library | Use Case |
|--------|---------|----------|
| Notebook | IPython.display | Interactive exploration |
| HTML | Jinja2 + base64 plots | Shareable single file |
| PDF | matplotlib savefig | Publication figures |

---

## 7. Dependencies

### Required (already in project)
- numpy
- matplotlib

### Optional (for enhanced features)
- scipy.stats (statistical tests)
- statsmodels (ACF, Ljung-Box, ADF)
- jinja2 (HTML export)

---

## 8. API Usage

### Quick Validation

```python
from rfq_simulator.output.realistic_diagnostics import ValidationReport

result = run_simulation(cfg)
report = ValidationReport(result)
report.run_all()  # Displays in notebook
```

### Individual Diagnostics

```python
from rfq_simulator.output.realistic_diagnostics import HawkesDiagnostics

hawkes = HawkesDiagnostics(result)
diag = hawkes.analyze()
print(diag.narrative)
print(diag.warnings)
```

### Export

```python
report.to_html("validation_report.html")
report.to_pdf("validation_figures.pdf")
```

---

## 9. Success Criteria

- [ ] All 12 plots render correctly (4 per feature)
- [ ] Statistical tests compute without errors
- [ ] Narrative templates populate with real values
- [ ] Warnings trigger appropriately on edge cases
- [ ] HTML export produces self-contained file
- [ ] PDF export produces publication-quality figures
- [ ] Existing tests continue to pass
