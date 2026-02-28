"""
Diagnostics for validating realistic market distributions.

Provides statistical validation, plots, and auto-generated narrative
for Hawkes arrivals, LogNormal spreads, and AR(1) imbalance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import numpy as np

from .narrative import format_narrative, format_warning, success_icon, warning_icon


@dataclass
class DiagnosticResult:
    """Result from a diagnostic analysis."""

    name: str                           # "Hawkes Arrivals"
    passed: bool                        # Overall validation passed?
    stats: Dict[str, float]             # {"acf_lag1": 0.12, "p_value": 0.03}
    figures: List[Any]                  # Matplotlib figures
    narrative: str                      # "Inter-arrival times show clustering..."
    warnings: List[str] = field(default_factory=list)  # ["ACF below threshold"]


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
        except Exception:
            # Manual ACF computation (fallback if statsmodels unavailable or broken)
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
        except Exception:
            # Skip test if statsmodels not available or broken
            return np.nan, np.nan

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
        ax.set_title('RFQ Arrival Intensity Heatmap')
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
            generate_plots: If True, generate matplotlib figures (for Task 4)

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

        figures = []
        if generate_plots:
            try:
                import matplotlib.pyplot as plt
                plt.switch_backend('Agg')  # Non-interactive backend for testing
                figures.append(self._plot_inter_arrival_histogram())
                figures.append(self._plot_acf())
                figures.append(self._plot_intensity_heatmap())
                figures.append(self._plot_qq())
            except ImportError as e:
                warnings.append(f"matplotlib not available for plots: {e}")

        return DiagnosticResult(
            name="Hawkes Arrivals",
            passed=passed,
            stats=stats,
            figures=figures,
            narrative=narrative,
            warnings=warnings,
        )
