"""Output module for simulation diagnostics and visualization."""

# Import existing diagnostics (keep whatever is there)
from .diagnostics import (
    plot_simulation_results,
    plot_pnl_decomposition,
    create_rfq_dataframe,
    generate_summary_report,
    plot_win_rate_calibration,
)

# Import new realistic diagnostics
from .realistic_diagnostics import (
    DiagnosticResult,
    HawkesDiagnostics,
    SpreadDiagnostics,
    ImbalanceDiagnostics,
    ValidationReport,
)

# Import narrative helpers
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
