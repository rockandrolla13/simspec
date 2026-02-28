"""
Output module: Diagnostics, plots, and result formatting.

- diagnostics: Visualization and summary statistics
"""

from .diagnostics import (
    plot_simulation_results,
    plot_pnl_decomposition,
    create_rfq_dataframe,
    generate_summary_report,
)

__all__ = [
    "plot_simulation_results",
    "plot_pnl_decomposition",
    "create_rfq_dataframe",
    "generate_summary_report",
]
