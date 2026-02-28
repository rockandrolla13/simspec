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
