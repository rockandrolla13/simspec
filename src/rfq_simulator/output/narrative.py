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
