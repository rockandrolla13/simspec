"""
Diagnostics: Visualization and summary statistics.

Provides:
- Time series plots (price, inventory, P&L)
- P&L decomposition charts
- RFQ log to DataFrame conversion
- Summary reports
"""

from typing import Optional, List, Dict, Any

import numpy as np

# Note: matplotlib and pandas are optional dependencies
# Functions will work but plots won't render if not installed


def plot_simulation_results(
    result,  # SimulationResult
    figsize: tuple = (14, 10),
    show: bool = True,
):
    """
    Plot comprehensive simulation results.

    Creates a 2x2 grid:
    - Price path with fills
    - Inventory vs target
    - Cumulative P&L components
    - Fill rate and spread earned

    Args:
        result: SimulationResult from run_simulation()
        figsize: Figure size
        show: Whether to display the plot

    Returns:
        matplotlib figure if available
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return None

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Extract data
    prices = result.prices
    time_steps = np.arange(len(prices)) * result.cfg.dt_minutes

    # Get RFQ log data
    rfq_log = result.final_state.rfq_log
    fill_times = [r.time for r in rfq_log if r.filled]
    fill_prices = [r.p_true for r in rfq_log if r.filled]

    # 1. Price path with fills
    ax1 = axes[0, 0]
    ax1.plot(time_steps, prices, 'b-', alpha=0.7, label='Price')
    if fill_times:
        ax1.scatter(fill_times, fill_prices, c='red', s=20, alpha=0.5, label='Fills')
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Price Path with Fills')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Inventory vs target
    ax2 = axes[0, 1]
    rfq_times = [r.time for r in rfq_log]
    q_after = [r.q_after for r in rfq_log]
    q_targets = [r.q_target for r in rfq_log]

    if rfq_times:
        ax2.plot(rfq_times, q_after, 'b-', label='Inventory', alpha=0.8)
        ax2.plot(rfq_times, q_targets, 'r--', label='Target', alpha=0.6)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Position (lots)')
    ax2.set_title('Inventory vs Target')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Cumulative P&L
    ax3 = axes[1, 0]
    ts = result.pnl_tracker.get_time_series()
    if len(ts['time']) > 0:
        ax3.plot(ts['time'], ts['alpha_pnl'], label='Alpha P&L', alpha=0.8)
        ax3.plot(ts['time'], ts['spread_pnl'], label='Spread P&L', alpha=0.8)
        ax3.plot(ts['time'], ts['total_pnl'], 'k-', label='Total P&L', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('P&L ($)')
    ax3.set_title('Cumulative P&L Decomposition')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Fill statistics
    ax4 = axes[1, 1]
    state = result.final_state

    # Bar chart of key metrics
    metrics = {
        'Fill Rate': state.get_fill_rate() * 100,
        'Quote Rate': state.get_quote_rate() * 100,
        'Avg Spread (bps)': state.get_average_spread(),
    }
    bars = ax4.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange'])
    ax4.set_ylabel('Value')
    ax4.set_title('Trading Statistics')

    # Add value labels on bars
    for bar, val in zip(bars, metrics.values()):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom')

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_pnl_decomposition(
    result,  # SimulationResult
    figsize: tuple = (10, 6),
    show: bool = True,
):
    """
    Plot P&L decomposition as a waterfall chart.

    Args:
        result: SimulationResult
        figsize: Figure size
        show: Whether to display

    Returns:
        matplotlib figure if available
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return None

    pnl = result.pnl

    # Components
    components = ['Alpha', 'Spread', 'Carry', 'Hedge', 'Aggress Cost', 'Total']
    values = [
        pnl.alpha_pnl,
        pnl.spread_pnl,
        pnl.carry_pnl,
        pnl.hedge_pnl,
        -pnl.aggress_cost,  # Negative because it's a cost
        pnl.total_pnl,
    ]

    fig, ax = plt.subplots(figsize=figsize)

    # Colors: green for positive, red for negative
    colors = ['green' if v >= 0 else 'red' for v in values]
    colors[-1] = 'blue'  # Total in blue

    bars = ax.bar(components, values, color=colors, alpha=0.7)

    # Add value labels
    for bar, val in zip(bars, values):
        y_pos = bar.get_height() + (50 if val >= 0 else -100)
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'${val:,.0f}', ha='center', va='bottom' if val >= 0 else 'top')

    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax.set_ylabel('P&L ($)')
    ax.set_title('P&L Decomposition')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def create_rfq_dataframe(result) -> Optional[Any]:
    """
    Convert RFQ log to pandas DataFrame for analysis.

    Args:
        result: SimulationResult

    Returns:
        pandas DataFrame if pandas installed, else None
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed, returning None")
        return None

    rfq_log = result.final_state.rfq_log

    if not rfq_log:
        return pd.DataFrame()

    data = []
    for r in rfq_log:
        data.append({
            'time': r.time,
            'day': int(r.time / result.cfg.minutes_per_day),
            'is_client_buy': r.is_client_buy,
            'size': r.size,
            'n_dealers': r.n_dealers,
            'toxicity': r.toxicity,
            'p_true': r.p_true,
            'theo': r.theo,
            'theo_error': r.theo - r.p_true,
            'markup_bps': r.markup_bps,
            'win_prob_est': r.win_prob_est,
            'declined': r.declined,
            'filled': r.filled,
            'spread_pnl': r.spread_pnl,
            'adverse_move': r.adverse_move,
            'q_before': r.q_before,
            'q_after': r.q_after,
            'q_target': r.q_target,
            'alpha_remaining': r.alpha_remaining,
            'regime': r.regime.name,
        })

    return pd.DataFrame(data)


def generate_summary_report(
    result,  # SimulationResult
    baseline_result=None,  # Optional BaselineResult
) -> str:
    """
    Generate a text summary report.

    Args:
        result: SimulationResult from LP strategy
        baseline_result: Optional BaselineResult for comparison

    Returns:
        Formatted text report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("RFQ SIMULATOR SUMMARY REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Configuration summary
    cfg = result.cfg
    lines.append("CONFIGURATION:")
    lines.append(f"  Simulation days: {cfg.T_days}")
    lines.append(f"  Daily vol (bps): {cfg.sigma_daily_bps}")
    lines.append(f"  IC: {cfg.IC}")
    lines.append(f"  RFQs per day: {cfg.rfq_rate_per_day}")
    lines.append(f"  Position limit: {cfg.q_max} lots")
    lines.append("")

    # P&L summary
    pnl = result.pnl
    lines.append("P&L DECOMPOSITION:")
    lines.append(f"  Alpha P&L:    ${pnl.alpha_pnl:>12,.2f}")
    lines.append(f"  Spread P&L:   ${pnl.spread_pnl:>12,.2f}")
    lines.append(f"  Carry P&L:    ${pnl.carry_pnl:>12,.2f}")
    lines.append(f"  Hedge P&L:    ${pnl.hedge_pnl:>12,.2f}")
    lines.append(f"  Aggress Cost: ${pnl.aggress_cost:>12,.2f}")
    lines.append(f"  ---------------------")
    lines.append(f"  TOTAL P&L:    ${pnl.total_pnl:>12,.2f}")
    lines.append("")

    # Trading statistics
    state = result.final_state
    lines.append("TRADING STATISTICS:")
    lines.append(f"  RFQs seen: {state.n_rfqs_seen}")
    lines.append(f"  RFQs quoted: {state.n_rfqs_quoted}")
    lines.append(f"  RFQs filled: {state.n_rfqs_filled}")
    lines.append(f"  Quote rate: {state.get_quote_rate():.1%}")
    lines.append(f"  Fill rate: {state.get_fill_rate():.1%}")
    lines.append(f"  Avg spread (bps): {state.get_average_spread():.2f}")
    lines.append(f"  Total volume: {state.total_volume:.1f} lots")
    lines.append(f"  Final inventory: {state.q:.1f} lots")
    lines.append("")

    # Risk metrics
    lines.append("RISK METRICS:")
    lines.append(f"  Sharpe ratio: {result.sharpe_ratio:.2f}")
    lines.append(f"  Max drawdown: ${result.max_drawdown:,.2f}")
    lines.append("")

    # Comparison with baseline if provided
    if baseline_result is not None:
        from ..simulation.baseline import compare_strategies
        comparison = compare_strategies(result, baseline_result)

        lines.append("BASELINE COMPARISON:")
        lines.append(f"  Baseline P&L: ${comparison['baseline_total_pnl']:>12,.2f}")
        lines.append(f"  LP P&L:       ${comparison['lp_total_pnl']:>12,.2f}")
        lines.append(f"  Difference:   ${comparison['pnl_difference']:>12,.2f}")
        lines.append("")
        lines.append(f"  Alpha sacrifice: ${comparison['alpha_sacrifice']:>12,.2f}")
        lines.append(f"  Spread earned:   ${comparison['lp_spread_pnl']:>12,.2f}")
        lines.append(f"  Net benefit:     ${comparison['spread_minus_alpha_loss']:>12,.2f}")
        lines.append("")

        if comparison['spread_minus_alpha_loss'] > 0:
            lines.append("  >> LP STRATEGY OUTPERFORMS BASELINE <<")
        else:
            lines.append("  >> BASELINE OUTPERFORMS LP STRATEGY <<")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def plot_win_rate_calibration(
    result,  # SimulationResult
    n_bins: int = 20,
    figsize: tuple = (10, 5),
    show: bool = True,
):
    """
    Plot calibration of estimated vs realized win rates.

    Args:
        result: SimulationResult
        n_bins: Number of probability bins
        figsize: Figure size
        show: Whether to display

    Returns:
        matplotlib figure if available
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    # Get quoted RFQs
    rfq_log = result.final_state.rfq_log
    quoted = [(r.win_prob_est, r.filled) for r in rfq_log if not r.declined]

    if len(quoted) < 50:
        print("Not enough data for calibration plot")
        return None

    probs = np.array([q[0] for q in quoted])
    filled = np.array([q[1] for q in quoted], dtype=float)

    # Bin by estimated probability
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    estimated = []
    realized = []
    counts = []

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() > 0:
            estimated.append(probs[mask].mean())
            realized.append(filled[mask].mean())
            counts.append(mask.sum())
        else:
            estimated.append(bin_centers[i])
            realized.append(np.nan)
            counts.append(0)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot calibration
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.scatter(estimated, realized, s=[c * 2 for c in counts], alpha=0.6, label='Bins')

    ax.set_xlabel('Estimated Win Probability')
    ax.set_ylabel('Realized Win Rate')
    ax.set_title('Win Rate Calibration')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if show:
        plt.show()

    return fig
