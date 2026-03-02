"""
Batch Monte Carlo runner for scenario analysis.

Provides:
- Multi-path simulation with parallel execution
- Scenario sweeps over parameter grids
- Statistical analysis of results
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from ..config import SimConfig
from .event_loop import run_simulation, SimulationResult
from .baseline import run_baseline, BaselineResult, compare_strategies


@dataclass
class BatchResult:
    """
    Results from a batch of Monte Carlo simulations.

    Attributes:
        results: List of individual SimulationResult objects
        baseline_results: Optional list of BaselineResult objects
        cfg: Base configuration used
        seeds: Random seeds used for each path
    """

    results: List[SimulationResult]
    baseline_results: Optional[List[BaselineResult]]
    cfg: SimConfig
    seeds: List[int]

    @property
    def n_paths(self) -> int:
        return len(self.results)

    def get_pnl_array(self) -> np.ndarray:
        """Get array of total P&L values."""
        return np.array([r.total_pnl for r in self.results])

    def get_alpha_pnl_array(self) -> np.ndarray:
        """Get array of alpha P&L values."""
        return np.array([r.pnl.alpha_pnl for r in self.results])

    def get_spread_pnl_array(self) -> np.ndarray:
        """Get array of spread P&L values."""
        return np.array([r.pnl.spread_pnl for r in self.results])

    def statistics(self) -> Dict[str, float]:
        """Compute summary statistics across paths."""
        pnls = self.get_pnl_array()
        alpha_pnls = self.get_alpha_pnl_array()
        spread_pnls = self.get_spread_pnl_array()

        fill_rates = [r.final_state.get_fill_rate() for r in self.results]
        spreads = [r.final_state.get_average_spread() for r in self.results]

        stats = {
            # P&L
            'pnl_mean': np.mean(pnls),
            'pnl_std': np.std(pnls),
            'pnl_median': np.median(pnls),
            'pnl_5pct': np.percentile(pnls, 5),
            'pnl_95pct': np.percentile(pnls, 95),

            # Components
            'alpha_pnl_mean': np.mean(alpha_pnls),
            'spread_pnl_mean': np.mean(spread_pnls),

            # Trading
            'fill_rate_mean': np.mean(fill_rates),
            'avg_spread_mean': np.mean(spreads),

            # Risk-adjusted
            'sharpe_of_pnls': np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0,
        }

        # Compare to baseline if available
        if self.baseline_results:
            bl_pnls = np.array([r.total_pnl for r in self.baseline_results])
            stats['baseline_pnl_mean'] = np.mean(bl_pnls)
            stats['pnl_vs_baseline'] = np.mean(pnls - bl_pnls)
            stats['outperformance_rate'] = np.mean(pnls > bl_pnls)

        return stats


def _run_single_path(
    cfg: SimConfig, seed: int, run_baseline_flag: bool
) -> tuple:
    """Run one MC path (strategy + optional baseline). Module-level for pickling."""
    result = run_simulation(cfg, seed=seed, verbose=False)
    bl_result = None
    if run_baseline_flag:
        bl_result = run_baseline(
            prices=result.prices,
            regime_path=result.regime_path,
            cfg=cfg,
            seed=seed,
            verbose=False,
        )
    return result, bl_result


def run_batch(
    cfg: SimConfig,
    n_paths: int = None,
    run_baseline_flag: bool = True,
    parallel: bool = False,
    max_workers: int = 4,
    verbose: bool = True,
) -> BatchResult:
    """
    Run a batch of Monte Carlo simulations.

    Args:
        cfg: Base SimConfig (uses cfg.n_mc_paths if n_paths not specified)
        n_paths: Number of paths (overrides cfg.n_mc_paths)
        run_baseline_flag: Also run baseline for comparison
        parallel: Use parallel execution
        max_workers: Number of parallel workers
        verbose: Print progress

    Returns:
        BatchResult with all paths
    """
    n_paths = n_paths if n_paths is not None else cfg.n_mc_paths

    # Generate seeds
    rng = np.random.default_rng(cfg.seed)
    seeds = [int(rng.integers(2**31)) for _ in range(n_paths)]

    results = []
    baseline_results = [] if run_baseline_flag else None

    if parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_single_path, cfg, seed, run_baseline_flag
                ): i
                for i, seed in enumerate(seeds)
            }
            for future in as_completed(futures):
                idx = futures[future]
                result, bl_result = future.result()
                results.append((idx, result))
                if run_baseline_flag:
                    baseline_results.append((idx, bl_result))

        # Sort by original index to maintain seed order
        results.sort(key=lambda x: x[0])
        results = [r for _, r in results]
        if run_baseline_flag:
            baseline_results.sort(key=lambda x: x[0])
            baseline_results = [r for _, r in baseline_results]
    else:
        # Sequential execution
        for i, seed in enumerate(seeds):
            if verbose and i % 10 == 0:
                print(f"Running path {i+1}/{n_paths}...")

            # Run LP strategy
            result = run_simulation(cfg, seed=seed, verbose=False)
            results.append(result)

            # Run baseline if requested
            if run_baseline_flag:
                bl_result = run_baseline(
                    prices=result.prices,
                    regime_path=result.regime_path,
                    cfg=cfg,
                    seed=seed,
                    verbose=False,
                )
                baseline_results.append(bl_result)

    if verbose:
        print(f"Batch complete: {n_paths} paths")
        stats = BatchResult(results, baseline_results, cfg, seeds).statistics()
        print(f"  Mean P&L: ${stats['pnl_mean']:,.2f}")
        print(f"  Std P&L: ${stats['pnl_std']:,.2f}")
        print(f"  Sharpe: {stats['sharpe_of_pnls']:.2f}")
        if run_baseline_flag:
            print(f"  vs Baseline: ${stats['pnl_vs_baseline']:,.2f}")
            print(f"  Outperformance rate: {stats['outperformance_rate']:.1%}")

    return BatchResult(results, baseline_results, cfg, seeds)


@dataclass
class ScenarioSweepResult:
    """
    Results from a parameter sweep.

    Attributes:
        param_name: Parameter that was swept
        param_values: Values tested
        batch_results: BatchResult for each parameter value
    """

    param_name: str
    param_values: List[Any]
    batch_results: List[BatchResult]

    def get_metric_vs_param(self, metric: str) -> tuple:
        """
        Get a metric across parameter values.

        Args:
            metric: Key from BatchResult.statistics()

        Returns:
            (param_values, metric_values) tuple
        """
        metric_values = [br.statistics()[metric] for br in self.batch_results]
        return self.param_values, metric_values


def run_scenario_sweep(
    base_cfg: SimConfig,
    param_name: str,
    param_values: List[Any],
    n_paths_per_scenario: int = 50,
    run_baseline_flag: bool = True,
    verbose: bool = True,
) -> ScenarioSweepResult:
    """
    Sweep a single parameter across values.

    Args:
        base_cfg: Base configuration
        param_name: Parameter to vary (e.g., 'IC', 'rfq_rate_per_day')
        param_values: List of values to test
        n_paths_per_scenario: Paths per parameter value
        run_baseline_flag: Run baseline for comparison
        verbose: Print progress

    Returns:
        ScenarioSweepResult with all scenarios
    """
    batch_results = []

    for i, val in enumerate(param_values):
        if verbose:
            print(f"\nScenario {i+1}/{len(param_values)}: {param_name}={val}")

        # Create modified config
        cfg_dict = {
            field: getattr(base_cfg, field)
            for field in base_cfg.__dataclass_fields__
            if not field.startswith('_')
        }
        cfg_dict[param_name] = val
        cfg = SimConfig(**cfg_dict)

        # Run batch
        batch = run_batch(
            cfg=cfg,
            n_paths=n_paths_per_scenario,
            run_baseline_flag=run_baseline_flag,
            verbose=verbose,
        )
        batch_results.append(batch)

    return ScenarioSweepResult(
        param_name=param_name,
        param_values=param_values,
        batch_results=batch_results,
    )


def run_grid_sweep(
    base_cfg: SimConfig,
    param_grid: Dict[str, List[Any]],
    n_paths_per_scenario: int = 20,
    verbose: bool = True,
) -> Dict[tuple, BatchResult]:
    """
    Sweep multiple parameters in a grid.

    Args:
        base_cfg: Base configuration
        param_grid: Dict mapping param names to value lists
        n_paths_per_scenario: Paths per scenario
        verbose: Print progress

    Returns:
        Dict mapping (param1_val, param2_val, ...) to BatchResult
    """
    import itertools

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    results = {}
    total = np.prod([len(v) for v in param_values])

    for i, combo in enumerate(itertools.product(*param_values)):
        if verbose:
            print(f"\nGrid scenario {i+1}/{total}: {dict(zip(param_names, combo))}")

        # Create modified config
        cfg_dict = {
            field: getattr(base_cfg, field)
            for field in base_cfg.__dataclass_fields__
            if not field.startswith('_')
        }
        for name, val in zip(param_names, combo):
            cfg_dict[name] = val
        cfg = SimConfig(**cfg_dict)

        # Run batch
        batch = run_batch(
            cfg=cfg,
            n_paths=n_paths_per_scenario,
            run_baseline_flag=True,
            verbose=False,
        )
        results[combo] = batch

    return results
