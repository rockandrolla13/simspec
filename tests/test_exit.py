"""Tests for hybrid exit logic."""

import numpy as np
import pytest

import sys
sys.path.insert(0, 'src')

from rfq_simulator.config import SimConfig
from rfq_simulator.agent.exit import (
    ExitMode,
    ExitDecision,
    HybridExitManager,
    compute_urgency_adjusted_lean,
)


class TestExitModeTransition:
    """Tests for patient/aggressive mode transitions."""

    def test_patient_before_window(self):
        """Should be in patient mode before aggressive window."""
        cfg = SimConfig(
            alpha_horizon_days=10.0,
            aggress_window_hours=8.0,
        )
        exit_mgr = HybridExitManager(cfg)

        horizon_minutes = cfg.alpha_horizon_days * cfg.minutes_per_day
        t_signal = 0.0

        # Well before aggressive window
        mode = exit_mgr.check_exit_mode(
            current_minute=100.0,
            t_signal=t_signal,
            horizon_minutes=horizon_minutes,
        )

        assert mode == ExitMode.PATIENT

    def test_aggressive_in_window(self):
        """Should be in aggressive mode within window."""
        cfg = SimConfig(
            alpha_horizon_days=10.0,
            aggress_window_hours=8.0,
        )
        exit_mgr = HybridExitManager(cfg)

        horizon_minutes = cfg.alpha_horizon_days * cfg.minutes_per_day  # 4800 min
        t_signal = 0.0
        expiry = t_signal + horizon_minutes
        aggress_start = expiry - cfg.aggress_window_hours * 60  # 480 before expiry

        # Just inside aggressive window
        mode = exit_mgr.check_exit_mode(
            current_minute=aggress_start + 10,
            t_signal=t_signal,
            horizon_minutes=horizon_minutes,
        )

        assert mode == ExitMode.AGGRESSIVE

    def test_mode_transition_boundary(self):
        """Mode should change exactly at boundary."""
        cfg = SimConfig(
            alpha_horizon_days=3.0,  # 1440 minutes
            aggress_window_hours=8.0,  # 480 minutes
        )
        exit_mgr = HybridExitManager(cfg)

        horizon_minutes = cfg.alpha_horizon_days * cfg.minutes_per_day
        t_signal = 0.0
        boundary = t_signal + horizon_minutes - cfg.aggress_window_hours * 60

        # Just before boundary
        mode_before = exit_mgr.check_exit_mode(boundary - 1, t_signal, horizon_minutes)

        # At boundary
        mode_at = exit_mgr.check_exit_mode(boundary, t_signal, horizon_minutes)

        assert mode_before == ExitMode.PATIENT
        assert mode_at == ExitMode.AGGRESSIVE


class TestExitDecision:
    """Tests for exit decision logic."""

    def test_no_aggress_in_patient_mode(self):
        """Should not aggress in patient mode."""
        cfg = SimConfig(alpha_horizon_days=10.0, aggress_window_hours=8.0)
        exit_mgr = HybridExitManager(cfg)

        horizon_minutes = cfg.alpha_horizon_days * cfg.minutes_per_day

        decision = exit_mgr.get_exit_decision(
            current_minute=100.0,
            current_q=5.0,
            t_signal=0.0,
            horizon_minutes=horizon_minutes,
        )

        assert decision.mode == ExitMode.PATIENT
        assert decision.should_aggress is False
        assert decision.reason == "patient_mode"

    def test_no_aggress_flat_position(self):
        """Should not aggress if position is flat."""
        cfg = SimConfig(alpha_horizon_days=3.0, aggress_window_hours=8.0)
        exit_mgr = HybridExitManager(cfg)

        horizon_minutes = cfg.alpha_horizon_days * cfg.minutes_per_day
        # Time well into aggressive window
        late_time = horizon_minutes - 100

        decision = exit_mgr.get_exit_decision(
            current_minute=late_time,
            current_q=0.0,  # Flat
            t_signal=0.0,
            horizon_minutes=horizon_minutes,
        )

        assert decision.mode == ExitMode.AGGRESSIVE
        assert decision.should_aggress is False
        assert decision.reason == "position_flat"

    def test_aggress_with_position(self):
        """Should aggress when has position in aggressive window."""
        cfg = SimConfig(alpha_horizon_days=3.0, aggress_window_hours=8.0)
        exit_mgr = HybridExitManager(cfg)

        horizon_minutes = cfg.alpha_horizon_days * cfg.minutes_per_day
        late_time = horizon_minutes - 100

        decision = exit_mgr.get_exit_decision(
            current_minute=late_time,
            current_q=5.0,  # Long position
            t_signal=0.0,
            horizon_minutes=horizon_minutes,
        )

        assert decision.mode == ExitMode.AGGRESSIVE
        assert decision.should_aggress is True
        assert decision.aggress_size < 0  # Selling to reduce long

    def test_aggress_direction_short(self):
        """Short position should buy to cover."""
        cfg = SimConfig(alpha_horizon_days=3.0, aggress_window_hours=8.0)
        exit_mgr = HybridExitManager(cfg)

        horizon_minutes = cfg.alpha_horizon_days * cfg.minutes_per_day
        late_time = horizon_minutes - 100

        decision = exit_mgr.get_exit_decision(
            current_minute=late_time,
            current_q=-5.0,  # Short position
            t_signal=0.0,
            horizon_minutes=horizon_minutes,
        )

        assert decision.aggress_size > 0  # Buying to cover short


class TestAggressCost:
    """Tests for aggressive execution cost calculation."""

    def test_cost_zero_for_zero_size(self):
        """Zero size should have zero cost."""
        cfg = SimConfig()
        exit_mgr = HybridExitManager(cfg)

        cost = exit_mgr.compute_aggress_cost(0.0)

        assert cost == 0.0

    def test_cost_positive_for_trade(self):
        """Trade should have positive cost."""
        cfg = SimConfig(
            aggress_halfspread_bps=8.0,
            aggress_impact_bps=2.0,
        )
        exit_mgr = HybridExitManager(cfg)

        cost = exit_mgr.compute_aggress_cost(1.0)

        assert cost > 0

    def test_cost_scales_with_size(self):
        """Larger trades should cost more."""
        cfg = SimConfig()
        exit_mgr = HybridExitManager(cfg)

        cost_small = exit_mgr.compute_aggress_cost(1.0)
        cost_large = exit_mgr.compute_aggress_cost(5.0)

        assert cost_large > cost_small

    def test_cost_sign_independent(self):
        """Buy and sell should have same cost magnitude."""
        cfg = SimConfig()
        exit_mgr = HybridExitManager(cfg)

        cost_buy = exit_mgr.compute_aggress_cost(3.0)
        cost_sell = exit_mgr.compute_aggress_cost(-3.0)

        assert cost_buy == cost_sell


class TestReset:
    """Tests for exit manager reset."""

    def test_reset_clears_state(self):
        """Reset should clear last aggress time."""
        cfg = SimConfig(alpha_horizon_days=3.0, aggress_window_hours=8.0)
        exit_mgr = HybridExitManager(cfg)

        horizon_minutes = cfg.alpha_horizon_days * cfg.minutes_per_day
        late_time = horizon_minutes - 100

        # Trigger an aggressive trade
        exit_mgr.get_exit_decision(late_time, 5.0, 0.0, horizon_minutes)

        # Reset
        exit_mgr.reset()

        assert exit_mgr.mode == ExitMode.PATIENT
        assert exit_mgr.last_aggress_time is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
