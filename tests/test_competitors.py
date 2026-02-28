"""Tests for competitor model (explicit dealer simulation)."""

import numpy as np
import pytest

import sys
sys.path.insert(0, 'src')

from rfq_simulator.config import SimConfig
from rfq_simulator.world.rfq_stream import RFQEvent
from rfq_simulator.world.competitors import (
    DealerPool,
    DealerQuote,
    simulate_competition,
    compute_dealer_markup,
    compute_response_probability,
    compute_empirical_win_rate,
)


def make_rfq(is_client_buy=True, size=1, n_dealers=4, toxicity=0.2):
    """Helper to create an RFQ event."""
    return RFQEvent(
        time=100.0,
        is_client_buy=is_client_buy,
        size=size,
        n_dealers=n_dealers,
        toxicity=toxicity,
    )


class TestDealerPool:
    """Tests for dealer pool management."""

    def test_dealer_pool_initialization(self):
        """Dealer pool should initialize with biases."""
        cfg = SimConfig()
        rng = np.random.default_rng(42)

        pool = DealerPool(cfg.n_dealers_max, cfg, rng)

        assert len(pool.dealer_biases) == cfg.n_dealers_max
        assert pool.current_day == 0

    def test_bias_refresh_with_street_lean(self):
        """Dealer biases should center on street lean after refresh."""
        cfg = SimConfig(dealer_bias_std_bps=3.0)
        rng = np.random.default_rng(42)

        pool = DealerPool(cfg.n_dealers_max, cfg, rng)
        initial_biases = pool.dealer_biases.copy()

        # Refresh with positive street lean
        pool.refresh_biases(day=1, street_lean=5.0)
        new_biases = pool.dealer_biases

        # Biases should have changed
        assert not np.array_equal(initial_biases, new_biases)

        # Mean should be close to street lean (with some variance)
        assert abs(np.mean(new_biases) - 5.0) < 2.0  # Allow statistical variance

    def test_no_refresh_same_day(self):
        """Biases should not refresh on same day."""
        cfg = SimConfig()
        rng = np.random.default_rng(42)

        pool = DealerPool(cfg.n_dealers_max, cfg, rng)
        pool.refresh_biases(day=1, street_lean=0.0)
        biases_after_first = pool.dealer_biases.copy()

        pool.refresh_biases(day=1, street_lean=10.0)  # Same day

        assert np.array_equal(pool.dealer_biases, biases_after_first)


class TestDealerMarkup:
    """Tests for dealer markup computation."""

    def test_base_markup(self):
        """Base markup should reflect config value."""
        cfg = SimConfig(markup_base_bps=8.0)
        rng = np.random.default_rng(42)
        rfq = make_rfq(n_dealers=int(cfg.n_dealers_mean), size=1, toxicity=0.0)

        markup = compute_dealer_markup(rfq, cfg, rng)

        # With N=mean, size=1, tox=0, markup should be close to base
        assert abs(markup - cfg.markup_base_bps) < 1.0

    def test_markup_increases_with_size(self):
        """Larger trades should have higher markups."""
        cfg = SimConfig()
        rng = np.random.default_rng(42)

        rfq_small = make_rfq(size=1)
        rfq_large = make_rfq(size=5)

        markup_small = compute_dealer_markup(rfq_small, cfg, rng)
        markup_large = compute_dealer_markup(rfq_large, cfg, rng)

        assert markup_large > markup_small

    def test_markup_increases_with_toxicity(self):
        """Toxic flow should have higher markups."""
        cfg = SimConfig()
        rng = np.random.default_rng(42)

        rfq_clean = make_rfq(toxicity=0.0)
        rfq_toxic = make_rfq(toxicity=0.5)

        markup_clean = compute_dealer_markup(rfq_clean, cfg, rng)
        markup_toxic = compute_dealer_markup(rfq_toxic, cfg, rng)

        assert markup_toxic > markup_clean


class TestResponseProbability:
    """Tests for dealer response probability."""

    def test_base_response_rate(self):
        """Small, clean RFQ should have high response rate."""
        cfg = SimConfig(respond_base=0.85)
        rfq = make_rfq(size=1, toxicity=0.0)

        prob = compute_response_probability(rfq, cfg)

        assert prob >= 0.8

    def test_response_decreases_with_size(self):
        """Larger trades should have lower response rates."""
        cfg = SimConfig()

        prob_small = compute_response_probability(make_rfq(size=1), cfg)
        prob_large = compute_response_probability(make_rfq(size=5), cfg)

        assert prob_large < prob_small

    def test_response_bounded(self):
        """Response probability should be in [0.3, 1.0]."""
        cfg = SimConfig()

        # Very large, toxic RFQ
        rfq_bad = make_rfq(size=10, toxicity=0.9)
        prob = compute_response_probability(rfq_bad, cfg)

        assert 0.3 <= prob <= 1.0


class TestCompetitionSimulation:
    """Tests for full competition simulation."""

    def test_competition_result_structure(self):
        """Competition should return valid result."""
        cfg = SimConfig()
        rng = np.random.default_rng(42)

        pool = DealerPool(cfg.n_dealers_max, cfg, rng)
        rfq = make_rfq(n_dealers=4)
        p_true = 100.0
        trader_price = 100.05  # Slightly wide

        result = simulate_competition(
            rfq=rfq,
            p_true=p_true,
            trader_price=trader_price,
            dealer_pool=pool,
            street_lean=0.0,
            rng=rng,
        )

        assert result.trader_price == trader_price
        assert len(result.competitor_quotes) == rfq.n_dealers - 1  # N-1 competitors

    def test_better_price_wins(self):
        """Significantly better price should almost always win."""
        cfg = SimConfig()
        rng = np.random.default_rng(42)

        pool = DealerPool(cfg.n_dealers_max, cfg, rng)
        rfq = make_rfq(is_client_buy=True, n_dealers=4)  # We're offering
        p_true = 100.0

        wins = 0
        n_trials = 100
        for _ in range(n_trials):
            # Very aggressive price (below true)
            result = simulate_competition(
                rfq=rfq,
                p_true=p_true,
                trader_price=p_true - 0.05,  # 5 bps below true
                dealer_pool=pool,
                street_lean=0.0,
                rng=rng,
            )
            if result.trader_won:
                wins += 1

        # Should win most of the time with aggressive pricing
        assert wins / n_trials > 0.7


class TestWinRateCurve:
    """Tests for win-rate curve validation."""

    def test_win_rate_monotonic(self):
        """Win rate should decrease with higher markup (for offers)."""
        cfg = SimConfig()
        rng = np.random.default_rng(42)

        pool = DealerPool(cfg.n_dealers_max, cfg, rng)
        rfq = make_rfq(is_client_buy=True)
        p_true = 100.0

        # Test at different markups
        markups = [0, 5, 10, 15]
        win_rates = []

        for m in markups:
            wr = compute_empirical_win_rate(
                markup_bps=m,
                rfq=rfq,
                p_true=p_true,
                dealer_pool=pool,
                street_lean=0.0,
                n_simulations=200,
                rng=rng,
            )
            win_rates.append(wr)

        # Win rate should generally decrease (allow some noise)
        assert win_rates[0] > win_rates[-1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
