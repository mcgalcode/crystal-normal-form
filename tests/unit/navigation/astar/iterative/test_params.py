"""Tests for iterative A* parameter adaptation utilities."""

import pytest

from cnf.navigation.astar.iterative.ratchet.params import adapt_params


class TestAdaptParams:
    """Tests for the adapt_params function."""

    def test_high_success_rate_keeps_dropout(self):
        """Success rate >= 67% should keep current dropout."""
        new_dropout, new_max_iters = adapt_params(
            found=7, total=10, successful_iters=[100, 150, 200],
            current_dropout=0.3, min_dropout=0.1, current_max_iters=500,
        )

        assert new_dropout == 0.3  # Unchanged
        assert new_max_iters == 300  # 1.5 * max(successful_iters) = 1.5 * 200

    def test_medium_success_rate_reduces_dropout(self):
        """Success rate 33-67% should reduce dropout by 0.1."""
        new_dropout, new_max_iters = adapt_params(
            found=5, total=10, successful_iters=[100, 150],
            current_dropout=0.3, min_dropout=0.1, current_max_iters=500,
        )

        assert new_dropout == pytest.approx(0.2)  # 0.3 - 0.1
        assert new_max_iters == 225  # 1.5 * max(150)

    def test_low_success_rate_halves_dropout(self):
        """Success rate < 33% should halve dropout."""
        new_dropout, new_max_iters = adapt_params(
            found=2, total=10, successful_iters=[100],
            current_dropout=0.4, min_dropout=0.1, current_max_iters=500,
        )

        assert new_dropout == 0.2  # 0.4 * 0.5
        assert new_max_iters == 150  # 1.5 * max(100)

    def test_dropout_respects_minimum(self):
        """Dropout should never go below min_dropout."""
        new_dropout, _ = adapt_params(
            found=1, total=10, successful_iters=[100],
            current_dropout=0.15, min_dropout=0.1, current_max_iters=500,
        )

        assert new_dropout == 0.1  # min_dropout, not 0.15 * 0.5 = 0.075

    def test_no_successes_doubles_max_iters(self):
        """No successful iterations should double max_iters."""
        new_dropout, new_max_iters = adapt_params(
            found=0, total=10, successful_iters=[],
            current_dropout=0.3, min_dropout=0.1, current_max_iters=500,
        )

        assert new_max_iters == 1000  # 500 * 2

    def test_max_iters_respects_cap(self):
        """max_iters should not exceed the max_iters cap."""
        new_dropout, new_max_iters = adapt_params(
            found=5, total=10, successful_iters=[8000],
            current_dropout=0.3, min_dropout=0.1, current_max_iters=5000,
            max_iters=10000,
        )

        # 1.5 * 8000 = 12000, but capped at 10000
        assert new_max_iters == 10000

    def test_no_successes_max_iters_respects_cap(self):
        """Doubling max_iters should respect the cap."""
        new_dropout, new_max_iters = adapt_params(
            found=0, total=10, successful_iters=[],
            current_dropout=0.3, min_dropout=0.1, current_max_iters=6000,
            max_iters=10000,
        )

        # 6000 * 2 = 12000, but capped at 10000
        assert new_max_iters == 10000

    def test_zero_total_gives_zero_rate(self):
        """Zero total searches should be treated as 0% success rate."""
        new_dropout, _ = adapt_params(
            found=0, total=0, successful_iters=[],
            current_dropout=0.4, min_dropout=0.1, current_max_iters=500,
        )

        # 0% success rate should halve dropout
        assert new_dropout == pytest.approx(0.2)

    def test_exactly_67_percent(self):
        """Exactly 67% should keep dropout unchanged."""
        new_dropout, _ = adapt_params(
            found=67, total=100, successful_iters=[100],
            current_dropout=0.3, min_dropout=0.1, current_max_iters=500,
        )

        assert new_dropout == 0.3

    def test_exactly_33_percent(self):
        """Exactly 33% should reduce dropout by 0.1."""
        new_dropout, _ = adapt_params(
            found=33, total=100, successful_iters=[100],
            current_dropout=0.3, min_dropout=0.1, current_max_iters=500,
        )

        assert new_dropout == pytest.approx(0.2)

    def test_default_max_iters_infinity(self):
        """Default max_iters cap should be infinity."""
        new_dropout, new_max_iters = adapt_params(
            found=5, total=10, successful_iters=[100000],
            current_dropout=0.3, min_dropout=0.1, current_max_iters=500,
        )

        # 1.5 * 100000 = 150000 (no cap)
        assert new_max_iters == 150000
