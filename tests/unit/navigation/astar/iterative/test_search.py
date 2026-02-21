"""Tests for iterative A* search execution utilities."""

import pytest
from unittest.mock import MagicMock, patch, call


class MockSearchState:
    """Mock A* search state."""

    def __init__(self, path=None, iterations=100):
        self.path = path
        self.iterations = iterations


class MockCNF:
    """Mock CNF for testing."""

    def __init__(self, coords, elements=None, xi=1.5, delta=10):
        self.coords = coords
        self.elements = elements or ["Zr"]
        self.xi = xi
        self.delta = delta


class MockCalculator:
    """Mock energy calculator."""

    def __init__(self, energy=-10.0):
        self.energy = energy

    def calculate_energy(self, cnf):
        return self.energy


class TestSearchAtCeiling:
    @patch('cnf.navigation.astar.iterative._search.astar_pathfind')
    @patch('cnf.navigation.astar.iterative._search.make_heuristic')
    def test_returns_not_found_when_no_path(self, mock_heuristic, mock_astar):
        """Should return found=False when A* finds no path."""
        from cnf.navigation.astar.iterative._search import search_at_ceiling

        mock_astar.return_value = MockSearchState(path=None, iterations=500)
        mock_heuristic.return_value = lambda x, y: 0

        start_cnfs = [MockCNF((1, 2, 3))]
        goal_cnfs = [MockCNF((4, 5, 6))]
        calc = MockCalculator()
        cache = {}

        result = search_at_ceiling(
            ceiling=-10.0,
            start_cnfs=start_cnfs,
            goal_cnfs=goal_cnfs,
            elements=["Zr"],
            xi=1.5,
            delta=10,
            calc=calc,
            cache=cache,
            dropout=0.1,
            max_iters=1000,
            beam_width=500,
            heuristic_mode="manhattan",
            heuristic_weight=0.5,
        )

        assert result["ceiling"] == -10.0
        assert result["found"] is False
        assert result["iterations"] == 500

    @patch('cnf.navigation.astar.iterative._search.astar_pathfind')
    @patch('cnf.navigation.astar.iterative._search.make_heuristic')
    @patch('cnf.navigation.astar.iterative._search.evaluate_path_energies')
    @patch('cnf.navigation.astar.iterative._search.path_barrier')
    def test_returns_found_with_path_data(
        self, mock_barrier, mock_eval, mock_heuristic, mock_astar
    ):
        """Should return path data when A* finds a path."""
        from cnf.navigation.astar.iterative._search import search_at_ceiling

        path = [(1, 2, 3), (4, 5, 6)]
        mock_astar.return_value = MockSearchState(path=path, iterations=250)
        mock_heuristic.return_value = lambda x, y: 0
        mock_eval.return_value = [-10.0, -9.5]
        mock_barrier.return_value = -9.5

        start_cnfs = [MockCNF((1, 2, 3))]
        goal_cnfs = [MockCNF((4, 5, 6))]
        calc = MockCalculator()
        cache = {}

        result = search_at_ceiling(
            ceiling=-10.0,
            start_cnfs=start_cnfs,
            goal_cnfs=goal_cnfs,
            elements=["Zr"],
            xi=1.5,
            delta=10,
            calc=calc,
            cache=cache,
            dropout=0.1,
            max_iters=1000,
            beam_width=500,
            heuristic_mode="manhattan",
            heuristic_weight=0.5,
        )

        assert result["ceiling"] == -10.0
        assert result["found"] is True
        assert result["iterations"] == 250
        assert result["barrier"] == -9.5
        assert result["path"] == path
        assert result["energies"] == [-10.0, -9.5]
        assert result["path_length"] == 2


class TestRetrySearch:
    @patch('cnf.navigation.astar.iterative._search.search_at_ceiling')
    def test_returns_on_first_success(self, mock_search, capsys):
        """Should return immediately when first attempt succeeds."""
        from cnf.navigation.astar.iterative._search import retry_search

        mock_search.return_value = {
            "ceiling": -10.0,
            "found": True,
            "iterations": 100,
            "barrier": -9.5,
            "path": [(1, 2, 3)],
            "energies": [-9.5],
            "path_length": 1,
            "energy_evals": 50,
        }

        result = retry_search(
            ceiling=-10.0,
            start_cnfs=[MockCNF((1, 2, 3))],
            goal_cnfs=[MockCNF((4, 5, 6))],
            elements=["Zr"],
            xi=1.5,
            delta=10,
            calc=MockCalculator(),
            cache={},
            dropout=0.1,
            max_iters=1000,
            beam_width=500,
            heuristic_mode="manhattan",
            heuristic_weight=0.5,
            attempts=3,
            verbose=False,
        )

        assert result["found"] is True
        assert mock_search.call_count == 1

    @patch('cnf.navigation.astar.iterative._search.search_at_ceiling')
    def test_retries_on_failure(self, mock_search, capsys):
        """Should retry up to `attempts` times on failure."""
        from cnf.navigation.astar.iterative._search import retry_search

        mock_search.return_value = {
            "ceiling": -10.0,
            "found": False,
            "iterations": 1000,
            "energy_evals": 100,
        }

        result = retry_search(
            ceiling=-10.0,
            start_cnfs=[MockCNF((1, 2, 3))],
            goal_cnfs=[MockCNF((4, 5, 6))],
            elements=["Zr"],
            xi=1.5,
            delta=10,
            calc=MockCalculator(),
            cache={},
            dropout=0.1,
            max_iters=1000,
            beam_width=500,
            heuristic_mode="manhattan",
            heuristic_weight=0.5,
            attempts=3,
            verbose=False,
        )

        assert result["found"] is False
        assert mock_search.call_count == 3

    @patch('cnf.navigation.astar.iterative._search.search_at_ceiling')
    def test_bumps_max_iters_on_retry(self, mock_search, capsys):
        """Should increase max_iters by scale factor on each retry."""
        from cnf.navigation.astar.iterative._search import retry_search

        mock_search.return_value = {
            "ceiling": -10.0,
            "found": False,
            "iterations": 1000,
            "energy_evals": 100,
        }

        retry_search(
            ceiling=-10.0,
            start_cnfs=[MockCNF((1, 2, 3))],
            goal_cnfs=[MockCNF((4, 5, 6))],
            elements=["Zr"],
            xi=1.5,
            delta=10,
            calc=MockCalculator(),
            cache={},
            dropout=0.1,
            max_iters=100,
            beam_width=500,
            heuristic_mode="manhattan",
            heuristic_weight=0.5,
            attempts=3,
            max_iters_scale=2.0,
            verbose=False,
        )

        # Check max_iters in each call: 100, 200, 400
        # max_iters is positional arg at index 9: (ceiling, start_cnfs, goal_cnfs,
        #   elements, xi, delta, calc, cache, dropout, max_iters, ...)
        calls = mock_search.call_args_list
        assert calls[0][0][9] == 100
        assert calls[1][0][9] == 200
        assert calls[2][0][9] == 400

    @patch('cnf.navigation.astar.iterative._search.search_at_ceiling')
    def test_succeeds_on_later_attempt(self, mock_search, capsys):
        """Should return success if a later attempt succeeds."""
        from cnf.navigation.astar.iterative._search import retry_search

        mock_search.side_effect = [
            {"ceiling": -10.0, "found": False, "iterations": 1000, "energy_evals": 100},
            {"ceiling": -10.0, "found": False, "iterations": 1000, "energy_evals": 100},
            {
                "ceiling": -10.0, "found": True, "iterations": 500,
                "barrier": -9.5, "path": [(1,)], "energies": [-9.5],
                "path_length": 1, "energy_evals": 200,
            },
        ]

        result = retry_search(
            ceiling=-10.0,
            start_cnfs=[MockCNF((1, 2, 3))],
            goal_cnfs=[MockCNF((4, 5, 6))],
            elements=["Zr"],
            xi=1.5,
            delta=10,
            calc=MockCalculator(),
            cache={},
            dropout=0.1,
            max_iters=100,
            beam_width=500,
            heuristic_mode="manhattan",
            heuristic_weight=0.5,
            attempts=5,
            verbose=False,
        )

        assert result["found"] is True
        assert mock_search.call_count == 3

    @patch('cnf.navigation.astar.iterative._search.search_at_ceiling')
    def test_verbose_prints_progress(self, mock_search, capsys):
        """Verbose mode should print progress messages."""
        from cnf.navigation.astar.iterative._search import retry_search

        mock_search.return_value = {
            "ceiling": -10.0,
            "found": True,
            "iterations": 100,
            "barrier": -9.5,
            "path": [(1,)],
            "energies": [-9.5],
            "path_length": 1,
            "energy_evals": 50,
        }

        retry_search(
            ceiling=-10.0,
            start_cnfs=[MockCNF((1, 2, 3))],
            goal_cnfs=[MockCNF((4, 5, 6))],
            elements=["Zr"],
            xi=1.5,
            delta=10,
            calc=MockCalculator(),
            cache={},
            dropout=0.1,
            max_iters=100,
            beam_width=500,
            heuristic_mode="manhattan",
            heuristic_weight=0.5,
            attempts=1,
            verbose=True,
        )

        captured = capsys.readouterr()
        assert "starting" in captured.out
        assert "path found" in captured.out

    @patch('cnf.navigation.astar.iterative._search.search_at_ceiling')
    def test_silent_mode_no_print(self, mock_search, capsys):
        """Silent mode (verbose=False) should not print."""
        from cnf.navigation.astar.iterative._search import retry_search

        mock_search.return_value = {
            "ceiling": -10.0,
            "found": True,
            "iterations": 100,
            "barrier": -9.5,
            "path": [(1,)],
            "energies": [-9.5],
            "path_length": 1,
            "energy_evals": 50,
        }

        retry_search(
            ceiling=-10.0,
            start_cnfs=[MockCNF((1, 2, 3))],
            goal_cnfs=[MockCNF((4, 5, 6))],
            elements=["Zr"],
            xi=1.5,
            delta=10,
            calc=MockCalculator(),
            cache={},
            dropout=0.1,
            max_iters=100,
            beam_width=500,
            heuristic_mode="manhattan",
            heuristic_weight=0.5,
            attempts=1,
            verbose=False,
        )

        captured = capsys.readouterr()
        assert captured.out == ""


class TestSearchCeilingWithAttempts:
    @patch('cnf.navigation.astar.iterative._search.retry_search')
    def test_delegates_to_retry_search(self, mock_retry):
        """Should delegate to retry_search with all parameters."""
        from cnf.navigation.astar.iterative._search import search_ceiling_with_attempts

        mock_retry.return_value = {"found": True}

        start_cnfs = [MockCNF((1, 2, 3))]
        goal_cnfs = [MockCNF((4, 5, 6))]
        calc = MockCalculator()
        cache = {}

        search_ceiling_with_attempts(
            ceiling=-10.0,
            start_cnfs=start_cnfs,
            goal_cnfs=goal_cnfs,
            elements=["Zr"],
            xi=1.5,
            delta=10,
            calc=calc,
            cache=cache,
            dropout=0.1,
            max_iters=1000,
            beam_width=500,
            heuristic_mode="manhattan",
            heuristic_weight=0.5,
            attempts=5,
            verbose=True,
        )

        mock_retry.assert_called_once()
        # attempts is positional arg at index 13, verbose is kwarg
        call_args = mock_retry.call_args
        assert call_args[0][13] == 5  # attempts
        assert call_args.kwargs['verbose'] is True

    @patch('cnf.navigation.astar.iterative._search.retry_search')
    def test_passes_verbose_false(self, mock_retry):
        """Should pass verbose=False to retry_search."""
        from cnf.navigation.astar.iterative._search import search_ceiling_with_attempts

        mock_retry.return_value = {"found": False}

        search_ceiling_with_attempts(
            ceiling=-10.0,
            start_cnfs=[MockCNF((1, 2, 3))],
            goal_cnfs=[MockCNF((4, 5, 6))],
            elements=["Zr"],
            xi=1.5,
            delta=10,
            calc=MockCalculator(),
            cache={},
            dropout=0.1,
            max_iters=1000,
            beam_width=500,
            heuristic_mode="manhattan",
            heuristic_weight=0.5,
            attempts=3,
            verbose=False,
        )

        call_kwargs = mock_retry.call_args.kwargs
        assert call_kwargs.get('verbose') is False

    @patch('cnf.navigation.astar.iterative._search.retry_search')
    def test_respects_attempts_in_silent_mode(self, mock_retry):
        """Should respect attempts parameter even when verbose=False."""
        from cnf.navigation.astar.iterative._search import search_ceiling_with_attempts

        mock_retry.return_value = {"found": False}

        search_ceiling_with_attempts(
            ceiling=-10.0,
            start_cnfs=[MockCNF((1, 2, 3))],
            goal_cnfs=[MockCNF((4, 5, 6))],
            elements=["Zr"],
            xi=1.5,
            delta=10,
            calc=MockCalculator(),
            cache={},
            dropout=0.1,
            max_iters=1000,
            beam_width=500,
            heuristic_mode="manhattan",
            heuristic_weight=0.5,
            attempts=7,
            verbose=False,
        )

        # Verify attempts=7 was passed (this was the bug - before it would
        # ignore attempts when verbose=False)
        # attempts is positional arg at index 13
        call_args = mock_retry.call_args
        assert call_args[0][13] == 7
