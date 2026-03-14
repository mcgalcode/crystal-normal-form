"""Tests for Phase 1 parameter search function."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from cnf.navigation.astar.models import ParameterSearchResult


class MockUnitCell:
    """Mock UnitCell for testing."""

    def __init__(self, structure_dict=None):
        self._structure_dict = structure_dict or {"lattice": {"a": 3.0}}

    def to_pymatgen_structure(self):
        return MagicMock(as_dict=lambda: self._structure_dict)


class TestBinarySearchMinDistance:
    @patch('cnf.navigation.astar.iterative.search.search.astar_rust')
    def test_finds_max_min_distance(self, mock_astar):
        """Should binary search to find the maximum working min_distance."""
        from cnf.navigation.astar.iterative.search.search import _binary_search_min_distance

        # Paths work below 1.0, fail at or above 1.0
        def astar_side_effect(start, goal, min_distance, **kwargs):
            if min_distance < 1.0:
                return ([(1,), (2,)], 100)
            return (None, 500)

        mock_astar.side_effect = astar_side_effect

        mock_cnfs = [MagicMock(coords=(1, 2, 3))]

        best_dist, iters = _binary_search_min_distance(
            start_cnfs=mock_cnfs,
            goal_cnfs=mock_cnfs,
            min_dist_low=0.5,
            min_dist_high=1.5,
            max_iterations=1000,
            beam_width=500,
            dropout=0.3,
            tolerance=0.05,
            verbosity=0,
        )

        # Should converge to just below 1.0
        assert best_dist is not None
        assert 0.9 < best_dist < 1.0

    @patch('cnf.navigation.astar.iterative.search.search.astar_rust')
    def test_returns_none_when_no_path_at_low_bound(self, mock_astar):
        """Should return None if no path found even at lowest min_distance."""
        from cnf.navigation.astar.iterative.search.search import _binary_search_min_distance

        # No paths work at any min_distance
        mock_astar.return_value = (None, 500)

        mock_cnfs = [MagicMock(coords=(1, 2, 3))]

        best_dist, iters = _binary_search_min_distance(
            start_cnfs=mock_cnfs,
            goal_cnfs=mock_cnfs,
            min_dist_low=0.5,
            min_dist_high=1.5,
            max_iterations=1000,
            beam_width=500,
            dropout=0.3,
            tolerance=0.05,
            verbosity=0,
        )

        assert best_dist is None
        assert iters is None


class TestSearchAtResolution:
    @patch('cnf.navigation.astar.iterative.search.search._binary_search_min_distance')
    @patch('cnf.navigation.astar.iterative.search.search.get_endpoint_cnfs_with_resolution')
    def test_computes_delta_from_atom_step(self, mock_endpoints, mock_binary):
        """Should compute delta from atom step length and structure."""
        from cnf.navigation.astar.iterative.search.search import _search_at_resolution

        mock_cnf = MagicMock()
        mock_cnf.elements = ("Zr",)
        # get_endpoint_cnfs_with_resolution returns (start_cnfs, goal_cnfs, delta)
        mock_endpoints.return_value = ([mock_cnf], [mock_cnf], 12)
        mock_binary.return_value = (0.8, 100)

        start_uc = MockUnitCell()
        end_uc = MockUnitCell()

        result = _search_at_resolution(
            start_uc=start_uc,
            end_uc=end_uc,
            xi=1.5,
            atom_step_length=0.3,
            min_dist_low=0.5,
            min_dist_high=1.5,
            max_iterations=1000,
            beam_width=500,
            dropout=0.3,
            tolerance=0.05,
            verbosity=0,
        )

        # Should call get_endpoint_cnfs_with_resolution once
        assert mock_endpoints.call_count == 1

        assert result["xi"] == 1.5
        assert result["delta"] == 12
        assert result["min_distance"] == 0.8
        assert result["found"] is True

    @patch('cnf.navigation.astar.iterative.search.search._binary_search_min_distance')
    @patch('cnf.navigation.astar.iterative.search.search.get_endpoint_cnfs_with_resolution')
    def test_returns_not_found_when_binary_search_fails(self, mock_endpoints, mock_binary):
        """Should mark as not found when binary search returns None."""
        from cnf.navigation.astar.iterative.search.search import _search_at_resolution

        mock_cnf = MagicMock()
        mock_cnf.elements = ("Zr",)
        mock_endpoints.return_value = ([mock_cnf], [mock_cnf], 10)
        mock_binary.return_value = (None, None)

        start_uc = MockUnitCell()
        end_uc = MockUnitCell()

        result = _search_at_resolution(
            start_uc=start_uc,
            end_uc=end_uc,
            xi=1.5,
            atom_step_length=0.3,
            min_dist_low=0.5,
            min_dist_high=1.5,
            max_iterations=1000,
            beam_width=500,
            dropout=0.3,
            tolerance=0.05,
            verbosity=0,
        )

        assert result["found"] is False
        assert result["min_distance"] is None


class TestSearch:
    @patch('cnf.navigation.astar.iterative.search.search._search_at_resolution')
    def test_iterates_through_resolutions(self, mock_search_res):
        """Should iterate through all xi/atom_step pairs."""
        from cnf.navigation.astar.iterative.search.search import search

        mock_search_res.return_value = {
            "xi": 1.5,
            "delta": 10,
            "atom_step_length": 0.4,
            "min_distance": 0.8,
            "found": True,
            "search_result": MagicMock(),
        }

        start_uc = MockUnitCell()
        end_uc = MockUnitCell()

        result = search(
            start_uc=start_uc,
            end_uc=end_uc,
            xi_values=[1.5, 1.0],
            atom_step_lengths=[0.4, 0.2],
            n_workers=1,
            verbosity=0,
        )

        # Should call _search_at_resolution for each resolution
        assert mock_search_res.call_count == 2

    @patch('cnf.navigation.astar.iterative.search.search._search_at_resolution')
    def test_raises_on_mismatched_lengths(self, mock_search_res):
        """Should raise ValueError if xi_values and atom_step_lengths differ in length."""
        from cnf.navigation.astar.iterative.search.search import search

        start_uc = MockUnitCell()
        end_uc = MockUnitCell()

        with pytest.raises(ValueError, match="must have the same length"):
            search(
                start_uc=start_uc,
                end_uc=end_uc,
                xi_values=[1.5, 1.0, 0.75],
                atom_step_lengths=[0.4, 0.2],
                n_workers=1,
                verbosity=0,
            )

    @patch('cnf.navigation.astar.iterative.search.search._search_at_resolution')
    def test_recommends_finest_successful_resolution(self, mock_search_res):
        """Should recommend the finest (last) successful resolution."""
        from cnf.navigation.astar.iterative.search.search import search

        # First resolution succeeds, second fails, third succeeds
        mock_search_res.side_effect = [
            {"xi": 1.5, "delta": 10, "atom_step_length": 0.4, "min_distance": 0.8,
             "found": True, "search_result": MagicMock()},
            {"xi": 1.25, "delta": 12, "atom_step_length": 0.3, "min_distance": None,
             "found": False, "search_result": MagicMock()},
            {"xi": 1.0, "delta": 15, "atom_step_length": 0.2, "min_distance": 0.6,
             "found": True, "search_result": MagicMock()},
        ]

        start_uc = MockUnitCell()
        end_uc = MockUnitCell()

        result = search(
            start_uc=start_uc,
            end_uc=end_uc,
            xi_values=[1.5, 1.25, 1.0],
            atom_step_lengths=[0.4, 0.3, 0.2],
            n_workers=1,
            verbosity=0,
        )

        # Should recommend the finest successful: xi=1.0, delta=15
        assert result.recommended_xi == 1.0
        assert result.recommended_delta == 15
        assert result.recommended_min_distance == 0.6
        assert len(result.successful_params) == 2

    @patch('cnf.navigation.astar.iterative.search.search._search_at_resolution')
    def test_returns_partial_results_on_all_failures(self, mock_search_res):
        """Should return partial results if no resolution succeeds."""
        from cnf.navigation.astar.iterative.search.search import search

        mock_search_res.return_value = {
            "xi": 1.5,
            "delta": 10,
            "atom_step_length": 0.4,
            "min_distance": None,
            "found": False,
            "search_result": MagicMock(),
        }

        start_uc = MockUnitCell()
        end_uc = MockUnitCell()

        result = search(
            start_uc=start_uc,
            end_uc=end_uc,
            xi_values=[1.5],
            atom_step_lengths=[0.4],
            n_workers=1,
            verbosity=0,
        )

        assert result.success is False
        assert result.recommended_xi is None
        assert len(result.successful_params) == 0
        assert len(result.results) == 1

    @patch('cnf.navigation.astar.iterative.search.search._search_at_resolution')
    def test_uses_default_values(self, mock_search_res):
        """Should use default xi_values and atom_step_lengths if not provided."""
        from cnf.navigation.astar.iterative.search.search import search, _DEFAULT_XI_VALUES, _DEFAULT_ATOM_STEP_LENGTHS

        mock_search_res.return_value = {
            "xi": 1.5,
            "delta": 10,
            "atom_step_length": 0.4,
            "min_distance": 0.8,
            "found": True,
            "search_result": MagicMock(),
        }

        start_uc = MockUnitCell()
        end_uc = MockUnitCell()

        result = search(
            start_uc=start_uc,
            end_uc=end_uc,
            n_workers=1,
            verbosity=0,
        )

        # Should iterate through all defaults
        assert mock_search_res.call_count == len(_DEFAULT_XI_VALUES)

    @patch('cnf.navigation.astar.iterative.search.search._search_at_resolution')
    def test_writes_output_file(self, mock_search_res):
        """Should write results to output_dir if provided."""
        from cnf.navigation.astar.iterative.search.search import search

        mock_search_res.return_value = {
            "xi": 1.5,
            "delta": 10,
            "atom_step_length": 0.4,
            "min_distance": 0.8,
            "found": True,
            "search_result": MagicMock(to_dict=lambda: {}),
        }

        start_uc = MockUnitCell()
        end_uc = MockUnitCell()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = search(
                start_uc=start_uc,
                end_uc=end_uc,
                xi_values=[1.5],
                atom_step_lengths=[0.4],
                n_workers=1,
                verbosity=0,
                output_dir=tmpdir,
            )

            output_file = Path(tmpdir) / "parameter_search_result.json"
            assert output_file.exists()
