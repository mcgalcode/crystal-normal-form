"""Tests for incremental saving during sweep."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pymatgen.core import Structure

from cnf import UnitCell
from cnf.calculation.constant_calculator import ConstantCalcProvider
from cnf.navigation.astar.iterative import sweep
from cnf.navigation.astar.iterative._batch import run_batch


class MockCNF:
    """Mock CNF for unit tests."""
    def __init__(self, coords=(1, 2, 3, 4, 5, 6, 7)):
        self.coords = coords
        self.elements = ("Zr",)
        self.xi = 1.5
        self.delta = 10


class TestRunBatchOnResult:
    """Unit tests for run_batch on_result callback."""

    @patch('cnf.navigation.astar.iterative._batch.search_ceiling_with_attempts')
    def test_on_result_called_for_each_ceiling(self, mock_search):
        """on_result should be called once per ceiling."""
        mock_search.side_effect = [
            {"ceiling": -10.0, "found": False, "iterations": 100},
            {"ceiling": -9.0, "found": False, "iterations": 100},
            {"ceiling": -8.0, "found": False, "iterations": 100},
        ]

        callback_results = []

        run_batch(
            ceilings=[-10.0, -9.0, -8.0],
            start_cnfs=[MockCNF()],
            goal_cnfs=[MockCNF()],
            elements=("Zr",),
            xi=1.5,
            delta=10,
            calc=MagicMock(),
            cache={},
            dropout=0.1,
            max_iters=100,
            beam_width=50,
            n_workers=1,
            pool=None,
            verbosity=0,
            on_result=lambda r: callback_results.append(r),
        )

        assert len(callback_results) == 3
        assert [r["ceiling"] for r in callback_results] == [-10.0, -9.0, -8.0]

    @patch('cnf.navigation.astar.iterative._batch.search_ceiling_with_attempts')
    def test_on_result_called_before_early_exit(self, mock_search):
        """on_result should be called even when search succeeds and exits early."""
        mock_search.side_effect = [
            {"ceiling": -10.0, "found": False, "iterations": 100},
            {"ceiling": -9.0, "found": True, "iterations": 50,
             "barrier": -9.0, "path": [(1,), (2,)], "energies": [-10.0, -9.0],
             "path_length": 2},
            # This one shouldn't be reached due to early exit
            {"ceiling": -8.0, "found": False, "iterations": 100},
        ]

        callback_results = []

        run_batch(
            ceilings=[-10.0, -9.0, -8.0],
            start_cnfs=[MockCNF()],
            goal_cnfs=[MockCNF()],
            elements=("Zr",),
            xi=1.5,
            delta=10,
            calc=MagicMock(),
            cache={},
            dropout=0.1,
            max_iters=100,
            beam_width=50,
            n_workers=1,
            pool=None,
            verbosity=0,
            on_result=lambda r: callback_results.append(r),
        )

        # Should have 2 results: the failure and the success (before early exit)
        assert len(callback_results) == 2
        assert callback_results[0]["found"] is False
        assert callback_results[1]["found"] is True

    @patch('cnf.navigation.astar.iterative._batch.search_ceiling_with_attempts')
    def test_on_result_none_does_not_error(self, mock_search):
        """Should not error when on_result is None."""
        mock_search.return_value = {"ceiling": -10.0, "found": False, "iterations": 100}

        # Should not raise
        run_batch(
            ceilings=[-10.0],
            start_cnfs=[MockCNF()],
            goal_cnfs=[MockCNF()],
            elements=("Zr",),
            xi=1.5,
            delta=10,
            calc=MagicMock(),
            cache={},
            dropout=0.1,
            max_iters=100,
            beam_width=50,
            n_workers=1,
            pool=None,
            verbosity=0,
            on_result=None,
        )


@pytest.fixture
def zr_bcc():
    """Load Zr BCC structure."""
    cif_path = Path(__file__).parents[4] / "data" / "specific_cifs" / "Zr_BCC.cif"
    return UnitCell.from_pymatgen_structure(Structure.from_file(str(cif_path)))


@pytest.fixture
def zr_hcp():
    """Load Zr HCP structure."""
    cif_path = Path(__file__).parents[4] / "data" / "specific_cifs" / "Zr_HCP.cif"
    return UnitCell.from_pymatgen_structure(Structure.from_file(str(cif_path)))


class TestSweepIncrementalSave:
    """Tests that sweep saves results incrementally."""

    def test_saves_result_file(self, zr_bcc, zr_hcp):
        """Sweep should create a result file in output_dir."""
        calc_provider = ConstantCalcProvider(-10.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            sweep(
                start_uc=zr_bcc,
                end_uc=zr_hcp,
                max_ceiling=-5.0,
                calc_provider=calc_provider,
                xi=2.0,
                delta=5,
                num_ceilings=2,
                max_passes=1,
                max_iterations=100,
                beam_width=50,
                n_workers=1,
                verbosity=0,
                output_dir=str(output_path),
            )

            result_file = output_path / "ceiling_sweep_result.json"
            assert result_file.exists(), "Result file should be created"

    def test_result_file_has_results_per_ceiling(self, zr_bcc, zr_hcp):
        """Result file should contain one SearchResult per ceiling searched."""
        calc_provider = ConstantCalcProvider(-10.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            sweep(
                start_uc=zr_bcc,
                end_uc=zr_hcp,
                max_ceiling=-5.0,
                calc_provider=calc_provider,
                xi=2.0,
                delta=5,
                num_ceilings=3,
                max_passes=1,
                max_iterations=100,
                beam_width=50,
                n_workers=1,
                verbosity=0,
                output_dir=str(output_path),
            )

            result_file = output_path / "ceiling_sweep_result.json"
            with open(result_file) as f:
                saved = json.load(f)

            assert "results" in saved
            # Each ceiling produces a SearchResult
            assert len(saved["results"]) >= 1

            # Verify results have expected structure
            for r in saved["results"]:
                assert "context" in r
                assert "parameters" in r
                assert "attempts" in r
                assert "metadata" in r
                assert "ceiling" in r["metadata"]
