"""Integration tests for A* pathfinding phases.

These tests exercise the full pipeline with real crystal structures
but use smaller parameters for faster execution.
"""

import pytest
import tempfile
from pathlib import Path

from cnf import UnitCell
from cnf.navigation.endpoints import get_endpoint_cnfs
from cnf.navigation.astar.models import ParameterSearchResult, SearchResult

import helpers


@pytest.fixture
def zr_bcc_uc(zr_bcc_mp):
    """Zr BCC unit cell from CIF."""
    return UnitCell.from_pymatgen_structure(zr_bcc_mp)


@pytest.fixture
def zr_hcp_uc(zr_hcp_mp):
    """Zr HCP unit cell from CIF."""
    return UnitCell.from_pymatgen_structure(zr_hcp_mp)


@pytest.fixture
def start_cnfs(zr_bcc_uc):
    """Pre-computed CNFs at coarse resolution for fast tests."""
    cnfs, _ = get_endpoint_cnfs(zr_bcc_uc, zr_bcc_uc, xi=2.0, delta=5)
    return cnfs


@pytest.fixture
def goal_cnfs(zr_hcp_uc):
    """Pre-computed CNFs at coarse resolution for fast tests."""
    _, cnfs = get_endpoint_cnfs(zr_hcp_uc, zr_hcp_uc, xi=2.0, delta=5)
    return cnfs


class TestPhase1Search:
    """Tests for Phase 1: Parameter search."""

    @pytest.mark.slow
    def test_search_finds_parameters(self, zr_bcc_uc, zr_hcp_uc):
        """Should find working parameters for Zr BCC-HCP transition."""
        from cnf.navigation.astar.iterative import search

        result = search(
            start_uc=zr_bcc_uc,
            end_uc=zr_hcp_uc,
            xi_values=[2.0, 1.5],
            atom_step_lengths=[0.5, 0.4],
            min_dist_low=0.3,
            min_dist_high=1.0,
            tolerance=0.1,
            max_iterations=1000,
            beam_width=500,
            n_workers=1,
            verbose=False,
        )

        assert isinstance(result, ParameterSearchResult)
        # At least the coarsest resolution should work
        assert len(result.results) == 2

    @pytest.mark.slow
    def test_search_writes_output(self, zr_bcc_uc, zr_hcp_uc):
        """Should write results to output directory."""
        from cnf.navigation.astar.iterative import search

        with tempfile.TemporaryDirectory() as tmpdir:
            result = search(
                start_uc=zr_bcc_uc,
                end_uc=zr_hcp_uc,
                xi_values=[2.0],
                atom_step_lengths=[0.5],
                max_iterations=500,
                n_workers=1,
                verbose=False,
                output_dir=tmpdir,
            )

            output_file = Path(tmpdir) / "parameter_search_result.json"
            assert output_file.exists()

            # Should be loadable
            loaded = ParameterSearchResult.from_json(str(output_file))
            assert loaded.metadata is not None


class TestPhase2Sample:
    """Tests for Phase 2: Path sampling."""

    @pytest.mark.slow
    def test_sample_finds_paths(self, start_cnfs, goal_cnfs):
        """Should find diverse paths through sampling."""
        from cnf.navigation.astar.iterative import sample
        from cnf.calculation.constant_calculator import ConstantCalculator

        result = sample(
            start_cnfs=start_cnfs,
            goal_cnfs=goal_cnfs,
            energy_calc=ConstantCalculator(val=-10.0),
            num_samples=3,
            dropout_range=(0.3, 0.5),
            max_iterations=500,
            beam_width=500,
            verbose=False,
        )

        assert isinstance(result, SearchResult)
        # At least some samples should find paths
        assert len(result.attempts) == 3

    @pytest.mark.slow
    def test_sample_with_min_distance(self, start_cnfs, goal_cnfs):
        """Should respect min_distance filter."""
        from cnf.navigation.astar.iterative import sample
        from cnf.calculation.constant_calculator import ConstantCalculator

        result = sample(
            start_cnfs=start_cnfs,
            goal_cnfs=goal_cnfs,
            energy_calc=ConstantCalculator(val=-10.0),
            num_samples=2,
            min_distance=0.5,
            max_iterations=500,
            verbose=False,
        )

        # Should run with min_distance filter applied
        assert result.parameters.min_distance == 0.5


class TestPhase3Sweep:
    """Tests for Phase 3: Ceiling sweep."""

    @pytest.mark.slow
    def test_sweep_with_single_ceiling(self, zr_bcc_uc, zr_hcp_uc):
        """Should sweep a single energy ceiling."""
        from cnf.navigation.astar.iterative import sweep
        from cnf.calculation.constant_calculator import ConstantCalculator

        result = sweep(
            start_uc=zr_bcc_uc,
            end_uc=zr_hcp_uc,
            max_ceiling=-5.0,
            energy_calc=ConstantCalculator(val=-10.0),
            xi=2.0,
            delta=5,
            num_ceilings=1,
            max_passes=1,
            beam_width=500,
            n_workers=1,
            verbose=False,
        )

        # Should return CeilingSweepResult
        from cnf.navigation.astar.models import CeilingSweepResult
        assert isinstance(result, CeilingSweepResult)


class TestPhase4Ratchet:
    """Tests for Phase 4: Ratchet refinement."""

    @pytest.mark.slow
    def test_ratchet_basic(self, start_cnfs, goal_cnfs):
        """Should run ratchet refinement."""
        from cnf.navigation.astar.iterative import ratchet
        from cnf.calculation.constant_calculator import ConstantCalculator

        result = ratchet(
            start_cnfs=start_cnfs,
            goal_cnfs=goal_cnfs,
            initial_ceiling=-5.0,
            energy_calc=ConstantCalculator(val=-10.0),
            max_rounds=2,
            max_iterations=500,
            beam_width=500,
            verbose=False,
        )

        from cnf.navigation.astar.models import RefinementResult
        assert isinstance(result, RefinementResult)


class TestEndToEnd:
    """End-to-end tests combining multiple phases."""

    @pytest.mark.slow
    def test_sample_then_sweep(self, start_cnfs, goal_cnfs, zr_bcc_uc, zr_hcp_uc):
        """Should use sample result to inform sweep ceiling."""
        from cnf.navigation.astar.iterative import sample, sweep
        from cnf.calculation.constant_calculator import ConstantCalculator

        calc = ConstantCalculator(val=-10.0)

        # Phase 2: Sample to find initial ceiling
        sample_result = sample(
            start_cnfs=start_cnfs,
            goal_cnfs=goal_cnfs,
            energy_calc=calc,
            num_samples=2,
            max_iterations=500,
            verbose=False,
        )

        # Use best barrier (or a fallback) as ceiling for sweep
        initial_ceiling = sample_result.best_barrier
        if initial_ceiling is None:
            initial_ceiling = -5.0

        # Phase 3: Sweep with discovered ceiling
        sweep_result = sweep(
            start_uc=zr_bcc_uc,
            end_uc=zr_hcp_uc,
            max_ceiling=initial_ceiling + 1.0,
            energy_calc=calc,
            xi=2.0,
            delta=5,
            num_ceilings=2,
            max_passes=1,
            beam_width=500,
            n_workers=1,
            verbose=False,
        )

        # Both phases should complete
        assert sample_result is not None
        assert sweep_result is not None


class TestDataModelSerialization:
    """Tests for data model round-trip serialization."""

    @pytest.mark.slow
    def test_search_result_round_trip(self, start_cnfs, goal_cnfs):
        """SearchResult should serialize and deserialize correctly."""
        from cnf.navigation.astar.iterative import sample
        from cnf.calculation.constant_calculator import ConstantCalculator

        result = sample(
            start_cnfs=start_cnfs,
            goal_cnfs=goal_cnfs,
            energy_calc=ConstantCalculator(val=-10.0),
            num_samples=1,
            max_iterations=500,
            verbose=False,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            result.to_json(path)
            loaded = SearchResult.from_json(path)

            assert loaded.context.xi == result.context.xi
            assert loaded.context.delta == result.context.delta
            assert len(loaded.attempts) == len(result.attempts)
        finally:
            Path(path).unlink()
