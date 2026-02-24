"""Integration tests for iterative A* barrier search algorithms.

Uses GraceCalculator (GRACE-FS-OAM model) for realistic energy evaluations
with small unit cells at low resolution for reasonable test speed.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

import helpers
from cnf import CrystalNormalForm, UnitCell
from cnf.calculation.grace import GraceCalculator
from cnf.navigation import find_neighbors
from cnf.navigation.astar.iterative import (
    iterative_astar_barrier,
    ceiling_barrier_search,
)
from cnf.navigation.astar.iterative._energy import path_barrier


@pytest.fixture(scope="module")
def grace_calc():
    """Shared GraceCalculator instance for all tests in module."""
    return GraceCalculator()


@pytest.fixture
def zr_hcp_struct():
    """Load Zr HCP structure."""
    return helpers.load_specific_cif("Zr_HCP.cif")


@pytest.fixture
def zr_bcc_struct():
    """Load Zr BCC structure."""
    return helpers.load_specific_cif("Zr_BCC.cif")


@pytest.fixture
def output_dir():
    """Create a temporary output directory for test artifacts."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestIterativeAstarBarrier:
    """Integration tests for iterative_astar_barrier algorithm."""

    def test_finds_path_between_nearby_points(self, zr_hcp_struct, grace_calc):
        """Should find a path between a structure and its neighbor."""
        xi = 1.5
        delta = 10

        start_cnf = CrystalNormalForm.from_pmg_struct(zr_hcp_struct, xi, delta)
        neighbors = find_neighbors(start_cnf)
        goal_cnf = neighbors[0]

        barrier, path_cnfs, path_energies = iterative_astar_barrier(
            start_cnfs=[start_cnf],
            goal_cnfs=[goal_cnf],
            energy_calc=grace_calc,
            paths_per_round=3,
            max_rounds=2,
            dropout=0.3,
            max_iterations_per_path=1000,
            beam_width=100,
            verbose=False,
        )

        assert barrier is not None, "Should find a path"
        assert path_cnfs is not None
        assert len(path_cnfs) >= 2, "Path should have at least start and goal"
        assert path_energies is not None
        assert len(path_energies) == len(path_cnfs)

    def test_path_is_valid_neighbor_chain(self, zr_hcp_struct, grace_calc):
        """Each step in returned path should be a valid neighbor transition."""
        xi = 1.5
        delta = 10

        start_cnf = CrystalNormalForm.from_pmg_struct(zr_hcp_struct, xi, delta)
        neighbors = find_neighbors(start_cnf)
        # Go to a neighbor's neighbor for a slightly longer path
        ring2 = set()
        for nb in neighbors[:3]:
            ring2.update(find_neighbors(nb))
        ring2 = ring2 - set(neighbors) - {start_cnf}
        goal_cnf = list(ring2)[0]

        barrier, path_cnfs, path_energies = iterative_astar_barrier(
            start_cnfs=[start_cnf],
            goal_cnfs=[goal_cnf],
            energy_calc=grace_calc,
            paths_per_round=5,
            max_rounds=3,
            dropout=0.3,
            max_iterations_per_path=2000,
            beam_width=200,
            verbose=False,
        )

        assert path_cnfs is not None, "Should find a path"

        # Verify each consecutive pair is a valid neighbor relationship
        for i in range(len(path_cnfs) - 1):
            current = path_cnfs[i]
            next_cnf = path_cnfs[i + 1]
            current_neighbors = find_neighbors(current)
            neighbor_coords = {nb.coords for nb in current_neighbors}
            assert next_cnf.coords in neighbor_coords, (
                f"Step {i} -> {i+1} is not a valid neighbor transition"
            )

    def test_barrier_equals_max_path_energy(self, zr_hcp_struct, grace_calc):
        """Barrier should equal the maximum energy along the path."""
        xi = 1.5
        delta = 10

        start_cnf = CrystalNormalForm.from_pmg_struct(zr_hcp_struct, xi, delta)
        neighbors = find_neighbors(start_cnf)
        goal_cnf = neighbors[0]

        barrier, path_cnfs, path_energies = iterative_astar_barrier(
            start_cnfs=[start_cnf],
            goal_cnfs=[goal_cnf],
            energy_calc=grace_calc,
            paths_per_round=3,
            max_rounds=2,
            dropout=0.3,
            max_iterations_per_path=1000,
            beam_width=100,
            verbose=False,
        )

        assert path_energies is not None
        expected_barrier = max(path_energies)
        assert barrier == pytest.approx(expected_barrier), (
            f"Barrier {barrier} should equal max energy {expected_barrier}"
        )

    def test_endpoints_in_path(self, zr_hcp_struct, grace_calc):
        """Path should start at start_cnf and end at goal_cnf."""
        xi = 1.5
        delta = 10

        start_cnf = CrystalNormalForm.from_pmg_struct(zr_hcp_struct, xi, delta)
        neighbors = find_neighbors(start_cnf)
        goal_cnf = neighbors[0]

        barrier, path_cnfs, path_energies = iterative_astar_barrier(
            start_cnfs=[start_cnf],
            goal_cnfs=[goal_cnf],
            energy_calc=grace_calc,
            paths_per_round=3,
            max_rounds=2,
            dropout=0.3,
            max_iterations_per_path=1000,
            beam_width=100,
            verbose=False,
        )

        assert path_cnfs is not None
        assert path_cnfs[0].coords == start_cnf.coords, "Path should start at start_cnf"
        assert path_cnfs[-1].coords == goal_cnf.coords, "Path should end at goal_cnf"

    def test_writes_output_files(self, zr_hcp_struct, grace_calc, output_dir):
        """Should write round JSON files and manifest when output_dir provided."""
        xi = 1.5
        delta = 10

        start_cnf = CrystalNormalForm.from_pmg_struct(zr_hcp_struct, xi, delta)
        neighbors = find_neighbors(start_cnf)
        goal_cnf = neighbors[0]

        iterative_astar_barrier(
            start_cnfs=[start_cnf],
            goal_cnfs=[goal_cnf],
            energy_calc=grace_calc,
            paths_per_round=2,
            max_rounds=2,
            dropout=0.3,
            max_iterations_per_path=500,
            beam_width=100,
            verbose=False,
            output_dir=output_dir,
        )

        # Check that round files were created
        rounds_dir = output_dir / "rounds"
        assert rounds_dir.exists(), "Should create rounds directory"
        round_files = list(rounds_dir.glob("round_*.json"))
        assert len(round_files) >= 1, "Should write at least one round file"

    def test_respects_initial_ceiling(self, zr_hcp_struct, grace_calc):
        """When initial_ceiling provided, should skip round 0 and use it."""
        xi = 1.5
        delta = 10

        start_cnf = CrystalNormalForm.from_pmg_struct(zr_hcp_struct, xi, delta)
        neighbors = find_neighbors(start_cnf)
        goal_cnf = neighbors[0]

        # Get endpoint energies to set a reasonable ceiling
        start_energy = grace_calc.calculate_energy(start_cnf)
        goal_energy = grace_calc.calculate_energy(goal_cnf)
        initial_ceiling = max(start_energy, goal_energy) + 1.0  # Slightly above endpoints

        barrier, path_cnfs, path_energies = iterative_astar_barrier(
            start_cnfs=[start_cnf],
            goal_cnfs=[goal_cnf],
            energy_calc=grace_calc,
            paths_per_round=3,
            max_rounds=2,
            dropout=0.3,
            max_iterations_per_path=1000,
            beam_width=100,
            verbose=False,
            initial_ceiling=initial_ceiling,
        )

        # Should still find a path (ceiling is above endpoints)
        assert path_cnfs is not None, "Should find path with reasonable ceiling"
        # Barrier should be at or below the initial ceiling
        assert barrier <= initial_ceiling + 0.1, (
            f"Barrier {barrier} should be near or below initial ceiling {initial_ceiling}"
        )


class TestIterativeAstarBarrierEdgeCases:
    """Edge case tests for iterative_astar_barrier."""

    def test_same_start_and_goal(self, zr_hcp_struct, grace_calc):
        """When start equals goal, should return trivial path."""
        xi = 1.5
        delta = 10

        cnf = CrystalNormalForm.from_pmg_struct(zr_hcp_struct, xi, delta)

        barrier, path_cnfs, path_energies = iterative_astar_barrier(
            start_cnfs=[cnf],
            goal_cnfs=[cnf],
            energy_calc=grace_calc,
            paths_per_round=2,
            max_rounds=1,
            dropout=0.3,
            max_iterations_per_path=100,
            beam_width=50,
            verbose=False,
        )

        assert path_cnfs is not None
        assert len(path_cnfs) == 1, "Trivial path should have length 1"
        assert path_cnfs[0].coords == cnf.coords

    def test_multiple_start_cnfs(self, zr_hcp_struct, grace_calc):
        """Should work with multiple starting points."""
        xi = 1.5
        delta = 10

        start_cnf = CrystalNormalForm.from_pmg_struct(zr_hcp_struct, xi, delta)
        neighbors = find_neighbors(start_cnf)

        # Use start_cnf and its first neighbor as multiple starts
        start_cnfs = [start_cnf, neighbors[0]]
        # Goal is another neighbor
        goal_cnf = neighbors[1]

        barrier, path_cnfs, path_energies = iterative_astar_barrier(
            start_cnfs=start_cnfs,
            goal_cnfs=[goal_cnf],
            energy_calc=grace_calc,
            paths_per_round=3,
            max_rounds=2,
            dropout=0.3,
            max_iterations_per_path=1000,
            beam_width=100,
            verbose=False,
        )

        assert path_cnfs is not None, "Should find path from one of the starts"
        # Path should start from one of the start_cnfs
        start_coords = {cnf.coords for cnf in start_cnfs}
        assert path_cnfs[0].coords in start_coords


class TestCeilingBarrierSearch:
    """Integration tests for ceiling_barrier_search algorithm."""

    def test_finds_path_between_unit_cells(self, zr_hcp_struct, grace_calc, output_dir):
        """Should find a path between two unit cells with ceiling sweep."""
        start_uc = UnitCell.from_pymatgen_structure(zr_hcp_struct)

        # Create goal by discretizing and taking a neighbor
        xi = 1.5
        delta = 10
        start_cnf = CrystalNormalForm.from_pmg_struct(zr_hcp_struct, xi, delta)
        neighbors = find_neighbors(start_cnf)
        goal_cnf = neighbors[0]
        goal_uc = UnitCell.from_cnf(goal_cnf)

        barrier, path_cnfs, path_energies = ceiling_barrier_search(
            start_uc=start_uc,
            end_uc=goal_uc,
            energy_calc=grace_calc,
            xi=1.5,
            delta=10,
            step_per_atom=0.5,
            num_ceilings=3,
            attempts_per_ceiling=1,
            max_passes=1,
            max_sweep_rounds=3,
            dropout=0.2,
            beam_width=100,
            n_workers=1,  # Sequential for test simplicity
            verbose=False,
            output_dir=output_dir,
        )

        assert barrier is not None, "Should find a path"
        assert path_cnfs is not None
        assert len(path_cnfs) >= 2
        assert path_energies is not None
        assert len(path_energies) == len(path_cnfs)

        # Check manifest was written
        manifest = output_dir / "manifest.json"
        assert manifest.exists(), "Should write manifest.json"

    def test_path_energies_below_barrier(self, zr_hcp_struct, grace_calc):
        """All path energies should be at or below the reported barrier."""
        start_uc = UnitCell.from_pymatgen_structure(zr_hcp_struct)

        xi = 1.5
        delta = 10
        start_cnf = CrystalNormalForm.from_pmg_struct(zr_hcp_struct, xi, delta)
        neighbors = find_neighbors(start_cnf)
        goal_uc = UnitCell.from_cnf(neighbors[0])

        barrier, path_cnfs, path_energies = ceiling_barrier_search(
            start_uc=start_uc,
            end_uc=goal_uc,
            energy_calc=grace_calc,
            xi=1.5,
            delta=10,
            step_per_atom=0.5,
            num_ceilings=3,
            attempts_per_ceiling=1,
            max_passes=1,
            max_sweep_rounds=3,
            dropout=0.2,
            beam_width=100,
            n_workers=1,
            verbose=False,
        )

        assert path_energies is not None
        for i, energy in enumerate(path_energies):
            assert energy <= barrier + 1e-6, (
                f"Energy at step {i} ({energy}) exceeds barrier ({barrier})"
            )

    def test_with_max_ceiling_set(self, zr_hcp_struct, grace_calc):
        """Should respect max_ceiling parameter for sweep range."""
        start_uc = UnitCell.from_pymatgen_structure(zr_hcp_struct)

        xi = 1.5
        delta = 10
        start_cnf = CrystalNormalForm.from_pmg_struct(zr_hcp_struct, xi, delta)
        neighbors = find_neighbors(start_cnf)
        goal_uc = UnitCell.from_cnf(neighbors[0])

        # Compute a reasonable max_ceiling
        start_energy = grace_calc.calculate_energy(start_cnf)
        goal_energy = grace_calc.calculate_energy(neighbors[0])
        max_ceiling = max(start_energy, goal_energy) + 2.0

        barrier, path_cnfs, path_energies = ceiling_barrier_search(
            start_uc=start_uc,
            end_uc=goal_uc,
            energy_calc=grace_calc,
            xi=1.5,
            delta=10,
            max_ceiling=max_ceiling,
            num_ceilings=3,
            attempts_per_ceiling=1,
            max_passes=1,
            dropout=0.2,
            beam_width=100,
            n_workers=1,
            verbose=False,
        )

        assert path_cnfs is not None, "Should find path with explicit max_ceiling"
        assert barrier <= max_ceiling, (
            f"Barrier {barrier} should be at or below max_ceiling {max_ceiling}"
        )
