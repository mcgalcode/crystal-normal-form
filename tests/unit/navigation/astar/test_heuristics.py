import pytest
import numpy as np

import helpers

from cnf import CrystalNormalForm
from cnf.navigation import NeighborFinder
from cnf.navigation.endpoints import get_endpoint_cnfs
from cnf.navigation.astar.heuristics import (
    manhattan_distance,
    manhattan_dist_cnfs,
    squared_euclidean_heuristic,
    make_heuristic,
    UnimodularManhattanHeuristic,
    _precompute_goal_variants,
    _precompute_goal_variants_full,
)


def _min_unimodular_manhattan(cnf: CrystalNormalForm, neighbor: CrystalNormalForm) -> float:
    """Compute min manhattan distance from cnf to any unimodular variant of neighbor."""
    variants = _precompute_goal_variants_full(neighbor)
    current = np.array(cnf.coords, dtype=np.int64)
    dists = np.sum(np.abs(variants - current), axis=1)
    return int(np.min(dists))


# Use a few structures × discretizations so we cover different Voronoi classes.
STRUCTURES = [
    ("Zr_BCC.cif", 1.5, 10),
    ("Zr_HCP.cif", 1.5, 10),
    ("TiO2_anatase.cif", 1.0, 12),
    ("TiO2_rutile.cif", 1.0, 12),
]


@pytest.fixture(params=STRUCTURES, ids=[s[0] for s in STRUCTURES])
def cnf_with_neighbors(request):
    """Yield (cnf, lattice_neighbors, motif_neighbors) for each test structure."""
    cif_name, xi, delta = request.param
    struct = helpers.load_specific_cif(cif_name)

    # get_endpoint_cnfs with itself just converts to CNF at the right supercell
    cnfs, _ = get_endpoint_cnfs(struct, struct, xi=xi, delta=delta)
    cnf = cnfs[0]

    finder = NeighborFinder.from_cnf(cnf)
    lattice_nbrs = finder.find_lattice_neighbor_cnfs(cnf)
    motif_nbrs = finder.find_motif_neighbor_cnfs(cnf)

    return cnf, lattice_nbrs, motif_nbrs


def test_lattice_neighbors_within_2(cnf_with_neighbors):
    """Every lattice neighbor should be at most 2 unimodular-manhattan units away.

    A lattice step changes one conorm by ±1, which alters exactly 2 vonorms.
    If the unimodular variant search is comprehensive, the min-over-variants
    manhattan distance should be <= 2 for every lattice neighbor.
    """
    cnf, lattice_nbrs, _ = cnf_with_neighbors
    assert len(lattice_nbrs) > 0, "Expected at least one lattice neighbor"

    for nbr in lattice_nbrs:
        dist = _min_unimodular_manhattan(cnf, nbr)
        assert dist <= 2, (
            f"Lattice neighbor at unimodular manhattan distance {dist} > 2.\n"
            f"  node:     {cnf.coords}\n"
            f"  neighbor: {nbr.coords}"
        )


def test_motif_neighbors_within_3(cnf_with_neighbors):
    """Every motif neighbor should be at most 3 unimodular-manhattan units away.

    Motif neighbors include both single-coordinate steps (±1 in one coord)
    and uniform atom shifts (±1 in all 3 coords of one atom). The uniform
    shifts change 3 coordinates at once, so the bound is 3.
    """
    cnf, _, motif_nbrs = cnf_with_neighbors
    assert len(motif_nbrs) > 0, "Expected at least one motif neighbor"

    for nbr in motif_nbrs:
        dist = _min_unimodular_manhattan(cnf, nbr)
        assert dist <= 3, (
            f"Motif neighbor at unimodular manhattan distance {dist} > 3.\n"
            f"  node:     {cnf.coords}\n"
            f"  neighbor: {nbr.coords}"
        )


# -----------------------------------------------------------------------------
# Manhattan distance tests
# -----------------------------------------------------------------------------

class TestManhattanDistance:

    def test_zero_distance_to_self(self, cnf_with_neighbors):
        cnf, _, _ = cnf_with_neighbors
        dist = manhattan_distance(cnf.coords, [cnf])
        assert dist == 0

    def test_positive_distance_to_neighbor(self, cnf_with_neighbors):
        cnf, lattice_nbrs, _ = cnf_with_neighbors
        if len(lattice_nbrs) == 0:
            pytest.skip("No lattice neighbors")
        dist = manhattan_distance(cnf.coords, [lattice_nbrs[0]])
        assert dist > 0

    def test_returns_min_over_goals(self, cnf_with_neighbors):
        cnf, lattice_nbrs, _ = cnf_with_neighbors
        if len(lattice_nbrs) < 2:
            pytest.skip("Need at least 2 neighbors")

        dist_to_first = manhattan_distance(cnf.coords, [lattice_nbrs[0]])
        dist_to_both = manhattan_distance(cnf.coords, [lattice_nbrs[0], cnf])

        assert dist_to_both == 0  # cnf itself is a goal
        assert dist_to_first >= dist_to_both

    def test_distance_is_scaled_by_2(self, cnf_with_neighbors):
        cnf, lattice_nbrs, _ = cnf_with_neighbors
        if len(lattice_nbrs) == 0:
            pytest.skip("No lattice neighbors")

        goal = lattice_nbrs[0]
        raw_dist = np.sum(np.abs(np.array(cnf.coords) - np.array(goal.coords)))
        scaled_dist = manhattan_distance(cnf.coords, [goal])

        assert scaled_dist == raw_dist * 2


class TestManhattanDistCnfs:

    def test_symmetric(self, cnf_with_neighbors):
        cnf, lattice_nbrs, _ = cnf_with_neighbors
        if len(lattice_nbrs) == 0:
            pytest.skip("No lattice neighbors")

        nbr = lattice_nbrs[0]
        dist1 = manhattan_dist_cnfs(cnf, nbr)
        dist2 = manhattan_dist_cnfs(nbr, cnf)
        assert dist1 == dist2

    def test_zero_to_self(self, cnf_with_neighbors):
        cnf, _, _ = cnf_with_neighbors
        assert manhattan_dist_cnfs(cnf, cnf) == 0


# -----------------------------------------------------------------------------
# Squared Euclidean heuristic tests
# -----------------------------------------------------------------------------

class TestSquaredEuclideanHeuristic:

    def test_zero_distance_to_self(self, cnf_with_neighbors):
        cnf, _, _ = cnf_with_neighbors
        dist = squared_euclidean_heuristic(cnf.coords, [cnf])
        assert dist == 0

    def test_positive_distance_to_neighbor(self, cnf_with_neighbors):
        cnf, lattice_nbrs, _ = cnf_with_neighbors
        if len(lattice_nbrs) == 0:
            pytest.skip("No lattice neighbors")
        dist = squared_euclidean_heuristic(cnf.coords, [lattice_nbrs[0]])
        assert dist > 0

    def test_returns_min_over_goals(self, cnf_with_neighbors):
        cnf, lattice_nbrs, _ = cnf_with_neighbors
        if len(lattice_nbrs) < 2:
            pytest.skip("Need at least 2 neighbors")

        dist_to_both = squared_euclidean_heuristic(cnf.coords, [lattice_nbrs[0], cnf])
        assert dist_to_both == 0  # cnf itself is a goal


# -----------------------------------------------------------------------------
# make_heuristic factory tests
# -----------------------------------------------------------------------------

class TestMakeHeuristic:

    def test_manhattan_mode(self):
        h = make_heuristic("manhattan")
        assert h is manhattan_distance

    def test_unimodular_light_mode(self):
        h = make_heuristic("unimodular_light")
        assert isinstance(h, UnimodularManhattanHeuristic)
        assert h.full is False
        assert h.partial is False

    def test_unimodular_partial_mode(self):
        h = make_heuristic("unimodular_partial")
        assert isinstance(h, UnimodularManhattanHeuristic)
        assert h.full is False
        assert h.partial is True

    def test_unimodular_full_mode(self):
        h = make_heuristic("unimodular_full")
        assert isinstance(h, UnimodularManhattanHeuristic)
        assert h.full is True

    def test_weight_passed_to_unimodular(self):
        h = make_heuristic("unimodular_full", weight=0.75)
        assert h.weight == 0.75

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown heuristic mode"):
            make_heuristic("invalid_mode")


# -----------------------------------------------------------------------------
# UnimodularManhattanHeuristic class tests
# -----------------------------------------------------------------------------

class TestUnimodularManhattanHeuristic:

    def test_zero_distance_to_self(self, cnf_with_neighbors):
        cnf, _, _ = cnf_with_neighbors
        h = UnimodularManhattanHeuristic(weight=1.0)
        dist = h(cnf.coords, [cnf])
        assert dist == 0

    def test_caches_goal_variants(self, cnf_with_neighbors):
        cnf, lattice_nbrs, _ = cnf_with_neighbors
        h = UnimodularManhattanHeuristic(weight=1.0)

        goals = [cnf]
        h(cnf.coords, goals)
        assert h._goal_variants is not None
        assert h._goals_id == id(goals)

    def test_cache_invalidation_on_new_goals(self, cnf_with_neighbors):
        cnf, lattice_nbrs, _ = cnf_with_neighbors
        if len(lattice_nbrs) == 0:
            pytest.skip("No lattice neighbors")

        h = UnimodularManhattanHeuristic(weight=1.0)

        goals1 = [cnf]
        h(cnf.coords, goals1)
        id1 = h._goals_id

        goals2 = [lattice_nbrs[0]]
        h(cnf.coords, goals2)
        id2 = h._goals_id

        assert id1 != id2

    def test_weight_scales_result(self, cnf_with_neighbors):
        cnf, lattice_nbrs, _ = cnf_with_neighbors
        if len(lattice_nbrs) == 0:
            pytest.skip("No lattice neighbors")

        h1 = UnimodularManhattanHeuristic(weight=1.0)
        h2 = UnimodularManhattanHeuristic(weight=0.5)

        goals = [lattice_nbrs[0]]
        dist1 = h1(cnf.coords, goals)
        dist2 = h2(cnf.coords, goals)

        assert dist2 == pytest.approx(dist1 * 0.5)

    def test_full_produces_variants(self, cnf_with_neighbors):
        cnf, _, _ = cnf_with_neighbors

        variants_full = _precompute_goal_variants_full(cnf)

        # Should produce at least some variants
        assert len(variants_full) > 0


# -----------------------------------------------------------------------------
# Precompute variants tests
# -----------------------------------------------------------------------------

class TestPrecomputeGoalVariants:

    def test_variants_include_original(self, cnf_with_neighbors):
        cnf, _, _ = cnf_with_neighbors
        variants = _precompute_goal_variants(cnf)

        # The original coords should be among the variants
        original = np.array(cnf.coords, dtype=np.int64)
        matches = np.all(variants == original, axis=1)
        assert np.any(matches), "Original coords not found in variants"

    def test_variants_are_unique(self, cnf_with_neighbors):
        cnf, _, _ = cnf_with_neighbors
        variants = _precompute_goal_variants_full(cnf)

        unique = np.unique(variants, axis=0)
        assert len(unique) == len(variants), "Duplicate variants found"

    def test_partial_reduces_variant_count(self, cnf_with_neighbors):
        cnf, _, _ = cnf_with_neighbors

        variants_full = _precompute_goal_variants_full(cnf, partial=False)
        variants_partial = _precompute_goal_variants_full(cnf, partial=True)

        assert len(variants_partial) <= len(variants_full)
