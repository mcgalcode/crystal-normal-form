import pytest
import numpy as np

import helpers

from cnf import CrystalNormalForm
from cnf.navigation import NeighborFinder
from cnf.navigation.endpoints import get_endpoint_cnfs
from cnf.navigation.astar.heuristics import (
    manhattan_distance,
    UnimodularManhattanHeuristic,
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
