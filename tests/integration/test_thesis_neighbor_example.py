import pytest

from cnf import CrystalNormalForm, LatticeNormalForm, BasisNormalForm
from cnf.navigation.lattice_neighbor_finder import LatticeNeighborFinder
from cnf.navigation.basis_neighbor_finder import BasisNeighborFinder

LATTICE_NEIGHBORS = [
    (6, 6, 15, 16, 4, 18, 21, 2, 10, 10),
    (6, 6, 15, 16, 4, 19, 20, 10, 2, 10),
    (6, 6, 15, 16, 3, 19, 21, 2, 10, 10),
    (6, 6, 15, 16, 5, 19, 19, 2, 10, 10),
    (6, 6, 15, 16, 3, 20, 20, 2, 10, 10),
    (6, 6, 15, 16, 5, 18, 20, 2, 10, 10),
    (6, 6, 15, 15, 4, 19, 19, 0, 8, 10),
    (6, 6, 15, 17, 4, 19, 21, 2, 10, 10),
    (6, 6, 15, 15, 4, 18, 20, 0, 8, 10),
    (6, 6, 15, 17, 4, 20, 20, 2, 10, 10),
    (6, 6, 15, 15, 3, 19, 20, 0, 8, 10),
    (6, 6, 15, 17, 5, 19, 20, 2, 10, 10),
    (6, 6, 14, 16, 4, 19, 19, 2, 10, 10),
    (6, 6, 16, 16, 4, 19, 21, 0, 8, 10),
    (6, 6, 14, 16, 4, 18, 20, 2, 10, 10),
    (6, 6, 16, 16, 4, 20, 20, 0, 8, 10),
    (6, 6, 14, 16, 3, 19, 20, 2, 10, 10),
    (6, 6, 16, 16, 5, 19, 20, 0, 8, 10),
    (6, 6, 14, 17, 4, 19, 20, 2, 10, 10),
    (6, 6, 15, 16, 4, 19, 20, 0, 8, 10),
    (5, 6, 15, 16, 4, 19, 19, 10, 2, 10),
    (6, 7, 15, 16, 4, 19, 21, 2, 10, 10),
    (5, 6, 15, 16, 4, 20, 18, 10, 2, 10),
    (6, 7, 15, 16, 4, 20, 20, 2, 10, 10),
    (5, 6, 15, 16, 3, 20, 19, 10, 2, 10),
    (6, 7, 15, 16, 5, 19, 20, 2, 10, 10),
    (5, 6, 15, 17, 4, 20, 19, 10, 2, 10),
    (6, 7, 15, 15, 4, 19, 20, 2, 10, 10),
    (5, 6, 16, 16, 4, 19, 20, 0, 8, 10),
    (6, 7, 14, 16, 4, 19, 20, 2, 10, 10),
    (5, 6, 15, 16, 4, 19, 19, 2, 10, 10),
    (6, 7, 15, 16, 4, 21, 19, 10, 2, 10),
    (5, 6, 15, 16, 4, 18, 20, 2, 10, 10),
    (6, 7, 15, 16, 4, 20, 20, 10, 2, 10),
    (5, 6, 15, 16, 3, 19, 20, 2, 10, 10),
    (6, 7, 15, 16, 5, 20, 19, 10, 2, 10),
    (5, 6, 15, 17, 4, 19, 20, 2, 10, 10),
    (6, 7, 15, 15, 4, 19, 20, 0, 8, 10),
    (5, 6, 16, 16, 4, 19, 20, 2, 10, 10),
    (6, 7, 14, 16, 4, 20, 19, 10, 2, 10),
    (5, 7, 15, 16, 4, 19, 20, 2, 10, 10),
    (5, 7, 15, 16, 4, 20, 19, 10, 2, 10)
]

BASIS_NEIGHBORS = [
    (6, 6, 15, 16, 4, 19, 20, 1, 9, 9),
    (6, 6, 15, 16, 4, 19, 20, 3, 11, 11),
    (6, 6, 15, 16, 4, 19, 20, 2, 10, 9),
    (6, 6, 15, 16, 4, 19, 20, 2, 10, 11),
    (6, 6, 15, 16, 4, 19, 20, 2, 9, 10),
    (6, 6, 15, 16, 4, 19, 20, 2, 11, 10),
    (6, 6, 15, 16, 4, 19, 20, 1, 10, 10),
    (6, 6, 15, 16, 4, 19, 20, 3, 10, 10),
]

def test_thesis_neighbors():
    xi = 1.5
    delta = 20

    starting_lnf = LatticeNormalForm.from_coords((6, 6, 15, 16, 4, 19, 20), xi)
    starting_bnf = BasisNormalForm((2, 10, 10), ["Li", "Li"], delta)
    starting_cnf = CrystalNormalForm(starting_lnf, starting_bnf)

    lnf_neighbor_finder = LatticeNeighborFinder(starting_cnf)
    bnf_neighbor_finder = BasisNeighborFinder(starting_cnf)

    lnf_neighbor_set = lnf_neighbor_finder.find_cnf_neighbors()
    lnf_neighbors = lnf_neighbor_set.neighbors
    lnf_neighbors = [n.point for n in lnf_neighbors]

    bnf_neighbor_set = bnf_neighbor_finder.find_basis_neighbors()
    bnf_neighbors = bnf_neighbor_set.neighbors
    bnf_neighbors = [n.point for n in bnf_neighbors]

    neighbors = lnf_neighbors

    print(f"Found {len(set(neighbors))} distinct neighbors!")

    found = set(LATTICE_NEIGHBORS).intersection(set(neighbors))
    not_found = set(LATTICE_NEIGHBORS) - set(neighbors)

    print(f"Found {len(found)} of {len(LATTICE_NEIGHBORS)} expected CNF neighbors.")
    print(f"Missed {len(not_found)} of {len(LATTICE_NEIGHBORS)} expected CNF neighbors.")

    found_neighbor_lnf_coords = [tuple(n.lattice_normal_form.coords) for n in neighbors]
    LATTICE_NEIGHBOR_LNFS = set([tuple(n[:7]) for n in LATTICE_NEIGHBORS])

    found_lnfs = set(LATTICE_NEIGHBOR_LNFS).intersection(set(found_neighbor_lnf_coords))
    not_found_lnfs = set(LATTICE_NEIGHBOR_LNFS) - set(found_neighbor_lnf_coords)

    print(f"Found {len(found_lnfs)} of {len(LATTICE_NEIGHBOR_LNFS)} expected lnf neighbors.")
    print(f"Missed {len(not_found_lnfs)} of {len(LATTICE_NEIGHBOR_LNFS)} expected lnf neighbors.")

    distinct_neighbor_vonorm_sets = set([tuple(sorted(n.lattice_normal_form.vonorms.vonorms)) for n in neighbors])
    DISTINCT_LATTICE_NEIGHBOR_VONORMS = set([tuple(sorted(n)) for n in LATTICE_NEIGHBOR_LNFS])

    found_vonorm_sets = set(DISTINCT_LATTICE_NEIGHBOR_VONORMS).intersection(set(distinct_neighbor_vonorm_sets))
    not_found_vonorm_sets = set(DISTINCT_LATTICE_NEIGHBOR_VONORMS) - set(distinct_neighbor_vonorm_sets)

    print(f"Found {len(found_vonorm_sets)} of {len(DISTINCT_LATTICE_NEIGHBOR_VONORMS)} expected vonorm sets.")
    print(f"Missed {len(not_found_vonorm_sets)} of {len(DISTINCT_LATTICE_NEIGHBOR_VONORMS)} expected vonorm sets.")
