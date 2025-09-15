import pytest
import numpy as np

from cnf.lattice import Superbasis, VonormList
from cnf.lattice.stabilizer import search_for_stabilizers

from pymatgen.core.lattice import Lattice

def test_basic_stabilizer():
    superbasis_vecs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    sb = Superbasis.from_generating_vecs(superbasis_vecs)
    vonorms = sb.compute_vonorms()
    stabilizer_permutations, explored_permutations = search_for_stabilizers(vonorms)
    print(f"Explored {len(explored_permutations)} permutations")
    for p in sorted(explored_permutations):
        print(p)
    print(f"Identified {len(stabilizer_permutations)} equivalent permutations")

def test_low_symmetry_stabilizer():
    superbasis_vecs = Lattice.from_parameters(1.0, 1.2, 1.33, 35, 80, 110).matrix
    sb = Superbasis.from_generating_vecs(superbasis_vecs)
    vonorms = sb.compute_vonorms()
    stabilizer_permutations, explored_permutations = search_for_stabilizers(vonorms)
    print(f"Explored {len(explored_permutations)} permutations")
    for p in sorted(explored_permutations):
        print(p)
    print(f"Identified {len(stabilizer_permutations)} equivalent permutations")

def test_medium_sym_stabilizer():
    superbasis_vecs = Lattice.hexagonal(1.2, 2.3).matrix
    sb = Superbasis.from_generating_vecs(superbasis_vecs)
    vonorms = sb.compute_vonorms()
    stabilizer_permutations, explored_permutations = search_for_stabilizers(vonorms)
    print(f"Explored {len(explored_permutations)} permutations")
    for p in sorted(explored_permutations):
        print(p)
    print(f"Identified {len(stabilizer_permutations)} equivalent permutations")
    print(stabilizer_permutations)