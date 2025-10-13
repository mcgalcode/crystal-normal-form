import pytest
import numpy as np

from pymatgen.core.lattice import Lattice
from cnf.lattice.rounding import DiscretizedVonormComputer
from cnf.lattice.selling import VonormListSellingReducer
from cnf.lattice import Superbasis
from cnf.lattice.voronoi import VonormList
from cnf.sublattice.sublattice_generator import transform_lattice_vecs
from cnf.sublattice.gamma_matrices import GammaMatrixTuple

@pytest.fixture
def Zr_BCC_lattice():
    return Lattice.cubic(3.42)

def test_can_discretize_vonorms(Zr_BCC_lattice):
    reducer = VonormListSellingReducer()
    superbasis = Superbasis.from_pymatgen_lattice(Zr_BCC_lattice)
    vonorms = reducer.reduce(superbasis.compute_vonorms()).reduced_object
    computer = DiscretizedVonormComputer(1.5)
    rounded_vonorms = computer.find_closest_valid_vonorms(vonorms)

    assert (np.abs(vonorms.vonorms - (np.array(rounded_vonorms.vonorms) * 1.5)) < 1.5).all()

def test_pathological_case_1(mp_structures):
    PATHO_IDX = 119
    # These are two canonicalized vonorm lists
    xi = 1.1
    vonorms_1 = VonormList((np.float64(34.58861224423054), np.float64(34.58861506721064), np.float64(40.67657149762606), np.float64(40.90386871588968), np.float64(69.16390162496408), np.float64(40.796879456568746), np.float64(40.7968864434241)))
    vonorms_2 = VonormList((np.float64(34.588612244230525), np.float64(34.58861506721064), np.float64(40.67657149762607), np.float64(40.90386871588968), np.float64(69.16390162496404), np.float64(40.79687945656875), np.float64(40.79688644342413)))

    dvc = DiscretizedVonormComputer(xi, verbose_log=False)

    # assert dvc.uncorrected_discretized_vonorms(vonorms_1) == dvc.uncorrected_discretized_vonorms(vonorms_2) 
    print("Vonorms 1")
    dv1 = dvc.find_closest_valid_vonorms(vonorms_1)

    print("Vonorms 2")
    dv2 = dvc.find_closest_valid_vonorms(vonorms_2)

    assert dv1 == dv2