import pytest

from cnf.lattice.superbasis import Superbasis
from cnf.lattice.selling import SuperbasisSellingReducer
from cnf.lattice.permutations import is_permutation_set_closed
from pymatgen.core.lattice import Lattice
from cnf.lattice.voronoi import ConormListForm

@pytest.fixture
def v5_lattice():
    cuboid = Lattice.cubic(1.0)
    return cuboid

@pytest.fixture
def reduced_v5_superbasis(v5_lattice):
    sb = Superbasis.from_pymatgen_lattice(v5_lattice)    
    r = SuperbasisSellingReducer()
    sb: Superbasis = r.reduce(sb).reduced_object
    return sb

@pytest.fixture
def v4_lattice():
    hexagonal_prism = Lattice.hexagonal(1.0, 2.0)
    return hexagonal_prism

@pytest.fixture
def reduced_v4_superbasis(v4_lattice):    
    sb = Superbasis.from_pymatgen_lattice(v4_lattice)
    return SuperbasisSellingReducer().reduce(sb).reduced_object

@pytest.fixture
def v3_lattice():
    rhombic_dodecahedron = Lattice([
        [1, 1, 0],
        [1, -1, 0],
        [-1, 0, 1],
    ])
    return rhombic_dodecahedron

@pytest.fixture
def reduced_v3_superbasis(v3_lattice):
    sb = Superbasis.from_pymatgen_lattice(v3_lattice)
    return SuperbasisSellingReducer().reduce(sb).reduced_object

@pytest.fixture
def v2_lattice():
    hexarhombic_dodecahedron = Lattice([
        [1, 1, 0],
        [1, -1, 0],
        [-1, 0.2, 1],
    ])
    return hexarhombic_dodecahedron

@pytest.fixture
def reduced_v2_superbasis(v2_lattice):
    sb = Superbasis.from_pymatgen_lattice(v2_lattice)
    return SuperbasisSellingReducer().reduce(sb).reduced_object    

@pytest.fixture
def v1_lattice():
    truncated_octahedron = Lattice([
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1],
    ])
    return truncated_octahedron

@pytest.fixture
def reduced_v1_superbasis(v1_lattice):
    sb = Superbasis.from_pymatgen_lattice(v1_lattice)
    return SuperbasisSellingReducer().reduce(sb).reduced_object