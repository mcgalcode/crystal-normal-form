import helpers
import pytest
import json
from cnf import CrystalNormalForm, UnitCell
from cnf.navigation.crystal_map import CrystalMap
from cnf.navigation.neighbor_finder import NeighborFinder

@pytest.fixture(scope='module')
def point_set():
    struct = helpers.ALL_MP_STRUCTURES(1)[0]
    xi = 1.5 
    delta = 10
    cnf = UnitCell.from_pymatgen_structure(struct).to_cnf(xi=xi, delta=delta)
    nf = NeighborFinder(cnf)
    nbs = nf.find_neighbors()
    return xi, delta, nbs

@pytest.fixture
def ti_o2_anatase():
    return helpers.load_specific_cif("TiO2_anatase.cif")

@pytest.fixture
def ti_o2_rutile():
    return helpers.load_specific_cif("TiO2_rutile.cif")

@pytest.fixture
def zr_hcp():
    return helpers.load_specific_cif("Zr_HCP.cif")

@pytest.fixture
def zr_bcc():
    return helpers.load_specific_cif("Zr_BCC.cif")

def test_instantiate_crystal_map(point_set):
    xi, delta, cnfs = point_set
    
    cnf = cnfs[0]
    cmap = CrystalMap.from_cnf(cnf)
    assert cmap.delta == delta
    assert cmap.xi == xi


    assert cnf in cmap
    point_id = cmap.get_point_id(cnf)
    recovered = cmap.get_point_by_id(point_id)
    assert recovered == cnf
    assert len(cmap) == 1

def test_can_add_and_remove_points(point_set):
    xi, delta, cnfs = point_set

    cmap = CrystalMap(xi, delta, cnfs[0].elements)

    cnf1 = cnfs[0]
    cnf2 = cnfs[1]

    assert cnf1 not in cmap
    assert cnf2 not in cmap

    cmap.add_point(cnf1)
    assert cnf1 in cmap
    assert cnf2 not in cmap

    cmap.remove_point(cnf2)
    assert cnf1 in cmap
    assert cnf2 not in cmap

    cmap.remove_point(cnf1)
    assert cnf1 not in cmap
    assert cnf2 not in cmap

    cmap.add_point(cnf1)
    cmap.add_point(cnf2)
    assert cnf1 in cmap
    assert cnf2 in cmap

def test_can_add_connections(point_set):
    xi, delta, cnfs = point_set
    cmap = CrystalMap(xi, delta, cnfs[0].elements)

    cnf1 = cnfs[0]
    cnf2 = cnfs[1]

    cmap.add_point(cnf1)
    cmap.add_point(cnf2)

    cmap.add_connection(cnf1, cnf2)
    assert cmap.connection_exists(cnf1, cnf2)

    cmap.remove_connection(cnf1, cnf2)
    assert not cmap.connection_exists(cnf1, cnf2)
    assert cnf1 in cmap
    assert cnf2 in cmap

    cmap.add_connection(cnf1, cnf2)
    assert cmap.connection_exists(cnf1, cnf2)
    cmap.remove_point(cnf1)
    with pytest.raises(ValueError) as pyexcp:
        cmap.connection_exists(cnf1, cnf2)
    
    assert "Tried to look for connections" in pyexcp.value.__repr__()

def test_adding_node_is_idempotent(point_set):
    xi, delta, cnfs = point_set
    cmap = CrystalMap(xi, delta, cnfs[0].elements)

    cnf1 = cnfs[0]
    
    assert isinstance(cmap.add_point(cnf1), int)
    assert cmap.add_point(cnf1) is None
    assert cmap.add_point(cnf1) is None
    assert cmap.add_point(cnf1) is None
    assert len(cmap) == 1

    cmap.remove_point(cnf1)
    assert isinstance(cmap.add_point(cnf1), int)
    assert cmap.add_point(cnf1) is None

def test_removing_node_is_idempotent(point_set):
    xi, delta, cnfs = point_set
    cmap = CrystalMap(xi, delta, cnfs[0].elements)

    cnf1 = cnfs[0]
    

    assert cmap.remove_point(cnf1) is None
    nid = cmap.add_point(cnf1)
    assert cmap.remove_point(cnf1) == nid
    assert cmap.remove_point(cnf1) is None

def test_adding_and_removing_connections_is_idempotent(point_set):
    xi, delta, cnfs = point_set
    cmap = CrystalMap(xi, delta, cnfs[0].elements)

    cnf1 = cnfs[0]
    cnf2 = cnfs[1]

    id1 = cmap.add_point(cnf1)
    id2 = cmap.add_point(cnf2)

    assert cmap.add_connection(cnf1, cnf2)
    assert not cmap.add_connection(cnf1, cnf2)
    assert not cmap.add_connection(cnf1, cnf2)
    assert cmap.remove_connection(cnf1, cnf2)
    assert not cmap.remove_connection(cnf1, cnf2)
    assert not cmap.remove_connection(cnf1, cnf2)