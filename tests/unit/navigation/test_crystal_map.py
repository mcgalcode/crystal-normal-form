import helpers
import pytest
from cnf import CrystalNormalForm, UnitCell
from cnf.navigation.crystal_map import CrystalMap
from cnf.navigation.neighbor_finder import NeighborFinder
from cnf.navigation.exploration import get_endpoints_from_structs, CrystalExplorer
from tqdm import tqdm

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
    assert not cmap.is_point_explored(cnf1)

    cmap.remove_point(cnf2)
    assert cnf1 in cmap
    assert cnf2 not in cmap

    cmap.remove_point(cnf1)
    assert cnf1 not in cmap
    assert cnf2 not in cmap

    with pytest.raises(ValueError) as pyexcp:
        cmap.is_point_explored(cnf1)

    assert "Tried to check if nonexistant node" in pyexcp.value.__repr__()

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

def test_can_explore_point(ti_o2_anatase):
    xi = 1.5
    delta = 10
    cnf = CrystalNormalForm.from_pmg_struct(ti_o2_anatase, xi, delta)

    cmap = CrystalMap.from_cnf(cnf)
    assert not cmap.is_point_explored(cnf)

    pid = cmap.get_point_id(cnf)
    new_pt_ids = cmap.explore_point(pid)
    for nid in new_pt_ids:
        pt = cmap.get_point_by_id(nid)
        assert pt in cmap
        assert cmap.connection_exists(cnf, pt)
        assert not cmap.is_id_explored(nid)
    
    assert cmap.is_point_explored(cnf)

def test_can_connect_two_points(zr_bcc, zr_hcp):
    xi = 1.5
    delta = 10

    start_structs, end_structs = get_endpoints_from_structs(zr_bcc, zr_hcp)
    start_cnfs = [s.to_cnf(xi, delta) for s in start_structs]
    end_cnfs = [s.to_cnf(xi, delta) for s in end_structs]
    cmap = CrystalMap.from_cnf(start_cnfs[0])
    volumes = [s.volume for s in [*start_structs, *end_structs]]
    min_vol = min(volumes) * 0.8
    max_vol = max(volumes) * 1.2

    explorer = CrystalExplorer(cmap, min_vol, max_vol, end_cnfs)
    for cnf in start_cnfs:
        cmap.add_point(cnf)
    
    any_endpts_found = False
    tries = 0
    found_endpt = None
    while not any_endpts_found:
        print(f"Starting round {tries} of searching for endpts, map has {len(cmap)} pts")
        total_added = 0
        pt_to_explore = explorer.best_current_point()
        print(f"")
        print(f"Current best score: {explorer.best_current_score()}")
        diff = 0
        # print(explorer.sorted_pts())
        for pt_pair in explorer.sorted_pts():
            score, pt_id = pt_pair
            pt = cmap.get_point_by_id(pt_id)
            if not explorer.is_point_explored(pt):
                pt_id = cmap.get_point_id(pt)
                print(f"Exploring pt: {pt.coords} (score: {explorer.score_for_point(pt_id)})")
                before_len = len(cmap)
                explorer.explore_point(pt_id)
                after_len = len(cmap)
                diff = after_len - before_len
                total_added += diff
                print(f"Added {after_len - before_len} pts ({before_len} -> {after_len})")
                if diff > 0:
                    break

        
        for cnf in end_cnfs:
            if cnf in cmap:
                any_endpts_found = True
                found_endpt = cnf
        tries += 1
        if total_added == 0:
            break
    print(f"Found endpoint: {found_endpt}")
    explorer.to_json("exploration2.json")
