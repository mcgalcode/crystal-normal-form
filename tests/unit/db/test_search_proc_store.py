import pytest
import tempfile

from cnf import UnitCell

from cnf.db.crystal_map_store import CrystalMapStore
from cnf.db.search_store import SearchProcessStore
from cnf.db.setup import setup_cnf_db, instantiate_search
from cnf.search import explore_pt
from cnf.navigation.neighbor_finder import find_neighbors
from cnf.calculation.grace import GraceCalculator

XI = 1.5
DELTA = 10

@pytest.fixture
def zr_bcc_cnfs(zr_bcc_mp):
    scs = UnitCell.from_pymatgen_structure(zr_bcc_mp).supercells(2)
    return list(set([sc.to_cnf(XI, DELTA) for sc in scs]))

@pytest.fixture
def zr_hcp_cnfs(zr_hcp_mp):
    uc = UnitCell.from_pymatgen_structure(zr_hcp_mp)
    scs = uc.supercells(2)
    return list(set([sc.to_cnf(XI, DELTA) for sc in scs]))


@pytest.fixture(scope='function')
def store_file(zr_hcp_cnfs):
    with tempfile.NamedTemporaryFile() as tf:
        setup_cnf_db(tf.name, XI, DELTA, zr_hcp_cnfs[0].elements)
        yield tf.name

@pytest.fixture
def crystal_map_store(store_file, zr_bcc_cnfs, zr_hcp_cnfs):
    store = CrystalMapStore.from_file(store_file)  
    return store

@pytest.fixture(scope='function')
def search_store(crystal_map_store):
    return SearchProcessStore.from_file(crystal_map_store.db_filename)


def test_can_add_and_rm_pt_from_frontier(search_store, zr_bcc_cnfs, zr_hcp_cnfs):
    sp_id = instantiate_search(
        "test process",
        zr_bcc_cnfs,
        zr_hcp_cnfs,
        search_store.db_filename,
        GraceCalculator()
    )

    start_pt_frontier = search_store.get_frontier_points_in_search(sp_id)
    assert len(start_pt_frontier) == len(zr_bcc_cnfs)

    search_store.add_to_search_frontier(sp_id, zr_hcp_cnfs[0])
    new_frontier = search_store.get_frontier_points_in_search(sp_id)
    assert len(new_frontier) == len(zr_bcc_cnfs) + 1
    assert set([pt.cnf for pt in new_frontier]) == set(zr_bcc_cnfs).union(set(zr_hcp_cnfs[:1]))

    search_store.remove_from_search_frontier(sp_id, zr_hcp_cnfs[0])
    start_pt_frontier = search_store.get_frontier_points_in_search(sp_id)
    assert len(start_pt_frontier) == len(zr_bcc_cnfs)

def test_can_mark_point_as_searched(search_store, zr_bcc_cnfs, zr_hcp_cnfs):
    sp_id = instantiate_search(
        "test process",
        zr_bcc_cnfs,
        zr_hcp_cnfs,
        search_store.db_filename,
        GraceCalculator()
    )

    searched_pts = search_store.get_searched_points_in_search(sp_id)
    assert len(searched_pts) == 0

    search_store.mark_point_searched(sp_id, zr_bcc_cnfs[0])
    searched_pts = search_store.get_searched_points_in_search(sp_id)
    assert len(searched_pts) == 1
    assert searched_pts[0].cnf == zr_bcc_cnfs[0]

    search_store.mark_point_searched(sp_id, zr_bcc_cnfs[1])
    searched_pts = search_store.get_searched_points_in_search(sp_id)
    searched_cnfs = [pt.cnf for pt in searched_pts]
    assert len(searched_cnfs) == 2
    assert set(searched_cnfs) == set(zr_bcc_cnfs[:2])

def test_candidate_neighbors_are_not_searched(search_store, crystal_map_store, zr_bcc_cnfs, zr_hcp_cnfs):
    sp_id = instantiate_search(
        "test process",
        zr_bcc_cnfs,
        zr_hcp_cnfs,
        search_store.db_filename,
        GraceCalculator()
    )
    start_pt = zr_bcc_cnfs[0]
    start_pt_id = crystal_map_store.get_point_by_cnf(start_pt).id
    all_nb_ids, new_nb_ids = explore_pt(crystal_map_store, start_pt_id)
    # label some of these as searched
    searched_nb_ids = all_nb_ids[:10]
    unsearched_nb_ids = all_nb_ids[10:]
    assert len(unsearched_nb_ids) > 10
    for sid in searched_nb_ids:
        search_store.mark_point_searched_by_id(sp_id, sid)
    
    simple_nbs = crystal_map_store.get_neighbors(start_pt_id)
    simple_nb_ids = [snb.id for snb in simple_nbs]
    assert len(simple_nb_ids) == len(all_nb_ids)
    assert set(simple_nb_ids) == set(all_nb_ids)

def test_candidate_neighbors_are_not_in_frontier(search_store: SearchProcessStore,
                                                 crystal_map_store: CrystalMapStore,
                                                 zr_bcc_cnfs,
                                                 zr_hcp_cnfs):
    sp_id = instantiate_search(
        "test process",
        zr_bcc_cnfs,
        zr_hcp_cnfs,
        search_store.db_filename,
        GraceCalculator()
    )

    start_pt = zr_bcc_cnfs[1]
    start_pt_id = crystal_map_store.get_point_by_cnf(start_pt).id
    all_nb_ids, new_nb_ids = explore_pt(crystal_map_store, start_pt_id)
    # label some of these as searched
    frontier_nb_ids = all_nb_ids[:10]
    non_frontier_nb_ids = all_nb_ids[10:]
    assert len(non_frontier_nb_ids) > 3
    for sid in frontier_nb_ids:
        search_store.add_to_search_frontier_by_id(sp_id, sid)
    
    simple_nbs = crystal_map_store.get_neighbors(start_pt_id)
    simple_nb_ids = [snb.id for snb in simple_nbs]
    assert len(simple_nb_ids) == len(all_nb_ids)
    assert set(simple_nb_ids) == set(all_nb_ids)

def test_can_get_endpoint_ids_in_frontier(search_store: SearchProcessStore,
                                          crystal_map_store: CrystalMapStore,
                                          zr_bcc_cnfs,
                                          zr_hcp_cnfs):
    sp_id = instantiate_search(
        "test process",
        zr_bcc_cnfs,
        zr_hcp_cnfs,
        search_store.db_filename,
        GraceCalculator()
    )

    endpt_ids = search_store.get_endpoint_ids_in_frontier(sp_id)
    assert len(endpt_ids) == 0

    search_store.add_to_search_frontier(sp_id, zr_hcp_cnfs[0])
    endpt_id1 = crystal_map_store.get_point_ids([zr_hcp_cnfs[0]])[0]

    endpt_ids = search_store.get_endpoint_ids_in_frontier(sp_id)
    assert len(endpt_ids) == 1
    assert endpt_ids[0] == endpt_id1

    search_store.add_to_search_frontier(sp_id, zr_hcp_cnfs[1])
    endpt_id2 = crystal_map_store.get_point_ids([zr_hcp_cnfs[1]])[0]

    endpt_ids = search_store.get_endpoint_ids_in_frontier(sp_id)
    assert len(endpt_ids) == 2
    assert set(endpt_ids) == set([endpt_id1, endpt_id2])

    search_store.remove_from_search_frontier_by_id(sp_id, endpt_id1)

    endpt_ids = search_store.get_endpoint_ids_in_frontier(sp_id)
    assert len(endpt_ids) == 1
    assert endpt_ids[0] == endpt_id2

def test_can_manipulate_incoming_points(search_store: SearchProcessStore,
                                        crystal_map_store: CrystalMapStore,
                                        zr_bcc_cnfs,
                                        zr_hcp_cnfs):
    sp_id = instantiate_search(
        "test process",
        zr_bcc_cnfs,
        zr_hcp_cnfs,
        search_store.db_filename,
        GraceCalculator()
    )

    nbs = find_neighbors(zr_bcc_cnfs[0])

    incoming_pts = search_store.get_and_empty_incoming_points(sp_id)
    assert len(incoming_pts) == 0

    for nb in nbs:
        search_store.add_incoming_point(sp_id, nb)
    
    incoming_pts = search_store.get_and_empty_incoming_points(sp_id)
    assert len(incoming_pts) == len(nbs)
    assert set(incoming_pts) == set(nbs)

    incoming_pts = search_store.get_and_empty_incoming_points(sp_id)
    assert len(incoming_pts) == 0
