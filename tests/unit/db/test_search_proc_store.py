import pytest
import tempfile

from cnf import CrystalNormalForm, UnitCell
from cnf.navigation.crystal_explorer import CrystalExplorer, CrystalMap
from cnf.navigation.search_filters import SimpleVolumeAndOverlapFilter
from cnf.navigation.score_functions import NullScore

from cnf.db.crystal_map_store import CrystalMapStore
from cnf.db.search_store import SearchProcessStore
from cnf.db.setup import setup_cnf_db
from cnf.db.exploration import explore_pt

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
    for cnf in zr_bcc_cnfs:
        store.add_point(cnf)
    
    for cnf in zr_hcp_cnfs:
        store.add_point(cnf)    
    return store

@pytest.fixture
def search_store(crystal_map_store):
    return SearchProcessStore.from_file(crystal_map_store.db_filename)

def test_can_add_search_proc(search_store, zr_bcc_cnfs, zr_hcp_cnfs):
    
    sp_id = search_store.create_search_process(
        "test process",
        zr_bcc_cnfs,
        zr_hcp_cnfs
    )

    endpts = search_store.get_search_endpoints(sp_id)
    assert len(endpts) == len(zr_hcp_cnfs)
    assert set(zr_hcp_cnfs) == set([pt.cnf for pt in endpts])

    startpts = search_store.get_search_startpoints(sp_id)
    assert len(startpts) == len(zr_hcp_cnfs)
    assert set(zr_bcc_cnfs) == set([pt.cnf for pt in startpts])

def test_can_add_and_rm_pt_from_frontier(search_store, zr_bcc_cnfs, zr_hcp_cnfs):
    sp_id = search_store.create_search_process(
        "test process",
        zr_bcc_cnfs,
        zr_hcp_cnfs
    )

    empty_frontier = search_store.get_frontier_points_in_search(sp_id)
    assert len(empty_frontier) == 0

    search_store.add_to_search_frontier(sp_id, zr_bcc_cnfs[0])
    single_pt_frontier = search_store.get_frontier_points_in_search(sp_id)
    assert len(single_pt_frontier) == 1
    assert single_pt_frontier[0].cnf == zr_bcc_cnfs[0]

    search_store.remove_from_search_frontier(sp_id, zr_bcc_cnfs[0])
    empty_frontier = search_store.get_frontier_points_in_search(sp_id)
    assert len(empty_frontier) == 0

def test_can_mark_point_as_searched(search_store, zr_bcc_cnfs, zr_hcp_cnfs):
    sp_id = search_store.create_search_process(
        "test process",
        zr_bcc_cnfs,
        zr_hcp_cnfs
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
    sp_id = search_store.create_search_process(
        "test process",
        zr_bcc_cnfs,
        zr_hcp_cnfs
    )

    start_pt = zr_bcc_cnfs[0]
    start_pt_id = crystal_map_store.get_point_by_cnf(start_pt).id
    all_nb_ids = explore_pt(crystal_map_store, start_pt_id)
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

    nbs, _ = search_store.get_unsearched_neighbors_with_lock_info(sp_id, start_pt_id)
    retrieved_nb_ids = [nb.id for nb in nbs]
    assert len(retrieved_nb_ids) == len(unsearched_nb_ids)
    assert set(retrieved_nb_ids) == set(unsearched_nb_ids)

def test_candidate_neighbors_are_not_in_frontier(search_store, crystal_map_store, zr_bcc_cnfs, zr_hcp_cnfs):
    sp_id = search_store.create_search_process(
        "test process",
        zr_bcc_cnfs,
        zr_hcp_cnfs
    )

    start_pt = zr_bcc_cnfs[0]
    start_pt_id = crystal_map_store.get_point_by_cnf(start_pt).id
    all_nb_ids = explore_pt(crystal_map_store, start_pt_id)
    # label some of these as searched
    frontier_nb_ids = all_nb_ids[:10]
    non_frontier_nb_ids = all_nb_ids[10:]
    assert len(non_frontier_nb_ids) > 10
    for sid in frontier_nb_ids:
        search_store.add_to_search_frontier_by_id(sp_id, sid)
    
    simple_nbs = crystal_map_store.get_neighbors(start_pt_id)
    simple_nb_ids = [snb.id for snb in simple_nbs]
    assert len(simple_nb_ids) == len(all_nb_ids)
    assert set(simple_nb_ids) == set(all_nb_ids)

    nbs, _ = search_store.get_unsearched_neighbors_with_lock_info(sp_id, start_pt_id)
    retrieved_nb_ids = [nb.id for nb in nbs]
    assert len(retrieved_nb_ids) == len(non_frontier_nb_ids)
    assert set(retrieved_nb_ids) == set(non_frontier_nb_ids)

def test_candidate_neighbors_have_lock_info(search_store, crystal_map_store, zr_bcc_cnfs, zr_hcp_cnfs):
    sp_id = search_store.create_search_process(
        "test process",
        zr_bcc_cnfs,
        zr_hcp_cnfs
    )

    start_pt = zr_bcc_cnfs[0]
    start_pt_id = crystal_map_store.get_point_by_cnf(start_pt).id
    all_nb_ids = explore_pt(crystal_map_store, start_pt_id)
    # label some of these as searched
    locked_nb_ids = all_nb_ids[:10]
    unlocked_nb_ids = all_nb_ids[10:]
    assert len(unlocked_nb_ids) > 10
    for sid in locked_nb_ids:
        crystal_map_store.lock_point(sid)

    _, lock_info = search_store.get_unsearched_neighbors_with_lock_info(sp_id, start_pt_id)
    retrieved_locked_ids = [k for k, v in lock_info.items() if v == True]
    retrieved_unlocked_ids = [k for k, v in lock_info.items() if v == False]

    assert len(retrieved_locked_ids) == len(locked_nb_ids)
    assert set(retrieved_locked_ids) == set(locked_nb_ids)

    assert len(retrieved_unlocked_ids) == len(unlocked_nb_ids)
    assert set(retrieved_unlocked_ids) == set(unlocked_nb_ids)

    


