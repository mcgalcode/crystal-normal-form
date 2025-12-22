import tempfile

from cnf import CrystalNormalForm
from cnf.calculation import GraceCalculator
from cnf.search.waterfill import process_cnf_batch, explore_pt_partition, waterfill_step, continue_search_waterfill
from cnf.db.setup_partitions import setup_search_dir
from cnf.navigation.endpoints import get_endpoint_cnfs
from cnf.navigation import find_neighbors
from cnf.db import PartitionedDB
from cnf.utils.log import Logger

def test_process_cnf_batch_basic(zr_bcc_mp, zr_hcp_mp):
    xi = 1.5
    delta = 10
    sps, eps = get_endpoint_cnfs(zr_bcc_mp, zr_hcp_mp, xi, delta)

    with tempfile.TemporaryDirectory() as tmpdir:
        setup_search_dir(tmpdir, "test", 1, sps, eps, GraceCalculator())
        pdb = PartitionedDB(tmpdir, 1)

        cnfs_to_proc = find_neighbors(sps[0])
        assert len(cnfs_to_proc) > 15
        already_added_cnfs = cnfs_to_proc[:10]
        already_in_frontier_cnfs = already_added_cnfs[:3]
        already_searched_cnfs = already_added_cnfs[3:6]
        for cnf in already_added_cnfs:
            pdb.get_map_store(cnf).add_point(cnf)
        
        for cnf in already_searched_cnfs:
            pdb.get_search_store(cnf).mark_point_searched(1, cnf)
        
        for cnf in already_in_frontier_cnfs:
            pdb.get_search_store(cnf).add_to_search_frontier(1, cnf)
        
        process_cnf_batch(cnfs_to_proc,
                          pdb.get_map_store_by_idx(0),
                          pdb.get_search_store_by_idx(0),
                          1,
                          GraceCalculator())
        
        for cnf in already_searched_cnfs:
            assert cnf in pdb.get_search_store(cnf).get_searched_cnfs_in_search(1)
            assert not cnf in pdb.get_search_store(cnf).get_frontier_cnfs_in_search(1)
        
        for cnf in cnfs_to_proc:
            if cnf not in already_searched_cnfs:
                assert cnf in pdb.get_search_store(cnf).get_frontier_cnfs_in_search(1)
        
        for cnf in cnfs_to_proc:
            if cnf not in already_searched_cnfs and cnf not in already_in_frontier_cnfs:
                assert pdb.get_map_store(cnf).get_point_by_cnf(cnf).value is not None

def _assert_graph_is_correct_in_local_partition(pdb: PartitionedDB, explored_pt: CrystalNormalForm):
    local_partition_idx = pdb.get_partition_idx(explored_pt)
    local_pt_id = pdb.get_map_store(explored_pt).get_point_by_cnf(explored_pt).id

    known_nbs = find_neighbors(explored_pt)  
    expected_parts = pdb.partition_cnfs(known_nbs)    
    local_nodes = expected_parts[local_partition_idx]
    for n in local_nodes:
        assert pdb.get_map_store(explored_pt).get_point_by_cnf(n) is not None
    
    # assert that all local edges are added to local
    local_nbs = pdb.get_map_store(explored_pt).get_local_neighbors(local_pt_id)
    local_nb_cnfs = [pt.cnf for pt in local_nbs]
    assert len(local_nb_cnfs) == len(local_nodes)
    assert set(local_nb_cnfs) == set(local_nodes)

    # assert that all edges are added to local partition
    # (the only place where edges should be added)
    retrieved_nonlocal_nodes = pdb.get_map_store(explored_pt).get_nonlocal_neighbor_cnfs(local_pt_id)
    expected_nonlocal_nodes = set(known_nbs).difference(local_nodes)
    assert len(retrieved_nonlocal_nodes) == len(expected_nonlocal_nodes)
    assert set(retrieved_nonlocal_nodes) == set(expected_nonlocal_nodes)    


def test_explore_point_multiple_partitions(zr_bcc_mp, zr_hcp_mp):
    xi = 1.5
    delta = 10
    sps, eps = get_endpoint_cnfs(zr_bcc_mp, zr_hcp_mp, xi, delta)

    with tempfile.TemporaryDirectory() as tmpdir:
        sp_id = setup_search_dir(tmpdir, "test", 3, sps, eps, GraceCalculator())
        pdb = PartitionedDB(tmpdir, sp_id)

        # select a partition to act as the source partition
        pt_to_explore = sps[0]

        explore_pt_partition(pdb, pt_to_explore)
        local_partition_idx = pdb.get_partition_idx(pt_to_explore)
        
        # assert that point is marked as explored
        assert pdb.get_map_store(pt_to_explore).get_point_by_cnf(pt_to_explore).explored
        _assert_graph_is_correct_in_local_partition(pdb, pt_to_explore)

        # for every other partition / node_set
        for partition_idx, cnfs in pdb.partition_cnfs(find_neighbors(pt_to_explore)).items():
            if partition_idx == local_partition_idx:
                continue

            # Assert that the inbox holds the same set of CNFs
            inbox_cnfs = pdb.get_search_store_by_idx(partition_idx).peek_incoming_points(sp_id)
            assert len(inbox_cnfs) == len(cnfs)
            assert set(inbox_cnfs) == set(cnfs)

            # assert that these points have not been added anywhere yet
            for cnf in cnfs:
                assert pdb.get_map_store_by_idx(local_partition_idx).get_point_by_cnf(cnf) is None
                assert pdb.get_map_store_by_idx(partition_idx).get_point_by_cnf(cnf) is None

def test_explore_already_explored_pt(zr_bcc_mp, zr_hcp_mp):
    xi = 1.5
    delta = 10
    sps, eps = get_endpoint_cnfs(zr_bcc_mp, zr_hcp_mp, xi, delta)

    with tempfile.TemporaryDirectory() as tmpdir:
        sp_id = setup_search_dir(tmpdir, "test", 3, sps, eps, GraceCalculator())
        pdb = PartitionedDB(tmpdir, sp_id)

        # select a partition to act as the source partition
        pt_to_explore = sps[0]

        nbs = find_neighbors(pt_to_explore)
        local_nbs = pdb.partition_cnfs(nbs)[pdb.get_partition_idx(pt_to_explore)]
        local_ids_1 = explore_pt_partition(pdb, pt_to_explore)
        assert len(local_nbs) == len(local_ids_1)
        local_ids_2 = explore_pt_partition(pdb, pt_to_explore)
        assert set(local_ids_1) == set(local_ids_2)

        retrieved_local_nbs = pdb.get_map_store(pt_to_explore).get_points_by_ids(local_ids_2)
        retrieved_local_nbs = [pt.cnf for pt in retrieved_local_nbs]
        assert set(retrieved_local_nbs) == set(local_nbs)

def test_take_waterfill_step(zr_bcc_mp, zr_hcp_mp):
    xi = 1.5
    delta = 10
    sps, eps = get_endpoint_cnfs(zr_bcc_mp, zr_hcp_mp, xi, delta)
    logger = Logger()

    DUMB_ENERGY_LIMIT = 100

    with tempfile.TemporaryDirectory() as tmpdir:
        sp_id = setup_search_dir(tmpdir, "test", 3, sps, eps, GraceCalculator())
        pdb = PartitionedDB(tmpdir, sp_id)


        waterfill_step(pdb, 0, sp_id, logger, GraceCalculator(), 100, DUMB_ENERGY_LIMIT, 2)

        for cnf_pt in sps:
            local_partition_idx = pdb.get_partition_idx(cnf_pt)
            
            nbs = find_neighbors(cnf_pt)
            for nb in nbs:
                if pdb.get_partition_idx(nb) == local_partition_idx:
                    assert pdb.get_search_store(nb).is_point_in_frontier(sp_id, nb)
                    assert pdb.get_map_store(nb).get_point_by_cnf(nb).value is not None

            # assert that point is marked as explored
            assert pdb.get_map_store(cnf_pt).get_point_by_cnf(cnf_pt).explored
            assert pdb.get_search_store(cnf_pt).is_point_searched(sp_id, cnf_pt)
            _assert_graph_is_correct_in_local_partition(pdb, cnf_pt)

            # for every other partition / node_set
            for partition_idx, cnfs in pdb.partition_cnfs(nbs).items():
                if partition_idx == local_partition_idx:
                    continue

                # Assert that the inbox holds the same set of CNFs
                inbox_cnfs = pdb.get_search_store_by_idx(partition_idx).peek_incoming_points(sp_id)
                assert set(inbox_cnfs).issuperset(cnfs)

                # assert that these points have not been added anywhere yet
                for cnf in cnfs:
                    assert pdb.get_map_store_by_idx(local_partition_idx).get_point_by_cnf(cnf) is None
                    assert pdb.get_map_store_by_idx(partition_idx).get_point_by_cnf(cnf) is None   

def test_take_second_waterfill_step(zr_bcc_mp, zr_hcp_mp):
    xi = 1.5
    delta = 10
    sps, eps = get_endpoint_cnfs(zr_bcc_mp, zr_hcp_mp, xi, delta)
    logger = Logger()

    DUMB_ENERGY_LIMIT = 100

    with tempfile.TemporaryDirectory() as tmpdir:
        sp_id = setup_search_dir(tmpdir, "test", 3, sps, eps, GraceCalculator())
        pdb = PartitionedDB(tmpdir, sp_id)


        waterfill_step(pdb, 0, sp_id, logger, GraceCalculator(), 100, DUMB_ENERGY_LIMIT, 2)
        waterfill_step(pdb, 1, sp_id, logger, GraceCalculator(), 100, DUMB_ENERGY_LIMIT, 2)
        waterfill_step(pdb, 2, sp_id, logger, GraceCalculator(), 100, DUMB_ENERGY_LIMIT, 2)

        for cnf_pt in sps:
            local_partition_idx = pdb.get_partition_idx(cnf_pt)
            
            nbs = find_neighbors(cnf_pt)
            for nb in nbs:
                if pdb.get_partition_idx(nb) == local_partition_idx:
                    assert pdb.get_map_store(nb).get_point_by_cnf(nb).value is not None


def test_waterfill_tio2(ti_o2_anatase, ti_o2_rutile):
    xi = 1.5
    delta = 10
    sps, eps = get_endpoint_cnfs(ti_o2_anatase, ti_o2_rutile, xi, delta)
    logger = Logger()

    DUMB_ENERGY_LIMIT = 100

    with tempfile.TemporaryDirectory() as tmpdir:
        sp_id = setup_search_dir(tmpdir, "test", 3, sps, eps, GraceCalculator())


        continue_search_waterfill(sp_id, tmpdir, GraceCalculator(), 3, batch_size=10)

def test_waterfill_zr(zr_bcc_mp, zr_hcp_mp):
    xi = 1.5
    delta = 10
    sps, eps = get_endpoint_cnfs(zr_bcc_mp, zr_hcp_mp, xi, delta)
    logger = Logger()

    DUMB_ENERGY_LIMIT = 100

    with tempfile.TemporaryDirectory() as tmpdir:
        sp_id = setup_search_dir(tmpdir, "test", 32, sps, eps, GraceCalculator())


        continue_search_waterfill(sp_id, tmpdir, GraceCalculator(), 5, batch_size=10)