import tempfile

from cnf.calculation import GraceCalculator
from cnf.search.waterfill import process_cnf_batch
from cnf.db.setup_partitions import setup_search_dir
from cnf.navigation.endpoints import get_endpoint_cnfs
from cnf.navigation import find_neighbors
from cnf.db import PartitionedDB

def test_process_cnf_batch(zr_bcc_mp, zr_hcp_mp):
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
