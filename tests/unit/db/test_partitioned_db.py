import tempfile

from cnf import CrystalNormalForm
from cnf.calculation import GraceCalculator
from cnf.search.waterfill import process_cnf_batch, explore_pt_partition, waterfill_step
from cnf.db.setup_partitions import setup_search_dir
from cnf.navigation.endpoints import get_endpoint_cnfs
from cnf.navigation import find_neighbors
from cnf.db import PartitionedDB
from cnf.utils.log import Logger

def test_sync_water_level_basic(zr_bcc_mp, zr_hcp_mp):
    xi = 1.5
    delta = 10
    sps, eps = get_endpoint_cnfs(zr_bcc_mp, zr_hcp_mp, xi, delta)

    with tempfile.TemporaryDirectory() as tmpdir:
        sid = setup_search_dir(tmpdir, "test", 3, sps, eps, GraceCalculator())
        db = PartitionedDB(tmpdir, sid)
        starting_water_lvl = db.get_current_water_level()
        assert starting_water_lvl is not None
        assert starting_water_lvl > -17, "Hardcoded to current grace calc"

        new_pts = find_neighbors(sps[0])
        for pt in new_pts:
            pid = db.get_partition_idx(pt)
            id = db.get_map_store(pt).add_point(pt)
            db.get_search_store(pt).add_to_search_frontier(sid, pt)
            if pid == 0:
                val = -20
            if pid == 1:
                val = -22
            if pid == 2:
                val = -25
        
            db.get_map_store(pt).set_point_value(id, val)

        assert db.get_current_water_level() == starting_water_lvl

        db.sync_control_water_level()
        assert db.get_current_water_level() == -25

def test_sync_completion_status(zr_bcc_mp, zr_hcp_mp):
    xi = 1.5
    delta = 10
    sps, eps = get_endpoint_cnfs(zr_bcc_mp, zr_hcp_mp, xi, delta, min_atoms=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        sid = setup_search_dir(tmpdir, "test", 3, sps, eps, GraceCalculator())
        db = PartitionedDB(tmpdir, sid)

        assert not db.is_search_complete()
        
        db.get_search_store(eps[0]).add_to_search_frontier(sid, eps[0])
        assert not db.is_search_complete()
        db.sync_search_completion_status()
        assert db.is_search_complete()

        db.get_search_store(eps[0]).remove_point_from_search(sid, eps[0])
        assert db.is_search_complete()
        db.sync_search_completion_status()
        assert not db.is_search_complete()

        db.get_search_store(eps[1]).mark_point_searched(sid, eps[1])
        assert not db.is_search_complete()
        db.sync_search_completion_status()
        assert db.is_search_complete()
