import pytest
import json
import tempfile

from cnf.db.setup_partitions import setup_search_dir, setup_cnf_db, instantiate_search, PARTITION_SUFFIX, META_DB_NAME
from cnf.db.meta_file import load_meta_file
from cnf.db import CrystalMapStore, SearchProcessStore, MetaStore
from cnf.db.partitioned_db import PartitionedDB
from cnf.calculation.grace import GraceCalculator
from cnf.navigation.endpoints import get_endpoint_cnfs
import glob
import os

def _assert_file_is_cnf_store(fname, xi, delta, element_list):
    cm = CrystalMapStore.from_file(fname)
    md = cm.get_metadata()
    assert md.delta == delta
    assert md.xi == xi
    assert md.element_list == element_list

def _assert_search_instantiated(fname, search_id, eps, sps):
    sp = SearchProcessStore.from_file(fname)
    endpts = [pt.cnf for pt in sp.get_search_endpoints(search_id)]
    assert set(endpts) == set(eps)

    startpts = [pt.cnf for pt in sp.get_search_startpoints(search_id)]
    assert set(startpts) == set(sps)
    
    frontier_pts = sp.get_frontier_points_in_search(search_id)
    assert set([pt.cnf for pt in frontier_pts]) == set(sps)    

def test_can_setup_search_dir(zr_bcc_mp, zr_hcp_mp):
    zr_bcc_mp = zr_bcc_mp.to_primitive()

    xi = 1.5
    delta = 10
    calc = GraceCalculator()
    description = "test search"
    num_partitions = 4

    start_cnfs, end_cnfs = get_endpoint_cnfs(zr_bcc_mp, zr_hcp_mp, xi, delta, None)

    el_list = start_cnfs[0].elements

    with tempfile.TemporaryDirectory() as tmpdir:
        sid = setup_search_dir(tmpdir,
                         description,
                         num_partitions,
                         start_cnfs,
                         end_cnfs,
                         calc)
        
        partition_db = PartitionedDB(tmpdir)
        
        db_files = glob.glob(os.path.join(tmpdir, f"*{PARTITION_SUFFIX}"))
        assert len(db_files) == num_partitions
        for db_file in db_files:
            _assert_file_is_cnf_store(db_file, xi, delta, el_list)
            _assert_search_instantiated(db_file, sid, end_cnfs, start_cnfs)
        
        ms = MetaStore.from_file(os.path.join(tmpdir, META_DB_NAME))
        start_energies = [calc.calculate_energy(sc) for sc in start_cnfs]
        for i in range(num_partitions):
            wl = ms.get_partition_water_level(sid, i)
            assert wl == min(start_energies)

        metadata = partition_db.metadata
        assert metadata.xi == xi
        assert metadata.delta == delta
        assert metadata.calculator_model == calc.identifier()
        assert metadata.atom_list == el_list
        assert metadata.description == description

        sps = metadata.search_processes
        assert len(sps) == 1
        assert sps[0].search_id == sid
        assert set([tuple(cnf) for cnf in sps[0].start_cnfs]) == set([tuple(cnf.coords) for cnf in start_cnfs])
        assert set([tuple(cnf) for cnf in sps[0].end_cnfs]) == set([tuple(cnf.coords) for cnf in end_cnfs])

def test_setup_cnf_store():
    with tempfile.NamedTemporaryFile() as tmp:
        setup_cnf_db(tmp.name, 1.5, 10, ["Zr", "Zr"])
        _assert_file_is_cnf_store(tmp.name, 1.5, 10, ["Zr", "Zr"])
        
        sp = SearchProcessStore.from_file(tmp.name)
        sp.create_search_process("Test process")

def test_instantiate_search(zr_bcc_mp, zr_hcp_mp):
    zr_bcc_mp = zr_bcc_mp.to_primitive()

    sps, eps = get_endpoint_cnfs(zr_bcc_mp, zr_hcp_mp, 1.5, 10)

    with tempfile.NamedTemporaryFile() as tmp:
        setup_cnf_db(tmp.name, 1.5, 10, ["Zr", "Zr"])
        sid = instantiate_search("Test search", sps, eps, tmp.name, GraceCalculator())
        _assert_search_instantiated(tmp.name, sid, eps, sps)