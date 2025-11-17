import pytest
import tempfile

from cnf import CrystalNormalForm, UnitCell
from cnf.navigation.crystal_explorer import CrystalExplorer, CrystalMap
from cnf.navigation.search_filters import SimpleVolumeAndOverlapFilter
from cnf.navigation.score_functions import NullScore

from cnf.db.crystal_map_store import CrystalMapStore
from cnf.db.search_store import SearchProcessStore
from cnf.db.setup import setup_cnf_db

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
def crystal_map_store(store_file):
    return CrystalMapStore.from_file(store_file)

@pytest.fixture
def search_store(store_file):
    return SearchProcessStore.from_file(store_file)

def test_can_add_search_proc(crystal_map_store, search_store, zr_bcc_cnfs, zr_hcp_cnfs):

    for cnf in zr_bcc_cnfs:
        crystal_map_store.add_point(cnf)
    
    for cnf in zr_hcp_cnfs:
        crystal_map_store.add_point(cnf)

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
