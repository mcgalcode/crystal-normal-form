import pytest
import tempfile

from cnf.search import instantiate_search, explore_pt
from cnf.db.setup import setup_cnf_db
from cnf.db.search_store import SearchProcessStore
from cnf.db.crystal_map_store import CrystalMapStore
from cnf import UnitCell, CrystalNormalForm
from cnf.navigation.neighbor_finder import NeighborFinder


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
def cnf_db_file(zr_bcc_cnfs):
    with tempfile.NamedTemporaryFile() as tf:
        setup_cnf_db(tf.name, XI, DELTA, zr_bcc_cnfs[0].elements)
        yield tf.name

def test_can_instantiate_search(cnf_db_file, zr_bcc_cnfs, zr_hcp_cnfs):
    sp_id = instantiate_search(
        "test process",
        zr_bcc_cnfs,
        zr_hcp_cnfs,
        cnf_db_file
    )

    search_store = SearchProcessStore.from_file(cnf_db_file)
    cmap_store = CrystalMapStore.from_file(cnf_db_file)
    endpts = search_store.get_search_endpoints(sp_id)
    assert len(endpts) == len(zr_hcp_cnfs)
    assert set(zr_hcp_cnfs) == set([pt.cnf for pt in endpts])

    startpts = search_store.get_search_startpoints(sp_id)
    assert len(startpts) == len(zr_hcp_cnfs)
    assert set(zr_bcc_cnfs) == set([pt.cnf for pt in startpts])

    frontier_ids = search_store.get_frontier_point_ids(sp_id)
    assert sorted(frontier_ids) == sorted(cmap_store.get_point_ids(zr_bcc_cnfs))

def test_can_explore_point(cnf_db_file, zr_bcc_cnfs):
    cmap = CrystalMapStore.from_file(cnf_db_file)
    pt_id = cmap.add_point(zr_bcc_cnfs[0])

    nf = NeighborFinder(zr_bcc_cnfs[0])
    nbs = nf.find_neighbors()

    explore_pt(cmap, pt_id)

    for nb in nbs:
        nb_id = cmap.get_point_ids([nb])[0]
        assert cmap.connection_exists_by_id(nb_id, pt_id)
        assert cmap.connection_exists_by_id(pt_id, nb_id)
