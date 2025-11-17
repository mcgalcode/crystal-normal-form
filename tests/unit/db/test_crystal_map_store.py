import pytest
from cnf.db.crystal_map_store import CrystalMapStore
from cnf.db.setup import setup_cnf_db
from cnf import CrystalNormalForm
from cnf.navigation.neighbor_finder import LatticeNeighborFinder
import tempfile


@pytest.fixture
def zr_hcp_cnf(zr_hcp_mp):
    xi = 1.5
    delta = 10
    return CrystalNormalForm.from_pmg_struct(zr_hcp_mp, xi, delta)

@pytest.fixture
def zr_bcc_cnf(zr_bcc_mp):
    xi = 1.5
    delta = 10
    return CrystalNormalForm.from_pmg_struct(zr_bcc_mp, xi, delta)

@pytest.fixture(scope='function')
def temp_db(zr_hcp_cnf):
    els = zr_hcp_cnf.elements
    with tempfile.NamedTemporaryFile('w') as tf:
        cs = setup_cnf_db(tf.name, zr_hcp_cnf.xi, zr_hcp_cnf.delta, els)
        yield CrystalMapStore(cs)

def test_cannot_instantiate_with_bad_db():
    with tempfile.NamedTemporaryFile('w') as tf:
        fname = tf.name
        with pytest.raises(ValueError) as captured_excep:
            CrystalMapStore(fname)
        
        assert "Tried to instantiate CrystalMapStore from uninitialized DB file:" in captured_excep.value.__repr__()

def test_can_instantiate_after_setup(temp_db, zr_hcp_cnf):
    cs2 = CrystalMapStore(temp_db.db_filename)
    assert cs2 is not None

    metadata = cs2.get_metadata()
    assert metadata.delta == zr_hcp_cnf.delta
    assert metadata.xi == zr_hcp_cnf.xi
    assert metadata.element_list == zr_hcp_cnf.elements

def test_can_add_and_retrieve_row(zr_hcp_cnf, temp_db: CrystalMapStore):
    temp_db.add_point(zr_hcp_cnf)
    
    result = temp_db.get_point_by_cnf(zr_hcp_cnf)
    assert result.cnf == zr_hcp_cnf
    assert result.explored == False
    assert result.value is None
    assert result.external_id is None
    assert result.id
    
def test_can_add_and_remove_row(zr_hcp_cnf, temp_db: CrystalMapStore):
    temp_db.add_point(zr_hcp_cnf)
    
    result = temp_db.get_point_by_cnf(zr_hcp_cnf)
    assert result.cnf == zr_hcp_cnf
    assert result.explored == False
    assert result.value is None
    assert result.external_id is None
    assert result.id

    temp_db.remove_point(zr_hcp_cnf)

    result2 = temp_db.get_point_by_cnf(zr_hcp_cnf)
    assert result2 is None

def test_can_get_pt_by_id(zr_hcp_cnf, temp_db: CrystalMapStore):
    temp_db.add_point(zr_hcp_cnf)
    
    result = temp_db.get_point_by_cnf(zr_hcp_cnf)
    result = temp_db.get_point_by_id(result.id)
    assert result.cnf == zr_hcp_cnf
    assert result.explored == False
    assert result.value is None
    assert result.external_id is None
    assert result.id   

def test_can_get_multiple_ids(zr_hcp_cnf, temp_db: CrystalMapStore):
    lnfnf = LatticeNeighborFinder(zr_hcp_cnf)
    nbs = lnfnf.find_cnf_neighbors()
    cnfs = [nb.point for nb in nbs.neighbors]
    for c in cnfs:
        temp_db.add_point(c)
    
    cnfs_to_retrieve = cnfs[::5]

    all_ids = temp_db.get_point_ids(cnfs_to_retrieve)
    assert len(all_ids) == len(cnfs_to_retrieve)
    for cid, cnf in zip(all_ids, cnfs_to_retrieve):
        retrieved = temp_db.get_point_by_id(cid)
        assert retrieved.cnf == cnf

def test_can_getting_multiple_ids_with_bad_id_raises_error(zr_hcp_cnf, temp_db: CrystalMapStore):
    lnfnf = LatticeNeighborFinder(zr_hcp_cnf)
    nbs = lnfnf.find_cnf_neighbors()
    cnfs = [nb.point for nb in nbs.neighbors]
    for c in cnfs[:5]:
        temp_db.add_point(c)
    
    cnfs_to_retrieve = cnfs[:6]

    with pytest.raises(ValueError) as excep:
        all_ids = temp_db.get_point_ids(cnfs_to_retrieve)
        assert "No row in CNFStore found for CNF " in excep.value.__repr__()

def test_can_add_connection(zr_hcp_cnf, zr_bcc_cnf, temp_db: CrystalMapStore):
    temp_db.add_point(zr_hcp_cnf)
    temp_db.add_point(zr_bcc_cnf)

    assert not temp_db.connection_exists(zr_hcp_cnf, zr_bcc_cnf)
    assert not temp_db.connection_exists(zr_bcc_cnf, zr_hcp_cnf)

    temp_db.add_connection(zr_hcp_cnf, zr_bcc_cnf)

    assert temp_db.connection_exists(zr_hcp_cnf, zr_bcc_cnf)
    assert temp_db.connection_exists(zr_bcc_cnf, zr_hcp_cnf)

    temp_db.remove_connection(zr_hcp_cnf, zr_bcc_cnf)

    assert not temp_db.connection_exists(zr_hcp_cnf, zr_bcc_cnf)
    assert not temp_db.connection_exists(zr_bcc_cnf, zr_hcp_cnf)


