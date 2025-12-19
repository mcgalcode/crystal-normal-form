import pytest
from cnf.db.crystal_map_store import CrystalMapStore
from cnf.db.setup import setup_cnf_db
from cnf.search import explore_pt
from cnf import CrystalNormalForm
from cnf.navigation.neighbor_finder import NeighborFinder, find_neighbors
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
        yield CrystalMapStore.from_file(cs)

def test_cannot_instantiate_with_bad_db():
    with tempfile.NamedTemporaryFile('w') as tf:
        fname = tf.name
        with pytest.raises(ValueError) as captured_excep:
            CrystalMapStore.from_file(fname)
        
        assert "Tried to instantiate CrystalMapStore from uninitialized DB file:" in captured_excep.value.__repr__()

def test_can_instantiate_after_setup(temp_db, zr_hcp_cnf):
    cs2 = CrystalMapStore.from_file(temp_db.db_filename)
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
    lnfnf = NeighborFinder.from_cnf(zr_hcp_cnf)
    nbs = lnfnf.find_neighbors(zr_hcp_cnf)
    cnfs = [nb for nb in nbs]
    for c in cnfs:
        temp_db.add_point(c)
    
    cnfs_to_retrieve = cnfs[::5]

    all_ids = temp_db.get_point_ids(cnfs_to_retrieve)
    assert len(all_ids) == len(cnfs_to_retrieve)
    for cid, cnf in zip(all_ids, cnfs_to_retrieve):
        retrieved = temp_db.get_point_by_id(cid)
        assert retrieved.cnf == cnf

def test_can_getting_multiple_ids_with_bad_id_raises_error(zr_hcp_cnf, temp_db: CrystalMapStore):
    lnfnf = NeighborFinder.from_cnf(zr_hcp_cnf)
    nbs = lnfnf.find_neighbors(zr_hcp_cnf)
    cnfs = [nb for nb in nbs]
    for c in cnfs[:5]:
        temp_db.add_point(c)
    assert len(cnfs) > 5
    cnfs_to_retrieve = cnfs[:6]

    with pytest.raises(ValueError) as excep:
        all_ids = temp_db.get_point_ids(cnfs_to_retrieve)
    assert "No row in CNFStore found for CNF" in excep.value.__repr__()

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

def test_can_get_all_neighbors_of_point(zr_hcp_cnf, temp_db: CrystalMapStore):
    temp_db.add_point(zr_hcp_cnf)
    nf = NeighborFinder.from_cnf(zr_hcp_cnf)
    nbs = nf.find_neighbors(zr_hcp_cnf)
    for nb in nbs:
        temp_db.add_point(nb)
        temp_db.add_connection(zr_hcp_cnf, nb)

    other_nbs = nf.find_neighbors(nbs[-1])
    for onb in other_nbs:
        existing = temp_db.get_point_by_cnf(onb)
        if existing is None:
            temp_db.add_point(onb)

        temp_db.add_connection(nbs[-1], onb)
    
    retrieved_nbs = temp_db.get_local_neighbors(temp_db.get_point_ids([zr_hcp_cnf])[0])
    assert len(retrieved_nbs) == len(nbs)
    assert set([nb.cnf for nb in retrieved_nbs]) == set(nbs)

def test_can_mark_point_explored(zr_hcp_cnf, temp_db):
    pt_id = temp_db.add_point(zr_hcp_cnf)
    
    assert temp_db.get_point_by_id(pt_id).explored == False

    temp_db.mark_point_explored(pt_id)
    assert temp_db.get_point_by_id(pt_id).explored == True

    temp_db.mark_point_unexplored(pt_id)
    assert temp_db.get_point_by_id(pt_id).explored == False

def test_can_get_unexplored_points(temp_db, zr_hcp_cnf):
    pt_id = temp_db.add_point(zr_hcp_cnf)
    all_nb_ids, _ = explore_pt(temp_db, pt_id)
    for i in all_nb_ids[:10]:
        temp_db.mark_point_explored(i)
    
    all_unexplored_pts = temp_db.get_all_unexplored_points()
    unexplored_ids = [pt.id for pt in all_unexplored_pts]
    assert len(unexplored_ids) == len(all_nb_ids[10:])
    assert set(unexplored_ids) == set(all_nb_ids[10:])

    all_unexplored_pts = temp_db.get_all_explored_points()
    explored_ids = [pt.id for pt in all_unexplored_pts]
    assert len(explored_ids) == len(all_nb_ids[:10])
    assert set(explored_ids) == set(all_nb_ids[:10])

def test_can_set_point_value(zr_hcp_cnf, temp_db):
    pt_id = temp_db.add_point(zr_hcp_cnf)

    pt_val = temp_db.get_point_value(pt_id)
    assert pt_val is None

    v = 2
    temp_db.set_point_value(pt_id, v)
    assert temp_db.get_point_value(pt_id) == v

    v = 2.5
    temp_db.set_point_value(pt_id, v)
    assert temp_db.get_point_value(pt_id) == v

def test_can_get_points_batch_by_ids(zr_hcp_cnf, temp_db: CrystalMapStore):
    nbs = find_neighbors(zr_hcp_cnf)
    temp_db.bulk_insert_points(nbs)

    targets = nbs[:8]
    retrieval_ids = temp_db.get_point_ids(targets)

    retrieved = temp_db.get_points_by_ids(retrieval_ids)
    assert len(retrieved) == len(retrieval_ids)
    assert len(retrieved) == len(targets)
    assert set(targets) == set([pt.cnf for pt in retrieved])

    for id, pt in zip(retrieval_ids, retrieved):
        assert temp_db.get_point_by_id(id).cnf == pt.cnf
