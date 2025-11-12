import pytest
from cnf.db.cnf_store import CNFStore
from cnf import CrystalNormalForm
import tempfile


@pytest.fixture
def zr_hcp_cnf(zr_hcp_mp):
    xi = 1.5
    delta = 10
    return CrystalNormalForm.from_pmg_struct(zr_hcp_mp, xi, delta)

@pytest.fixture(scope='function')
def temp_db(zr_hcp_cnf):
    els = zr_hcp_cnf.elements
    with tempfile.NamedTemporaryFile('w') as tf:
        cs = CNFStore.setup(tf.name, zr_hcp_cnf.xi, zr_hcp_cnf.delta, els)
        yield cs

def test_cannot_instantiate_with_bad_db():
    with tempfile.NamedTemporaryFile('w') as tf:
        fname = tf.name
        with pytest.raises(ValueError) as captured_excep:
            CNFStore(fname)
        
        assert "Tried to instantiate campaign store from uninitialized DB file:" in captured_excep.value.__repr__()

def test_can_instantiate_after_setup(temp_db, zr_hcp_cnf):
    cs2 = CNFStore(temp_db.db_filename)
    assert cs2 is not None

    metadata = cs2.get_metadata()
    assert metadata.delta == zr_hcp_cnf.delta
    assert metadata.xi == zr_hcp_cnf.xi
    assert metadata.element_list == zr_hcp_cnf.elements

def test_can_add_and_retrieve_row(zr_hcp_cnf, temp_db: CNFStore):
    temp_db.add_point(zr_hcp_cnf)
    
    result = temp_db.get_point_by_cnf(zr_hcp_cnf)
    assert result.cnf == zr_hcp_cnf
    assert result.explored == False
    assert result.value is None
    assert result.external_id is None
    assert result.id
    
