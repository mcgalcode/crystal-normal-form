import pytest
from cnf.db.cnf_store import CNFStore
import tempfile

def test_cannot_instantiate_with_bad_db():
    with tempfile.NamedTemporaryFile('w') as tf:
        fname = tf.name
        with pytest.raises(ValueError) as captured_excep:
            CNFStore(fname)
        
        assert "Tried to instantiate campaign store from uninitialized DB file:" in captured_excep.value.__repr__()

def test_can_instantiate_after_setup():
    xi = 1.5
    delta = 10
    elements = ["Li", "Li"]
    with tempfile.NamedTemporaryFile('w') as tf:

        cs = CNFStore.setup(tf.name, xi, delta, elements)
        assert cs is not None

        cs2 = CNFStore(tf.name)
        assert cs2 is not None

        metadata = cs2.get_metadata()
        assert metadata.delta == delta
        assert metadata.xi == xi
        assert metadata.element_list == elements