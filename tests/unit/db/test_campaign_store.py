import pytest
from cnf.db import CampaignStore
import tempfile

def test_cannot_instantiate_with_bad_db():
    with tempfile.NamedTemporaryFile('w') as tf:
        fname = tf.name
        with pytest.raises(ValueError) as captured_excep:
            CampaignStore(fname)
        
        assert "Tried to instantiate campaign store from uninitialized DB file:" in captured_excep.value.__repr__()