import pytest
import tempfile

from cnf.db.setup import setup_meta_db
from cnf.db.meta_store import MetaStore

def test_can_setup_meta_db():

    with tempfile.NamedTemporaryFile() as t:
        setup_meta_db(t.name)
        MetaStore.from_file(t.name)
        