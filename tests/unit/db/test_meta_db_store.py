import pytest
import sqlite3
import tempfile

from cnf.db.setup import setup_meta_db
from cnf.db.meta_store import MetaStore

@pytest.fixture
def meta_store():
    with tempfile.NamedTemporaryFile() as t:
        setup_meta_db(t.name)
        yield MetaStore.from_file(t.name)

@pytest.fixture(scope='function')
def store_with_3_partitions():
    with tempfile.NamedTemporaryFile() as t:
        setup_meta_db(t.name)
        store = MetaStore.from_file(t.name)
        for i in range(3):
            store.create_partition_entry(i)
        yield store

def test_can_setup_meta_db():

    with tempfile.NamedTemporaryFile() as t:
        setup_meta_db(t.name)
        MetaStore.from_file(t.name)
        
def test_can_create_partition_entry(meta_store: MetaStore):
    meta_store.create_partition_entry(2)
    with pytest.raises(sqlite3.IntegrityError):
        meta_store.create_partition_entry(2)

def test_can_update_min_water_level(store_with_3_partitions):
    store_with_3_partitions.update_min_water_level(0, 1)
    store_with_3_partitions.update_min_water_level(1, 2)
    store_with_3_partitions.update_min_water_level(2, 3)

    global_lev = store_with_3_partitions.get_global_water_level()

    assert global_lev == 1