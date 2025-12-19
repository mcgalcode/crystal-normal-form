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
            store.create_partition_entry(1, i)
        yield store

def test_can_setup_meta_db():

    with tempfile.NamedTemporaryFile() as t:
        setup_meta_db(t.name)
        ms = MetaStore.from_file(t.name)
        ms.create_partition_entry(1, 1)
        
def test_can_create_partition_entry(meta_store: MetaStore):
    sid = 1
    meta_store.create_partition_entry(sid, 2)
    meta_store.create_partition_entry(2, 2)
    with pytest.raises(sqlite3.IntegrityError):
        meta_store.create_partition_entry(sid, 2)

def test_can_update_min_water_level(store_with_3_partitions):
    sid1 = 1
    sid2 = 2
    for i in range(3):
        store_with_3_partitions.create_partition_entry(2, i)    
    store_with_3_partitions.update_min_water_level(sid1, 0, 1)
    store_with_3_partitions.update_min_water_level(sid1, 1, 2)
    store_with_3_partitions.update_min_water_level(sid1, 2, 3)

    global_lev_1 = store_with_3_partitions.get_global_water_level(sid1)
    assert global_lev_1 == 1

    global_lev_2 = store_with_3_partitions.get_global_water_level(sid2)
    assert global_lev_2 == None

    store_with_3_partitions.update_min_water_level(sid2, 0, 0)
    store_with_3_partitions.update_min_water_level(sid2, 1, -1)
    global_lev_2 = store_with_3_partitions.get_global_water_level(sid2)
    assert global_lev_2 == -1

    global_lev_1 = store_with_3_partitions.get_global_water_level(sid1)
    assert global_lev_1 == 1
