import sqlite3

from .queries import constants
from .queries import meta as meta_queries
from .queries import general as general_queries
from .db_adapter import DBAdapter
from .base import BaseStore

class MetaStore(BaseStore):

    def __init__(self, adapter: DBAdapter):
        super().__init__(adapter)

        query = general_queries.table_exists.format(table_name=constants.PARTITION_STATUS_TABLE_NAME)
        res = self.cursor.execute(query)
        if res.fetchone() is None:
            raise ValueError(f"Tried to instantiate MetaStore from uninitialized DB file: {adapter.db_filename}")
        
    def create_partition_entry(self, search_id: int, partition_number: int):
        self.cursor.execute(
            meta_queries.create_partition_entry,
            ([partition_number, search_id])
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def get_global_water_level(self, search_id: int):
        result = self.cursor.execute(
            meta_queries.get_global_water_level,
            ([search_id])
        )
        self.conn.commit()
        return result.fetchone()[0]

    def get_partition_water_level(self, search_id: int, partition_number: int):
        result = self.cursor.execute(
            meta_queries.get_partition_water_level,
            ([partition_number, search_id])
        )
        result = result.fetchone()
        if result is not None:
            return result[0]
        return None
    
    def update_min_water_level(self, search_id, partition_number: int, energy_val: float):
        res = self.cursor.execute(
            meta_queries.update_min_water_level,
            ([energy_val, partition_number, search_id])
        )
        self.conn.commit()
        return res
    
    def create_search_status(self, search_id: int):
        res = self.cursor.execute(
            meta_queries.insert_search_status,
            ([search_id, False])
        )
        self.conn.commit()
        return None
    
    def set_search_status(self, search_id: int, is_complete: bool):
        res = self.cursor.execute(
            meta_queries.update_search_status,
            ([is_complete, search_id])
        )
        self.conn.commit()
        return search_id

    def is_search_complete(self, search_id: int):
        res = self.cursor.execute(
            meta_queries.select_search_status,
            ([search_id])
        )
        is_complete = res.fetchone()[0]
        return is_complete == 1
