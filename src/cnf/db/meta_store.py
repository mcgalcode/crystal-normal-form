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
        
    def create_partition_entry(self, partition_id):
        self.cursor.execute(
            meta_queries.create_partition_entry,
            ([partition_id])
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def get_global_water_level(self):
        result = self.cursor.execute(
            meta_queries.get_global_water_level
        )
        self.conn.commit()
        return result.fetchone()[0]
    
    def update_min_water_level(self, partition_id: int, energy_val: float):
        res = self.cursor.execute(
            meta_queries.update_min_water_level,
            ([energy_val, partition_id])
        )
        self.conn.commit()
        return res