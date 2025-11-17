import sqlite3

from .queries import constants
from .queries import general as general_queries
from .queries import search_process as sp_queries
from .db_adapter import DBAdapter
from .base import BaseStore
from dataclasses import dataclass
import json
from ..crystal_normal_form import CrystalNormalForm

class SearchProcessStore(BaseStore):

    def __init__(self, adapter: DBAdapter):
        super().__init__(adapter)

        query = general_queries.table_exists.format(table_name=constants.POINT_TABLE_NAME)
        res = self.cursor.execute(query)
        if res.fetchone() is None:
            raise ValueError(f"Tried to instantiate CrystalMapStore from uninitialized DB file: {adapter.db_filename}")
    
    def create_search_process(self, cnf1: CrystalNormalForm, cnf2: CrystalNormalForm):
        pass

    def mark_point_searched(self, cnf: CrystalNormalForm):
        pass

    def mark_point_searched_by_id(self, id: int):
        pass

    def add_to_search_frontier(self, cnf: CrystalNormalForm):
        pass

    def add_to_search_frontier_by_id(self, id: int):
        pass

    def remove_from_search_frontier(self, cnf: CrystalNormalForm):
        pass

    def remove_from_search_frontier_by_id(self, id: int):
        pass