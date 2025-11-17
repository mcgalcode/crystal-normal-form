import sqlite3

from .queries import constants
from .queries import general as general_queries
from .queries import search_process as sp_queries
from dataclasses import dataclass
import json
from ..crystal_normal_form import CrystalNormalForm

class SearchProcessStore():

    def __init__(self, dbfname: str):
        self.db_filename = dbfname
        self.conn = sqlite3.connect(self.db_filename)
        self.cursor = self.conn.cursor()

        query = general_queries.table_exists.format(table_name=constants.POINT_TABLE_NAME)
        res = self.cursor.execute(query)
        if res.fetchone() is None:
            raise ValueError(f"Tried to instantiate CrystalMapStore from uninitialized DB file: {dbfname}")
    
    def create_search_process(self, cnf1: CrystalNormalForm, cnf2: CrystalNormalForm):
        pass