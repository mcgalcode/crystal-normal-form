import sqlite3
from . import queries, constants
from dataclasses import dataclass
import json

### Database design notes:
#
# Table 1: point
# id: int, primary key
# cnf_str: JSON list (the actual CNF coordinate string)
# value: float (the energy value for pathfinding, or e.g. a distance metric)
# external_id: nullable, str (a field used to point to e.g. fireworks db entry for the calculation)
# explored: bool

#
# Table 2: edge
# source_id: int
# target_id: int

@dataclass
class CNFMetadata():

    delta: int
    xi: float
    element_list: list[str]

class CNFStore():

    @classmethod
    def setup(cls, dbfname: str, xi: float, delta: int, element_list: list[str]):
        conn = sqlite3.connect(dbfname)
        cur = conn.cursor()
        cur.execute(queries.create_point_table)
        cur.execute(queries.create_edge_table)
        cur.execute(queries.create_metadata_table)

        el_str = json.dumps(element_list)
        cur.execute(
            queries.set_metadata,
            (delta, xi, el_str)
        )
        conn.commit()
        return cls(dbfname)


    def __init__(self, dbfname: str):
        self.db_name = dbfname
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()

        query = queries.table_exists.format(table_name=constants.POINT_TABLE_NAME)
        res = self.cursor.execute(query)
        if res.fetchone() is None:
            raise ValueError(f"Tried to instantiate campaign store from uninitialized DB file: {dbfname}")
    
    def get_metadata(self):
        res = self.cursor.execute(queries.get_metadata())
        vals = res.fetchone()
        return CNFMetadata(delta=vals[0], xi=vals[1], element_list=json.loads(vals[2]))
