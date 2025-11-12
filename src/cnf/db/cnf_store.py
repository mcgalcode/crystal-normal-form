import sqlite3
from . import queries, constants
from dataclasses import dataclass
import json
from ..crystal_normal_form import CrystalNormalForm

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

@dataclass
class CNFPoint():

    id: int
    cnf: CrystalNormalForm
    explored: bool
    external_id: str
    value: float


def cnf_to_str(cnf: CrystalNormalForm):
    return json.dumps([str(c) for c in cnf.coords])

def cnf_pt_from_row(row: tuple, delta: int, xi: float, elements: list[str]):
    cnf_list = json.loads(row[1])
    cnf_list = [int(float(c)) for c in cnf_list]
    cnf = CrystalNormalForm.from_tuple(tuple(cnf_list), elements, xi, delta)
    return CNFPoint(
        id=row[0],
        cnf=cnf,
        explored=row[4],
        external_id=row[2],
        value=row[3]
    )

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
        self.db_filename = dbfname
        self.conn = sqlite3.connect(self.db_filename)
        self.cursor = self.conn.cursor()

        query = queries.table_exists.format(table_name=constants.POINT_TABLE_NAME)
        res = self.cursor.execute(query)
        if res.fetchone() is None:
            raise ValueError(f"Tried to instantiate campaign store from uninitialized DB file: {dbfname}")
    
        self.metadata = self.get_metadata()
    
    def get_metadata(self):
        res = self.cursor.execute(queries.get_metadata)
        vals = res.fetchone()
        return CNFMetadata(delta=vals[0], xi=vals[1], element_list=json.loads(vals[2]))

    def add_point(self, point: CrystalNormalForm):
        cnf_str = cnf_to_str(point)
        res = self.cursor.execute(
            queries.insert_point,
            (cnf_str, None, None, False)
        )
        self.conn.commit()
        return res.fetchall()
    
    def all_points(self):
        pass
    
    def all_node_ids(self):
        pass
    
    def get_point_by_cnf(self, point: CrystalNormalForm):
        cnf_str = cnf_to_str(point)
        res = self.cursor.execute(
            queries.get_point_by_cnf_str,
            ([cnf_str])
        )
        row = res.fetchone()
        if row is None:
            return None
        return cnf_pt_from_row(row, self.metadata.delta, self.metadata.xi, self.metadata.element_list)
    
    def get_point_ids(self, *points: list[CrystalNormalForm]):
        pass

    def get_point_by_id(self, id: int):
        res = self.cursor.execute(
            queries.get_point_by_id,
            ([id])
        )
        row = res.fetchone()
        if row is None:
            return None
        return cnf_pt_from_row(row, self.metadata.delta, self.metadata.xi, self.metadata.element_list)
    
    def remove_point(self, point: CrystalNormalForm):
        cnf_str = cnf_to_str(point)
        res = self.cursor.execute(
            queries.delete_point_by_point,
            ([cnf_str])
        )
        res = res.fetchone()
        self.conn.commit()
        return res

    def add_connection(self, pt1: CrystalNormalForm, pt2: CrystalNormalForm):
        pass

    def add_connection_by_ids(self, id1, id2):
        pass

    def remove_connection(self, pt1: CrystalNormalForm, pt2: CrystalNormalForm):
        pass

    def connection_exists(self, pt1: CrystalNormalForm, pt2: CrystalNormalForm):
        pass
    
    def connection_exists_by_id(self, id1: int, id2: int):
        pass

    def __contains__(self, item):
        pass
    
    def __len__(self):
        pass

