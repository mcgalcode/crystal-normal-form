import sqlite3
from .queries import constants
from .queries import crystal_map as queries
from .queries import general as general_queries
from .db_adapter import DBAdapter
from .base import BaseStore
from dataclasses import dataclass
import json
from ..crystal_normal_form import CrystalNormalForm
from .utilities import cnf_from_str, cnf_to_str, cnf_pt_from_row, CNFPoint


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


class CrystalMapStore(BaseStore):

    def __init__(self, adapter: DBAdapter):
        super().__init__(adapter)
        query = general_queries.table_exists.format(table_name=constants.POINT_TABLE_NAME)
        res = self.cursor.execute(query)
        if res.fetchone() is None:
            raise ValueError(f"Tried to instantiate CrystalMapStore from uninitialized DB file: {self.adapter.db_filename}")
    
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
        if self.cursor.rowcount > 0:
            return self.cursor.lastrowid
        
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
    
    def get_all_unexplored_points(self):
        res = self.cursor.execute(
            queries.get_all_unexplored_pts
        )
        rows = res.fetchall()
        return [cnf_pt_from_row(r, self.metadata.delta, self.metadata.xi, self.metadata.element_list) for r in rows]
    
    def get_all_explored_points(self):
        res = self.cursor.execute(
            queries.get_all_explored_pts
        )
        rows = res.fetchall()
        return [cnf_pt_from_row(r, self.metadata.delta, self.metadata.xi, self.metadata.element_list) for r in rows]
    
    def get_point_ids(self, points: list[CrystalNormalForm]):
        cnf_strs = [cnf_to_str(p) for p in points]
        res = self.cursor.execute(
            queries.get_points_ids(cnf_strs),
            (cnf_strs)
        )
        rows = res.fetchall()
        ids = []
        for c in cnf_strs:
            filtered_rows = [r for r in rows if r[1] == c]
            if len(filtered_rows) == 0:
                raise ValueError(f"No row in CNFStore found for CNF {c}")
            ids.append(filtered_rows[0][0])
        return ids


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
        ids = self.get_point_ids([pt1, pt2])
        return self.add_connection_by_ids(*ids)
    
    def add_connection_to_target_cnf(self, source_id: int, target_cnf: CrystalNormalForm):
        res = self.cursor.execute(
            queries.create_connection_to_target_cnf,
            ([source_id, cnf_to_str(target_cnf)])
        )
        self.conn.commit()
        return res.lastrowid is not None
    
    def add_connection_by_ids(self, id1: int, id2: int):
        res1 = self.cursor.execute(
            queries.create_connection,
            ([id1, id2])
        )
        res2 = self.cursor.execute(
            queries.create_connection,
            ([id2, id1])
        )
        self.conn.commit()
        return (res1.lastrowid is not None or res2.lastrowid is not None)

    def remove_connection(self, pt1: CrystalNormalForm, pt2: CrystalNormalForm):
        ids = self.get_point_ids([pt1, pt2])
        return self.remove_connection_by_ids(ids[0], ids[1])

    def remove_connection_by_ids(self, id1: int, id2: int):
        res = self.cursor.execute(
            queries.delete_connection_by_ids,
            ([id1, id2, id1, id2])
        )
        self.conn.commit()
        return True

    def connection_exists(self, pt1: CrystalNormalForm, pt2: CrystalNormalForm):
        ids = self.get_point_ids([pt1, pt2])
        return self.connection_exists_by_id(*ids)
        
    def connection_exists_by_id(self, id1: int, id2: int):
        res = self.cursor.execute(
            queries.get_connection_by_ids,
            ([id1, id2, id1, id2])
        )
        rows = res.fetchmany()
        return len(rows) > 0
    
    def get_neighbors(self, pt_id: int):
        res = self.cursor.execute(
            queries.select_neighbors,
            ([pt_id, pt_id])
        )
        rows = res.fetchall()
        return [cnf_pt_from_row(r, self.metadata.delta, self.metadata.xi, self.metadata.element_list) for r in rows]
    
    def mark_point_explored(self, pt_id: int):
        self.cursor.execute(
            queries.mark_point_explored,
            ([pt_id])
        )
        self.conn.commit()
        return pt_id

    def mark_point_unexplored(self, pt_id: int):
        self.cursor.execute(
            queries.mark_point_unexplored,
            ([pt_id])
        )
        self.conn.commit()
        return pt_id
    
    def lock_point(self, pt_id: int) -> bool:
        """Try to lock a point. Returns True if lock was acquired, False if already locked."""
        self.cursor.execute(
            queries.add_lock_for_point,
            ([pt_id])
        )
        self.conn.commit()
        # Check if a row was actually inserted (rowcount > 0 means successful insert)
        return self.cursor.rowcount > 0
    
    def unlock_point(self, pt_id: int):
        self.cursor.execute(
            queries.rm_lock_for_point,
            ([pt_id])
        )
        self.conn.commit()
    
    def is_point_locked(self, pt_id: int):
        res = self.cursor.execute(
            queries.get_lock_for_point,
            ([pt_id])
        )
        rows = res.fetchall()
        return len(rows) > 0
    
    def set_point_value(self, pt_id: int, value: float):
        self.cursor.execute(
            queries.set_value_for_point,
            ([value, pt_id])
        )
        self.conn.commit()
        return pt_id
    
    def get_point_value(self, pt_id: int):
        res = self.cursor.execute(
            queries.get_point_value,
            ([pt_id])
        )
        rows = res.fetchall()
        return rows[0][0]

    def __contains__(self, item):
        pass
    
    def __len__(self):
        pass

