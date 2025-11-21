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
    
    def bulk_insert_points(self, points: list[CrystalNormalForm]):
        rows = [(cnf_to_str(p), None, None, False) for p in points]
        res = self.cursor.executemany(
            queries.insert_point,
            rows
        )
        self.conn.commit()
        return res.rowcount
    
    def _cnf_pt_from_row(self, row):
        return cnf_pt_from_row(row, self.metadata.delta, self.metadata.xi, self.metadata.element_list)
        
    def get_point_by_cnf(self, point: CrystalNormalForm):
        cnf_str = cnf_to_str(point)
        res = self.cursor.execute(
            queries.get_point_by_cnf_str,
            ([cnf_str])
        )
        row = res.fetchone()
        if row is None:
            return None
        return self._cnf_pt_from_row(row)
    
    def get_all_unexplored_points(self):
        res = self.cursor.execute(
            queries.get_all_unexplored_pts
        )
        rows = res.fetchall()
        return [self._cnf_pt_from_row(r) for r in rows]
    
    def get_all_explored_points(self):
        res = self.cursor.execute(
            queries.get_all_explored_pts
        )
        rows = res.fetchall()
        return [self._cnf_pt_from_row(r) for r in rows]
    
    def get_point_ids(self, points: list[CrystalNormalForm]):
        pts = self.get_points_by_cnfs(points)
        return [pt.id for pt in pts]
    
    def get_points_by_cnfs(self, cnfs: list[CrystalNormalForm]):
        cnf_strs = [cnf_to_str(p) for p in cnfs]
        res = self.cursor.execute(
            queries.get_points_ids(cnf_strs),
            (cnf_strs)
        )
        rows = res.fetchall()
        pts = [self._cnf_pt_from_row(r) for r in rows]
        pt_lookup = {pt.cnf: pt for pt in pts}
        try:
            return [pt_lookup[c] for c in cnfs]
        except KeyError:
            raise ValueError(f"No row in CNFStore found for CNF!")

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
    

    def bulk_add_edges(self, edge_list: list[tuple[int, int, CrystalNormalForm]]):
        formatted_edge_lists = []
        for edge in edge_list:
            if edge[2] is not None:
                formatted_edge_lists.append((edge[0], edge[1], cnf_to_str(edge[2])))
            else:
                formatted_edge_lists.append(edge)
        res = self.cursor.executemany(
            queries.bulk_insert_edges,
            formatted_edge_lists
        )
        self.conn.commit()
        return res.rowcount
    
    def add_connection_to_target_cnf(self, source_id: int, target_cnf: CrystalNormalForm):
        res = self.cursor.execute(
            queries.create_connection_to_target_cnf,
            ([source_id, cnf_to_str(target_cnf)])
        )
        self.conn.commit()
        return self.cursor.rowcount > 0
    
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
        return self.cursor.rowcount > 0

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
    
    def get_neighbors(self, pt_id: int) -> list[CNFPoint]:
        res = self.cursor.execute(
            queries.select_neighbors,
            ([pt_id, pt_id])
        )
        rows = res.fetchall()
        return [self._cnf_pt_from_row(r) for r in rows]

    def get_neighbor_cnfs(self, pt_id: int) -> list[CrystalNormalForm]:
        """Get all neighbor CNFs for a point.

        This handles both same-partition neighbors (stored with target_id)
        and cross-partition neighbors (stored with target_cnf string).

        Returns a list of CrystalNormalForm objects.
        """
        res = self.cursor.execute(
            queries.select_neighbor_cnfs,
            ([pt_id])
        )
        rows = res.fetchall()
        return [cnf_from_str(row[0], self.metadata.xi, self.metadata.delta, self.metadata.element_list) for row in rows]
    
    def get_nonlocal_neighbor_cnfs(self, pt_id: int):
        res = self.cursor.execute(
            queries.select_nonlocal_cnf_neighbors,
            ([pt_id])
        )
        rows = res.fetchall()
        return [cnf_from_str(row[0], self.metadata.xi, self.metadata.delta, self.metadata.element_list) for row in rows]

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

