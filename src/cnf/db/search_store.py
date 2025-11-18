import sqlite3

from .queries import constants
from .queries import general as general_queries
from .queries import search_process as sp_queries
from .db_adapter import DBAdapter
from .base import BaseStore
from ..crystal_normal_form import CrystalNormalForm
from .crystal_map_store import CrystalMapStore
from .utilities import cnf_pt_from_row

class SearchProcessStore(BaseStore):

    def __init__(self, adapter: DBAdapter):
        super().__init__(adapter)

        query = general_queries.table_exists.format(table_name=constants.POINT_TABLE_NAME)
        res = self.cursor.execute(query)
        if res.fetchone() is None:
            raise ValueError(f"Tried to instantiate CrystalMapStore from uninitialized DB file: {adapter.db_filename}")
        
        self._map_store = CrystalMapStore(self.adapter)
        self.metadata = self._map_store.metadata
    
    def create_search_process(self,
                              description: str,
                              start_points: list[CrystalNormalForm],
                              end_points: list[CrystalNormalForm]):
        start_ids = self._map_store.get_point_ids(start_points)
        end_ids = self._map_store.get_point_ids(end_points)
        
        # Begin transaction for adding process and endpoints
        self.cursor.execute("BEGIN")
        self.cursor.execute(
            sp_queries.insert_search_process,
            ([description])
        )

        new_search_proc_id = self.cursor.lastrowid

        for sid in start_ids:
            self.cursor.execute(
                sp_queries.insert_search_start_point,
                ([new_search_proc_id, sid])
            )
        
        for eid in end_ids:
            self.cursor.execute(
                sp_queries.insert_search_end_point,
                ([new_search_proc_id, eid])
            )
        self.conn.commit()
        return new_search_proc_id

    def get_search_endpoints(self, search_id: int):
        res = self.cursor.execute(
            sp_queries.select_search_end_points,
            ([search_id])
        )
        rows = res.fetchall()
        return [cnf_pt_from_row(r, self.metadata.delta, self.metadata.xi, self.metadata.element_list) for r in rows]

    def get_search_startpoints(self, search_id: int):
        res = self.cursor.execute(
            sp_queries.select_search_start_points,
            ([search_id])
        )
        rows = res.fetchall()
        return [cnf_pt_from_row(r, self.metadata.delta, self.metadata.xi, self.metadata.element_list) for r in rows]
    
    def mark_point_searched(self, search_id: int, cnf: CrystalNormalForm):
        pt_id = self._map_store.get_point_ids([cnf])[0]
        return self.mark_point_searched_by_id(search_id, pt_id)

    def mark_point_searched_by_id(self, search_id: int, point_id: int):
        self.cursor.execute(
            sp_queries.mark_point_searched,
            ([search_id, point_id])
        )
        self.conn.commit()
        return point_id

    def add_to_search_frontier(self, search_id: int, cnf: CrystalNormalForm):
        pt_id = self._map_store.get_point_ids([cnf])[0]
        return self.add_to_search_frontier_by_id(search_id, pt_id)

    def add_to_search_frontier_by_id(self, search_id: int, point_id: int):
        res = self.cursor.execute(
            sp_queries.add_point_to_frontier,
            ([search_id, point_id])
        )
        self.conn.commit()
        return point_id

    def remove_from_search_frontier(self, search_id: int, cnf: CrystalNormalForm):
        pt_id = self._map_store.get_point_ids([cnf])[0]
        return self.remove_from_search_frontier_by_id(search_id, pt_id)        

    def remove_from_search_frontier_by_id(self, search_id: int, point_id: int):
        self.cursor.execute(
            sp_queries.rm_point_from_frontier,
            ([search_id, point_id])
        )
        self.conn.commit()

    def get_searched_points_in_search(self, search_id: int):
        res = self.cursor.execute(
            sp_queries.select_searched_points,
            ([search_id])
        )
        rows = res.fetchall()
        return [cnf_pt_from_row(r, self.metadata.delta, self.metadata.xi, self.metadata.element_list) for r in rows]

    def get_frontier_points_in_search(self, search_id: int):
        res = self.cursor.execute(
            sp_queries.select_frontier_points,
            ([search_id])
        )
        rows = res.fetchall()
        return [cnf_pt_from_row(r, self.metadata.delta, self.metadata.xi, self.metadata.element_list) for r in rows]
    
    def get_frontier_point_ids(self, search_id: int):
        res = self.cursor.execute(
            sp_queries.select_frontier_point_ids,
            ([search_id])
        )
        rows = res.fetchall()
        return rows
    
    def get_unsearched_neighbors_with_lock_info(self, search_id: int, pt_id: int):
        res = self.cursor.execute(
            sp_queries.select_unsearched_neighbors_w_lock,
            ([pt_id, search_id, search_id, pt_id, search_id, search_id])
        )
        rows = res.fetchall()
        cnfs = [cnf_pt_from_row(r, self.metadata.delta, self.metadata.xi, self.metadata.element_list) for r in rows]
        lock_info = {row[0]: row[-1] for row in rows}
        return cnfs, lock_info

    def get_endpoint_ids_in_frontier(self, search_id: int):
        res = self.cursor.execute(
            sp_queries.select_endpt_ids_in_frontier,
            ([search_id, search_id])
        )
        rows = res.fetchall()
        return [r[0] for r in rows]