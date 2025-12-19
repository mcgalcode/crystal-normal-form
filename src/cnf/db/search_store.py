import sqlite3

from .queries import constants
from .queries import general as general_queries
from .queries import search_process as sp_queries
from .db_adapter import DBAdapter
from .base import BaseStore
from ..crystal_normal_form import CrystalNormalForm
from .crystal_map_store import CrystalMapStore
from .utilities import cnf_pt_from_row, CNFPoint, cnf_to_str, cnf_from_str

class SearchProcessStore(BaseStore):

    def __init__(self, adapter: DBAdapter):
        super().__init__(adapter)

        query = general_queries.table_exists.format(table_name=constants.POINT_TABLE_NAME)
        res = self.cursor.execute(query)
        if res.fetchone() is None:
            raise ValueError(f"Tried to instantiate CrystalMapStore from uninitialized DB file: {adapter.db_filename}")
        
        self._map_store = CrystalMapStore(self.adapter)
        self.metadata = self._map_store.metadata
    
    def create_search_process(self, description: str):
        self.cursor.execute(
            sp_queries.insert_search_process,
            ([description])
        )
        self.conn.commit()
        return self.cursor.lastrowid
    
    def add_search_start_point(self, search_id: int, start_point_id: int):
        self.cursor.execute(
            sp_queries.insert_search_start_point,
            ([search_id, start_point_id])
        )
        self.conn.commit()
        return self.cursor.lastrowid
    
    def add_search_end_point(self, search_id: int, end_point_id: int):
        self.cursor.execute(
            sp_queries.insert_search_end_point,
            ([search_id, end_point_id])
        )
        self.conn.commit()
        return self.cursor.lastrowid
            
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
            sp_queries.upsert_search_point_status_closed,
            ([search_id, point_id])
        )
        self.conn.commit()
        return point_id
    
    def remove_point_from_search(self, search_id: int, cnf: CrystalNormalForm):
        pt_id = self._map_store.get_point_ids([cnf])[0]
        return self.remove_point_from_search_by_id(search_id, pt_id)

    def remove_point_from_search_by_id(self, search_id: int, point_id: int):
        self.cursor.execute(
            sp_queries.delete_search_point_status,
            ([search_id, point_id])
        )
        self.conn.commit()
        return point_id

    def add_to_search_frontier(self, search_id: int, cnf: CrystalNormalForm):
        pt_id = self._map_store.get_point_ids([cnf])[0]
        return self.add_to_search_frontier_by_id(search_id, pt_id)

    def add_to_search_frontier_by_id(self, search_id: int, point_id: int):
        self.cursor.execute(
            sp_queries.upsert_search_point_status_open,
            ([search_id, point_id])
        )
        self.conn.commit()
        return point_id
    
    def bulk_add_to_search_frontier_by_id(self, search_id: int, point_ids: int):
        rows = [(search_id, pid) for pid in point_ids]
        self.cursor.executemany(
            sp_queries.upsert_search_point_status_open,
            rows
        )
        self.conn.commit()
        return point_ids

    def get_searched_cnfs_in_search(self, search_id: int):
        pts = self.get_searched_points_in_search(search_id)
        return [pt.cnf for pt in pts]

    def get_searched_points_in_search(self, search_id: int):
        res = self.cursor.execute(
            sp_queries.select_searched_points,
            ([search_id])
        )
        rows = res.fetchall()
        return [cnf_pt_from_row(r, self.metadata.delta, self.metadata.xi, self.metadata.element_list) for r in rows]
    
    def get_searched_ids_intersecting_with(self, search_id: int, id_set: list[int]):
        res = self.cursor.execute(
            sp_queries.select_searched_ids_scoped_by_id(id_set),
            ([search_id, *id_set])
        )
        rows = res.fetchall()
        return [r[0] for r in rows]

    def get_min_frontier_energy(self, search_id: int):
        """Get the minimum energy value among frontier points.

        Args:
            search_id: The search process ID

        Returns:
            Minimum energy value, or None if no frontier points have energy
        """
        res = self.cursor.execute(
            sp_queries.select_min_frontier_energy,
            ([search_id])
        )
        result = res.fetchone()
        return result[0] if result and result[0] is not None else None
    
    def get_frontier_cnfs_in_search(self, search_id: int, limit: int = 100, max_energy = None):
        return [pt.cnf for pt in self.get_frontier_points_in_search(search_id, limit, max_energy)]

    def get_frontier_points_in_search(self, search_id: int, limit: int = 100, max_energy=None):
        """Get frontier points for a search, ordered by energy (lowest first).

        Args:
            search_id: The search process ID
            limit: Maximum number of frontier points to return (default: 100)
                   This limits how many points we check, trading off optimality
                   for query speed. Lower = faster but may wait more often.
            max_energy: Optional maximum energy threshold. Only returns points
                       with energy <= max_energy (for water-filling algorithm)
        """
        if max_energy is not None:
            res = self.cursor.execute(
                sp_queries.select_frontier_points_with_max_energy,
                ([search_id, max_energy, limit])
            )
        else:
            res = self.cursor.execute(
                sp_queries.select_frontier_points,
                ([search_id, limit])
            )
        rows = res.fetchall()
        return [cnf_pt_from_row(r, self.metadata.delta, self.metadata.xi, self.metadata.element_list) for r in rows]
    
    def get_frontier_point_ids(self, search_id: int):
        res = self.cursor.execute(
            sp_queries.select_frontier_point_ids,
            ([search_id])
        )
        rows = res.fetchall()
        return [r[0] for r in rows]

    def get_endpoint_ids_in_frontier(self, search_id: int):
        res = self.cursor.execute(
            sp_queries.select_endpt_ids_in_frontier,
            ([search_id, search_id])
        )
        rows = res.fetchall()
        return [r[0] for r in rows]
    
    def get_and_empty_incoming_points(self, search_id: int):
        res = self.cursor.execute(
            sp_queries.select_all_incoming_points,
            ([search_id])
        )
        full_result = res.fetchall()
        cnfs = [cnf_from_str(row[0], self.metadata.xi, self.metadata.delta, self.metadata.element_list) for row in full_result]
        self.cursor.execute(
            sp_queries.delete_all_incoming_points,
            ([search_id])
        )
        self.conn.commit()
        return cnfs
    
    def add_incoming_point(self, search_id: int, incoming_point: CrystalNormalForm):
        cnf_str = cnf_to_str(incoming_point)
        self.cursor.execute(
            sp_queries.insert_incoming_point,
            ([search_id, cnf_str])
        )
        self.conn.commit()
    
    def bulk_add_incoming_points(self, search_id: int, cnfs: list[CrystalNormalForm]):
        params = [(search_id, cnf_to_str(cnf)) for cnf in cnfs]
        self.cursor.executemany(
            sp_queries.insert_incoming_point,
            params
        )
        self.conn.commit()
        return self.cursor.rowcount