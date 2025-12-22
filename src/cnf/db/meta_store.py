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
        res = res.fetchone()
        if res is None:
            return False
        return res[0] == 1

    def update_partition_stats(self, search_id: int, partition_number: int, stats: dict):
        """Update aggregate statistics for a partition.

        Args:
            search_id: The search process ID
            partition_number: The partition number
            stats: Dictionary containing:
                - total_points: Total points in partition
                - points_with_energy: Points with computed energy
                - explored_points: Points that have been explored
                - total_edges: Total edges in partition
                - frontier_points: Points in frontier for this search
                - searched_points: Points marked as searched for this search
                - inbox_size: Number of points in incoming mailbox
                - min_energy: Minimum energy in partition (or None)
                - max_energy: Maximum energy in partition (or None)
                - max_searched_energy: Maximum energy among searched points (or None)
        """
        self.cursor.execute(
            meta_queries.update_partition_stats,
            (
                stats.get('total_points', 0),
                stats.get('points_with_energy', 0),
                stats.get('explored_points', 0),
                stats.get('total_edges', 0),
                stats.get('frontier_points', 0),
                stats.get('searched_points', 0),
                stats.get('inbox_size', 0),
                stats.get('min_energy'),
                stats.get('max_energy'),
                stats.get('max_searched_energy'),
                partition_number,
                search_id
            )
        )
        self.conn.commit()

    def get_partition_stats(self, search_id: int, partition_number: int):
        """Get aggregate statistics for a specific partition.

        Returns:
            Dictionary with stats or None if not found
        """
        res = self.cursor.execute(
            meta_queries.get_partition_stats,
            (partition_number, search_id)
        )
        row = res.fetchone()
        if row is None:
            return None

        return {
            'total_points': row[0],
            'points_with_energy': row[1],
            'explored_points': row[2],
            'total_edges': row[3],
            'frontier_points': row[4],
            'searched_points': row[5],
            'inbox_size': row[6],
            'min_energy': row[7],
            'max_energy': row[8],
            'max_searched_energy': row[9],
            'stats_updated_at': row[10]
        }

    def get_all_partition_stats(self, search_id: int):
        """Get aggregate statistics for all partitions in a search.

        Returns:
            List of dictionaries with partition stats
        """
        res = self.cursor.execute(
            meta_queries.get_all_partition_stats,
            (search_id,)
        )
        rows = res.fetchall()

        results = []
        for row in rows:
            results.append({
                'partition_number': row[0],
                'total_points': row[1],
                'points_with_energy': row[2],
                'explored_points': row[3],
                'total_edges': row[4],
                'frontier_points': row[5],
                'searched_points': row[6],
                'inbox_size': row[7],
                'min_energy': row[8],
                'max_energy': row[9],
                'max_searched_energy': row[10],
                'min_frontier_energy': row[11],
                'stats_updated_at': row[12]
            })

        return results
