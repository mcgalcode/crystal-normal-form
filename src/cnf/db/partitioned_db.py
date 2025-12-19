import pathlib
import os
import random
from .search_store import SearchProcessStore
from .crystal_map_store import CrystalMapStore
from .meta_store import MetaStore
from .meta_file import load_meta_file
from .constants import PARTITION_SUFFIX, META_DB_NAME

from ..crystal_normal_form import CrystalNormalForm
from .utilities import CNFPoint
        

class PartitionedDB():

    def __init__(self, db_dir: str):
        self._db_dir = db_dir
        self.metadata = load_meta_file(db_dir)
        directory = pathlib.Path(self._db_dir)

        partition_files = sorted(list(directory.glob(f"*{PARTITION_SUFFIX}")))
        control_file = os.path.join(db_dir, META_DB_NAME)
        self.meta_store = MetaStore.from_file(control_file)
        self.num_partitions = len(partition_files)

        self.partition_map = {}
        for i, f in enumerate(partition_files):
            
            search_store = SearchProcessStore.from_file(f)
            map_store = CrystalMapStore.from_file(f)
            self.partition_map[i] = {
                "search_store": search_store,
                "map_store": map_store
            }
    
    def partition_cnfs(self, cnfs: list[CrystalNormalForm]):
        partitions = { i: [] for i in range(self.num_partitions) }
        for c in cnfs:
            partition = self.get_partition_idx(c)
            partitions[partition].append(c)
        return partitions

    def add_point(self, pt: CrystalNormalForm):
        return self.get_map_store(pt).add_point(pt)
    
    def get_point_by_cnf(self, pt: CrystalNormalForm):
        return self.get_map_store(pt).get_point_by_cnf(pt)

    def get_partition_idx(self, cnf: CrystalNormalForm):
        return hash(cnf) % self.num_partitions
    
    def get_search_store(self, cnf: CrystalNormalForm) -> SearchProcessStore:
        return self.partition_map[self.get_partition_idx(cnf)]["search_store"]

    def get_map_store(self, cnf: CrystalNormalForm) -> CrystalMapStore:
        return self.partition_map[self.get_partition_idx(cnf)]["map_store"]
    
    def get_search_store_by_idx(self, idx) -> SearchProcessStore:
        return self.partition_map[idx]["search_store"]
    
    def get_map_store_by_idx(self, idx) -> CrystalMapStore:
        return self.partition_map[idx]["map_store"]
    
    def get_random_partition_idx(self):
        return random.randint(0, self.num_partitions - 1)

    def get_current_water_level(self, search_id: int):
        """Get current water level across ALL partitions.

        Queries all partitions for their minimum frontier energy and returns
        the global minimum. This ensures we have the true lowest energy point
        on the frontier for proper water-filling behavior.

        Args:
            search_id: Search process ID

        Returns:
            Minimum frontier energy across all partitions, or None if no frontier points
        """
        min_energy = None

        for idx in range(self.num_partitions):
            search_store = self.get_search_store_by_idx(idx)
            partition_min = search_store.get_min_frontier_energy(search_id)
            if partition_min is not None:
                min_energy = min(min_energy, partition_min) if min_energy is not None else partition_min

        return min_energy
    
    def sync_control_water_level(self, search_id: int):
        for i in range(self.num_partitions):
            search_store = self.get_search_store_by_idx(i)
            partition_min = search_store.get_min_frontier_energy(search_id)
            self.meta_store.update_min_water_level(search_id, i, partition_min)


