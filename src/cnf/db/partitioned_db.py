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
        
def get_partition_number(cnf: CrystalNormalForm, total_num_partitions: int):
    return hash(cnf) % total_num_partitions

class PartitionedDB():

    def __init__(self,
                 db_dir: str,
                 search_id: int,
                 partition_range: list[int] = None):
        self._db_dir = db_dir
        self.db_metadata = load_meta_file(db_dir)
        self.search_id = search_id
        self.search_metadata = [s for s in self.db_metadata.search_processes if s.search_id == search_id][0]
        
        directory = pathlib.Path(self._db_dir)

        partition_files = sorted(list(directory.glob(f"*{PARTITION_SUFFIX}")))
        control_file = os.path.join(db_dir, META_DB_NAME)
        
        self.meta_store = MetaStore.from_file(control_file)
        self.num_partitions = len(partition_files)

        if partition_range is None:
            partition_range = list(range(self.num_partitions))
        
        self.partition_range = partition_range
        self.partition_map = {}

        for i, f in enumerate(partition_files):
            if i in self.partition_range:
                search_store = SearchProcessStore.from_file(f)
                map_store = CrystalMapStore.from_file(f)
                self.partition_map[i] = {
                    "search_store": search_store,
                    "map_store": map_store
                }
    
    def partition_cnfs(self, cnfs: list[CrystalNormalForm]) -> dict[int, list[CrystalNormalForm]]:
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
        return get_partition_number(cnf, self.num_partitions)
    
    def get_search_store(self, cnf: CrystalNormalForm) -> SearchProcessStore:
        return self.partition_map[self.get_partition_idx(cnf)]["search_store"]

    def get_map_store(self, cnf: CrystalNormalForm) -> CrystalMapStore:
        return self.partition_map[self.get_partition_idx(cnf)]["map_store"]
    
    def get_search_store_by_idx(self, idx) -> SearchProcessStore:
        return self.partition_map[idx]["search_store"]
    
    def get_map_store_by_idx(self, idx) -> CrystalMapStore:
        return self.partition_map[idx]["map_store"]
    
    def get_random_partition_idx(self):
        return random.choice(self.partition_range)

    def get_current_water_level(self):
        """Get current water level across ALL partitions.

        Returns:
            Minimum frontier energy across all partitions, or None if no frontier points
        """
        return self.meta_store.get_global_water_level(self.search_id)
    
    def is_search_complete(self):
        return self.meta_store.is_search_complete(self.search_id)
    
    def sync_control_water_level(self):
        for i in self.partition_range:
            search_store = self.get_search_store_by_idx(i)
            partition_min = search_store.get_min_frontier_energy(self.search_id)
            self.meta_store.update_min_water_level(self.search_id, i, partition_min)
    
    def sync_search_completion_status(self):
        found_it = False
        for pidx in self.partition_range:
            found_endpt_ids = self.get_search_store_by_idx(pidx).get_located_endpoint_ids(self.search_id)
            if len(found_endpt_ids) > 0:
                found_it = True
                self.meta_store.set_search_status(self.search_id, True)
        if not found_it:
            self.meta_store.set_search_status(self.search_id, False)

