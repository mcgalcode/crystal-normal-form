import pathlib
import random
from .search_store import SearchProcessStore
from .crystal_map_store import CrystalMapStore

from ..crystal_normal_form import CrystalNormalForm
from .utilities import CNFPoint
        
DB_PREFIX = "graph_partition"

class PartitionedDB():

    def __init__(self, db_dir: str):
        self._db_dir = db_dir
        directory = pathlib.Path(self._db_dir)

        db_files = sorted(list(directory.glob(f"{DB_PREFIX}*.db")))
        self.num_partitions = len(db_files)

        self.partition_map = {}
        for i, f in enumerate(db_files):
            self.partition_map[i] = {
                "search_store": SearchProcessStore.from_file(f),
                "map_store": CrystalMapStore.from_file(f)
            }

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
        
