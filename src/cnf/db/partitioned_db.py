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
            
            search_store = SearchProcessStore.from_file(f)
            map_store = CrystalMapStore.from_file(f)
            self.partition_map[i] = {
                "search_store": search_store,
                "map_store": map_store
            }

    def get_unsearched_neighbors_and_locks(self, cnf: CrystalNormalForm, search_id: int) -> tuple[list[CNFPoint], dict[int, dict[int, bool]]]:
        source_partition = self.get_partition_idx(cnf)
        source_map_store = self.get_map_store_by_idx(source_partition)
        source_search_store = self.get_search_store_by_idx(source_partition)

        pt_id = source_map_store.get_point_ids([cnf])[0]

        partition_locks = {}

        local_neighbors, local_locks = source_search_store.get_unsearched_neighbors_with_lock_info(search_id, pt_id)
        for ln in local_neighbors:
            ln.partition = source_partition
        partition_locks[source_partition] = local_locks

        all_nbs_with_partitions = local_neighbors

        nonlocal_nb_cnfs = source_map_store.get_nonlocal_neighbor_cnfs(pt_id)
        partitioned_nbs = self.partition_cnfs(nonlocal_nb_cnfs)
        for partition_idx, cnfs in partitioned_nbs.items():
            nb_store = self.get_search_store_by_idx(partition_idx)
            nb_pts, nb_locks = nb_store.get_unsearched_points_by_cnfs_with_lock_info(search_id, cnfs)
            for nb_pt in nb_pts:
                nb_pt.partition = partition_idx
                all_nbs_with_partitions.append(nb_pt)
            partition_locks[partition_idx] = nb_locks
        return all_nbs_with_partitions, partition_locks     
    
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
        
