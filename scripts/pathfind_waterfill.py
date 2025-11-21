from cnf.search import continue_search_waterfill, continue_search_flood_fill
from cnf.db.search_store import SearchProcessStore
from cnf.db.partitioned_db import PartitionedDB
from cnf.navigation.search_filters import VolumeLimitFilter, AtomOverlapFilter
from cnf.calculation.grace import GraceCalculator
import sys

def main():    
    partition_dir = sys.argv[1]
    if len(sys.argv) > 2:
        max_iters = int(sys.argv[2])
    else:
        max_iters = None
    

    print(f"Using partition directory {partition_dir}...")
    db = PartitionedDB(partition_dir)
    search_proc_id = 1
    search_store = db.get_search_store_by_idx(db.get_random_partition_idx())
    endpts = search_store.get_search_endpoints(search_proc_id)
    end_cnfs = [pt.cnf for pt in endpts]
    start_pts = search_store.get_search_startpoints(search_proc_id)
    start_cnfs = [pt.cnf for pt in start_pts]

    vol_filter = VolumeLimitFilter.from_endpoint_structs(
        [cnf.reconstruct() for cnf in start_cnfs + end_cnfs],
        0.7,
        1.3
    )

    atomic_overlap_filter = AtomOverlapFilter(0.8)

    filters = [
        vol_filter,
        atomic_overlap_filter
    ]

    continue_search_waterfill(1, partition_dir, GraceCalculator(), filters, log_lvl=2, max_iters=max_iters)

if __name__ == "__main__":
    main()