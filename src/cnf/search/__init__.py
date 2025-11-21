from ..crystal_normal_form import CrystalNormalForm
from ..db.crystal_map_store import CrystalMapStore
from ..db.partitioned_db import PartitionedDB
from ..db.search_store import SearchProcessStore
from ..db.setup import setup_cnf_db
from ..navigation.neighbor_finder import NeighborFinder
from ..navigation.search_filters import VolumeLimitFilter, SearchFilter
from ..calculation.base_calculator import BaseCalculator
import time
import math
import sqlite3
import pathlib
import json
import os
from datetime import datetime

def instantiate_search(search_description: str,
                       start_cnfs: list[CrystalNormalForm],
                       end_cnfs: list[CrystalNormalForm],
                       store_file: str):
    all_cnfs = start_cnfs + end_cnfs
    xis = [cnf.xi for cnf in all_cnfs]
    if len(set(xis)) > 1:
        raise ValueError("Tried to instantiate search with CNFs having different xi values!")
    
    deltas = [cnf.delta for cnf in all_cnfs]
    if len(set(deltas)) > 1:
        raise ValueError("Tried to instantiate search with CNFs having different delta values!")

    element_list = start_cnfs[0].elements
    for cnf in all_cnfs:
        if cnf.elements != element_list:
            raise ValueError("Tried to instantiate search with CNFs having different element lists!")
        
    crystal_map_store = CrystalMapStore.from_file(store_file)
    for cnf in all_cnfs:
        crystal_map_store.add_point(cnf)

    search_store = SearchProcessStore.from_file(store_file)
    search_id = search_store.create_search_process(search_description)

    start_point_ids = crystal_map_store.get_point_ids(start_cnfs)
    end_point_ids = crystal_map_store.get_point_ids(end_cnfs)

    for sid in start_point_ids:
        print(f"Adding pt {sid} to search start and frontier")
        search_store.add_search_start_point(search_id, sid)
        search_store.add_to_search_frontier_by_id(search_id, sid)
    
    for eid in end_point_ids:
        search_store.add_search_end_point(search_id, eid)
    return search_id


def explore_pt(map_store: CrystalMapStore, pt_id: int, filters: list[SearchFilter] = None, log_lvl=1):
    if filters is None:
        filters = []

    pt = map_store.get_point_by_id(pt_id)

    nbs = NeighborFinder(pt.cnf).find_neighbors()
    
    all_nb_ids = []
    new_nb_ids = []
    
    edges_to_add_to_point_store = []
    
    for nb in nbs:
    
        if len(filters) > 0:
            struct = nb.reconstruct()
            if not all([f.should_add_pt(nb, struct) for f in filters]):
                continue

        nb_id = map_store.add_point(nb)

        if nb_id is not None:
            new_nb_ids.append(nb_id)
        else:
            nb_id = map_store.get_point_by_cnf(nb).id

        edges_to_add_to_point_store.append((pt.id, nb_id, None))

        all_nb_ids.append(nb_id)
    
    map_store.bulk_add_edges(edges_to_add_to_point_store)

    map_store.mark_point_explored(pt.id)

    return all_nb_ids, new_nb_ids


def explore_pt_partition(partition_db: PartitionedDB, point_cnf: CrystalNormalForm, filters: list[SearchFilter] = None, log_lvl=1):
    def _log(msg, lvl = 1):
        if lvl > log_lvl:
            print(msg)
    if filters is None:
        filters = []

    # Timing instrumentation for explore_pt internals
    timings = {
        'find_neighbors': 0.0,
        'filter_checks': 0.0,
        'add_points': 0.0,
        'get_existing_points': 0.0,
        'add_edges': 0.0,
        'mark_explored': 0.0
    }

    point_partition = partition_db.get_partition_idx(point_cnf)
    point_map_store = partition_db.get_map_store(point_cnf)
    pt = point_map_store.get_point_by_cnf(point_cnf)

    # Time: Find neighbors (the actual computation)
    t_start = time.time()
    try:
        nbs = NeighborFinder(pt.cnf).find_neighbors()
    except Exception as e:
        print(f"Ran into a problem with point {pt.id}, {point_cnf}")
        return [], [], {}

    timings['find_neighbors'] += time.time() - t_start

    all_nb_ids = []
    new_nb_ids = []
    existing_nb_ids = []
    filtered_ct = 0
    edges_added = 0
    
    edges_to_add_to_point_store = []   

    for nb in nbs:
        nb_partition = partition_db.get_partition_idx(nb)
        nb_map_store = partition_db.get_map_store_by_idx(nb_partition)

        # Time: Filter checks
        if filters:
            t_start = time.time()
            struct = nb.reconstruct()
            if not all([f.should_add_pt(nb, struct) for f in filters]):
                filtered_ct += 1
                timings['filter_checks'] += time.time() - t_start
                continue
            timings['filter_checks'] += time.time() - t_start

        # Time: Add point to database
        t_start = time.time()
        nb_id = nb_map_store.add_point(nb)
        timings['add_points'] += time.time() - t_start

        if nb_id is not None:
            new_nb_ids.append((nb_partition, nb_id))
        else:
            # Time: Get existing point ID
            t_start = time.time()
            nb_id = nb_map_store.get_point_by_cnf(nb).id
            timings['get_existing_points'] += time.time() - t_start
            existing_nb_ids.append((nb_partition, nb_id))


        if point_partition == nb_partition:
            edges_to_add_to_point_store.append((pt.id, nb_id, None))
        else:
            edges_to_add_to_point_store.append((pt.id, None, nb))


        all_nb_ids.append((nb_partition, nb_id))
    
    # Time: Add edges
    t_start = time.time()
    point_map_store.bulk_add_edges(edges_to_add_to_point_store)
    timings['add_edges'] += time.time() - t_start


    _log(f"Exploration: {len(new_nb_ids)} new neighbors, {len(existing_nb_ids)} existing, {edges_added} edges added, {filtered_ct} filtered")

    # Time: Mark explored
    t_start = time.time()
    point_map_store.mark_point_explored(pt.id)
    timings['mark_explored'] += time.time() - t_start

    return all_nb_ids, new_nb_ids, timings

def continue_search(search_id,
                    cnf_store_file: str,
                    energy_calc: BaseCalculator,
                    search_filters: list[SearchFilter] = None,
                    max_iters: int = None,
                    frontier_limit: int = 100):
    """Continue a water-filling search process.

    Args:
        search_id: ID of the search process
        cnf_store_file: Path to the database file
        energy_calc: Calculator to compute point energies
        search_filters: Optional filters to apply to candidate points
        max_iters: Maximum iterations before stopping (default: infinite)
        frontier_limit: Max frontier points to consider per iteration (default: 100)
                       Lower = faster queries but may wait more if neighbors locked
                       Higher = more options but slower queries
    """
    search_store = SearchProcessStore.from_file(cnf_store_file)
    crystal_map_store = CrystalMapStore.from_file(cnf_store_file)

    if max_iters is None:
        max_iters = math.inf

    num_iters = 0
    while len(search_store.get_endpoint_ids_in_frontier(search_id)) == 0:
        print(f"================ BEGINNING STEP {num_iters} ================")
        if num_iters > max_iters:
            print(f"Reached {num_iters} iterations, quitting...")
            break
        print(f"Endpoint is not yet in the frontier, continuing search...")

        frontier_points = search_store.get_frontier_points_in_search(search_id, limit=frontier_limit)
        print(f"Found {len(frontier_points)} frontier points to consider (limited to {frontier_limit})...")
        # print([fp.id for fp in frontier_points])
        selected_point = None
        for frontier_point in frontier_points: # (lowest energy first)
            print(f"Considering frontier pt ID: {frontier_point.id}, CNF: {frontier_point.cnf.coords}")
            if not frontier_point.explored:
                print(f"Point has not been explored... computing neighbors and adding to the map...")
                _, new_nb_ids = explore_pt(crystal_map_store, frontier_point.id, filters=search_filters)
                print(f"Added {len(new_nb_ids)} neighbors to the map!")

            unsearched_neighbors, locks = search_store.get_unsearched_neighbors_with_lock_info(search_id, frontier_point.id)

            if len(unsearched_neighbors) == 0:
                print(f"Found 0 unsearched neighbors - point ID {frontier_point.id} is exhausted, marking as searched and removing from frontier.")
                search_store.mark_point_searched_by_id(search_id, frontier_point.id)
                search_store.remove_from_search_frontier_by_id(search_id, frontier_point.id)
            
            unlocked_neighbors = [n for n in unsearched_neighbors if not locks[n.id]]
            if len(unlocked_neighbors) > 0:
                selected_point = unlocked_neighbors[0]
                print(f"Identified unlocked candidate point ID {selected_point.id}")
                assert selected_point.id not in [fp.id for fp in frontier_points]
                print(f"Applying lock...")
                lock_acquired = crystal_map_store.lock_point(selected_point.id)
                if not lock_acquired:
                    print(f"Failed to acquire lock (another worker got it first), trying next point...")
                    selected_point = None
                    continue  # Continue to next frontier point
                print(f"Lock acquired successfully!")
                break
            print(f"All neighbors were locked, moving on...")

        if selected_point is None:
            print("Found no unlocked points to compute, sleeping for 5!")
            time.sleep(5)
            continue
        
        print(f"Calculating energy for point ID {selected_point.id}")
        energy = energy_calc.calculate_energy(selected_point.cnf)
        print(f"Energy: {energy}")
        crystal_map_store.set_point_value(selected_point.id, energy)
        search_store.add_to_search_frontier_by_id(search_id, selected_point.id)
        crystal_map_store.unlock_point(selected_point.id)
        print(f"Added point to search frontier and removed lock!")
        num_iters += 1
    
def continue_search_flood_fill(search_id,
                               partitions_dir: str,
                               search_filters: list[SearchFilter] = None,
                               max_iters: int = None,
                               frontier_limit: int = 100,
                               log_lvl=3,
                               timing_reports_dir="timing_reports"):
    """Continue a flood-fill search process. There is no energy involved inthis
    search process. Instead, points on the frontier are exhausted if
    they have been explored.

    Args:
        search_id: ID of the search process
        cnf_store_file: Path to the database file
        energy_calc: Calculator to compute point energies
        search_filters: Optional filters to apply to candidate points
        max_iters: Maximum iterations before stopping (default: infinite)
        frontier_limit: Max frontier points to consider per iteration (default: 100)
                       Lower = faster queries but may wait more if neighbors locked
                       Higher = more options but slower queries
    """
    db = PartitionedDB(partitions_dir)

    if max_iters is None:
        max_iters = math.inf

    num_iters = 0

    # Timing instrumentation
    timing_stats = {
        'get_frontier': 0.0,
        'lock_attempts': 0.0,
        'exploration': 0.0,
        'add_to_frontier': 0.0,
        'mark_searched': 0.0,
        # Detailed breakdown of exploration
        'explore_find_neighbors': 0.0,
        'explore_filter_checks': 0.0,
        'explore_add_points': 0.0,
        'explore_get_existing_points': 0.0,
        'explore_add_edges': 0.0,
        'explore_mark_explored': 0.0,
        'total_iters': 0
    }

    def _log(msg, lvl = 1):
        if lvl > log_lvl:
            print(msg)

    while True:
        _log(f"================ BEGINNING STEP {num_iters} ================")
        if num_iters > max_iters:
            _log(f"Reached {num_iters} iterations, quitting...")
            break
        _log(f"Endpoint is not yet in the frontier, continuing search...")

        # Time: Get frontier points
        t_start = time.time()
        random_read_store_idx = db.get_random_partition_idx()
        random_read_search_store = db.get_search_store_by_idx(random_read_store_idx)
        random_read_map_store = db.get_map_store_by_idx(random_read_store_idx)
        frontier_points = random_read_search_store.get_frontier_points_in_search(search_id, limit=frontier_limit)
        timing_stats['get_frontier'] += time.time() - t_start
        _log(f"Found {len(frontier_points)} frontier points to consider (limited to {frontier_limit})...")

        selected_point = None
        for frontier_point in frontier_points:
            _log(f"Attempting lock on frontier pt ID: {frontier_point.id}, CNF: {frontier_point.cnf.coords}")

            # Time: Lock attempt
            t_start = time.time()
            lock_acquired = random_read_map_store.lock_point(frontier_point.id)
            timing_stats['lock_attempts'] += time.time() - t_start

            if not lock_acquired:
                _log(f"Failed to acquire lock (another worker got it first), trying next point...")
                continue  # Continue to next frontier point
            _log(f"Lock acquired successfully!")
            selected_point = frontier_point

            if not selected_point.explored:
                _log(f"Point has not been explored... computing neighbors and adding to the map...")
                print(selected_point.cnf.coords)
                # Time: Exploration (find neighbors)
                t_start = time.time()
                _, new_nbs, explore_timings = explore_pt_partition(db, selected_point.cnf, filters=search_filters, log_lvl=log_lvl)
                timing_stats['exploration'] += time.time() - t_start

                # Accumulate detailed exploration timings
                timing_stats['explore_find_neighbors'] += explore_timings['find_neighbors']
                timing_stats['explore_filter_checks'] += explore_timings['filter_checks']
                timing_stats['explore_add_points'] += explore_timings['add_points']
                timing_stats['explore_get_existing_points'] += explore_timings['get_existing_points']
                timing_stats['explore_add_edges'] += explore_timings['add_edges']
                timing_stats['explore_mark_explored'] += explore_timings['mark_explored']

                # Time: Add to frontiers
                t_start = time.time()
                for nb in new_nbs:
                    partition, nb_id = nb
                    nb_store = db.get_search_store_by_idx(partition)
                    nb_store.add_to_search_frontier_by_id(search_id, nb_id)
                timing_stats['add_to_frontier'] += time.time() - t_start

                _log(f"Added {len(new_nbs)} neighbors to the map!")
            else:
                _log(f"Point ID {selected_point.id} has already been explored, marking as searched.")

            # Time: Mark as searched
            t_start = time.time()
            random_read_search_store.mark_point_searched_by_id(search_id, selected_point.id)
            random_read_search_store.remove_from_search_frontier_by_id(search_id, selected_point.id)
            timing_stats['mark_searched'] += time.time() - t_start

            break

        num_iters += 1
        timing_stats['total_iters'] = num_iters

        # Print timing stats every 100 iterations
        if num_iters % 100 == 0:
            # Calculate total from high-level categories only (exploration breakdown is already included in exploration)
            total_time = sum(timing_stats[k] for k in ['get_frontier', 'lock_attempts', 'exploration', 'add_to_frontier', 'mark_searched'])
            print(f"\n{'='*60}")
            print(f"TIMING STATS (after {num_iters} iterations)")
            print(f"{'='*60}")

            # High-level categories
            for key in ['get_frontier', 'lock_attempts', 'exploration', 'add_to_frontier', 'mark_searched']:
                pct = (timing_stats[key] / total_time * 100) if total_time > 0 else 0
                print(f"  {key:20s}: {timing_stats[key]:8.2f}s ({pct:5.1f}%)")

            print(f"\n  Exploration breakdown:")
            # Detailed exploration breakdown (percentages relative to exploration time)
            exploration_time = timing_stats['exploration']
            explore_keys = [
                ('find_neighbors', 'explore_find_neighbors'),
                ('filter_checks', 'explore_filter_checks'),
                ('add_points', 'explore_add_points'),
                ('get_existing', 'explore_get_existing_points'),
                ('add_edges', 'explore_add_edges'),
                ('mark_explored', 'explore_mark_explored')
            ]
            for label, key in explore_keys:
                pct = (timing_stats[key] / exploration_time * 100) if exploration_time > 0 else 0
                print(f"    {label:18s}: {timing_stats[key]:8.2f}s ({pct:5.1f}%)")

            print(f"\n  {'TOTAL':20s}: {total_time:8.2f}s")
            print(f"{'='*60}\n")

        if selected_point is None:
            _log(f"All tested frontier points locked, sleeping...")
            time.sleep(5)
            continue

        random_read_map_store.unlock_point(selected_point.id)
        _log(f"Removed lock!")

    # Write timing report to disk
    # Calculate total from high-level categories only (exploration breakdown is already included in exploration)
    total_time = sum(timing_stats[k] for k in ['get_frontier', 'lock_attempts', 'exploration', 'add_to_frontier', 'mark_searched'])

    # Build report
    report = {
        'metadata': {
            'partitions_dir': partitions_dir,
            'num_partitions': db.num_partitions,
            'search_id': search_id,
            'max_iters': max_iters if max_iters != math.inf else None,
            'actual_iters': timing_stats['total_iters'],
            'frontier_limit': frontier_limit,
            'timestamp': datetime.now().isoformat(),
            'worker_pid': os.getpid(),
        },
        'timings': {
            'total_time': total_time,
            'high_level': {
                'get_frontier': {
                    'seconds': timing_stats['get_frontier'],
                    'percent': (timing_stats['get_frontier'] / total_time * 100) if total_time > 0 else 0
                },
                'lock_attempts': {
                    'seconds': timing_stats['lock_attempts'],
                    'percent': (timing_stats['lock_attempts'] / total_time * 100) if total_time > 0 else 0
                },
                'exploration': {
                    'seconds': timing_stats['exploration'],
                    'percent': (timing_stats['exploration'] / total_time * 100) if total_time > 0 else 0
                },
                'add_to_frontier': {
                    'seconds': timing_stats['add_to_frontier'],
                    'percent': (timing_stats['add_to_frontier'] / total_time * 100) if total_time > 0 else 0
                },
                'mark_searched': {
                    'seconds': timing_stats['mark_searched'],
                    'percent': (timing_stats['mark_searched'] / total_time * 100) if total_time > 0 else 0
                }
            },
            'exploration_breakdown': {
                'find_neighbors': {
                    'seconds': timing_stats['explore_find_neighbors'],
                    'percent': (timing_stats['explore_find_neighbors'] / total_time * 100) if total_time > 0 else 0
                },
                'filter_checks': {
                    'seconds': timing_stats['explore_filter_checks'],
                    'percent': (timing_stats['explore_filter_checks'] / total_time * 100) if total_time > 0 else 0
                },
                'add_points': {
                    'seconds': timing_stats['explore_add_points'],
                    'percent': (timing_stats['explore_add_points'] / total_time * 100) if total_time > 0 else 0
                },
                'get_existing_points': {
                    'seconds': timing_stats['explore_get_existing_points'],
                    'percent': (timing_stats['explore_get_existing_points'] / total_time * 100) if total_time > 0 else 0
                },
                'add_edges': {
                    'seconds': timing_stats['explore_add_edges'],
                    'percent': (timing_stats['explore_add_edges'] / total_time * 100) if total_time > 0 else 0
                },
                'mark_explored': {
                    'seconds': timing_stats['explore_mark_explored'],
                    'percent': (timing_stats['explore_mark_explored'] / total_time * 100) if total_time > 0 else 0
                }
            }
        }
    }

    # Create timing_reports directory if it doesn't exist
    os.makedirs('timing_reports', exist_ok=True)

    # Generate filename with timestamp and PID
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"timing_reports/timing_report_{timestamp_str}_pid{os.getpid()}.json"

    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nTiming report written to: {filename}")


def continue_search_waterfill(search_id,
                               partitions_dir: str,
                               energy_calc: BaseCalculator,
                               search_filters: list[SearchFilter] = None,
                               max_iters: int = None,
                               frontier_limit: int = 100,
                               log_lvl: int = 0):
    """Continue a water-filling search process with partitioned database.

    Water-filling algorithm:
    1. Get frontier points (sorted by energy, lowest first)
    2. For each frontier point:
       - Get all its neighbors (across all partitions)
       - Find unsearched, unlocked neighbors
       - Select one, lock it, calculate energy, add to frontier
       - If no unsearched neighbors left, mark frontier point as searched

    Args:
        search_id: ID of the search process
        partitions_dir: Path to the partitions directory
        energy_calc: Calculator to compute point energies
        search_filters: Optional filters to apply to candidate points
        max_iters: Maximum iterations before stopping (default: infinite)
        frontier_limit: Max frontier points to consider per iteration (default: 100)
        log_lvl: Logging verbosity level
    """
    db = PartitionedDB(partitions_dir)
    log_lvl = 0
    def _log(msg, lvl=1):
        if lvl > log_lvl:
            print(msg)

    if max_iters is None:
        max_iters = math.inf

    num_iters = 0
    while True:
        _log(f"================ BEGINNING STEP {num_iters} ================")
        if num_iters > max_iters:
            _log(f"Reached {num_iters} iterations, quitting...")
            break
        _log(f"Continuing water-filling search...")

        # Get frontier points from a random partition
        # (In water-filling, frontier points have the lowest energy values)
        random_read_store_idx = db.get_random_partition_idx()
        random_read_search_store = db.get_search_store_by_idx(random_read_store_idx)
        random_read_map_store = db.get_map_store_by_idx(random_read_store_idx)

        frontier_points = random_read_search_store.get_frontier_points_in_search(search_id, limit=frontier_limit)
        _log(f"Found {len(frontier_points)} frontier points to consider (limited to {frontier_limit})...")

        selected_point = None
        selected_partition = None

        for frontier_point in frontier_points:
            _log(f"Considering frontier pt ID: {frontier_point.id}, CNF: {frontier_point.cnf.coords}")

            # If frontier point not explored yet, explore it first
            if not frontier_point.explored:
                _log(f"Frontier point not explored yet, exploring...")
                _, new_nb_ids, _ = explore_pt_partition(db, frontier_point.cnf, filters=search_filters, log_lvl=log_lvl)
                _log(f"Added {len(new_nb_ids)} neighbors to the map!")

            # Get all neighbors across all partitions with their metadata
            # NOTE: Don't pass frontier_point.id! It's only valid in the partition we read from,
            # not the partition the point belongs to (determined by CNF hash).
            # Pass None to look up the correct point ID in its actual partition.
            unsearched_neighbors, partition_locks = db.get_unsearched_neighbors_and_locks(frontier_point.cnf, search_id)
            _log(f"Found {len(unsearched_neighbors)} unsearched neighbors")

            if len(unsearched_neighbors) == 0:
                _log(f"No unsearched neighbors - frontier point exhausted, marking as searched")
                random_read_search_store.mark_point_searched_by_id(search_id, frontier_point.id)
                random_read_search_store.remove_from_search_frontier_by_id(search_id, frontier_point.id)
                continue

            for candidate in unsearched_neighbors:
                candidate_partition = candidate.partition
                candidate_map_store = db.get_map_store_by_idx(candidate_partition)
                lock_acquired = candidate_map_store.lock_point(candidate.id)
                if not lock_acquired:
                    _log(f"Failed to acquire lock (another worker got it first), trying next candidate point...")
                    continue
                else:
                    _log(f"Lock acquired successfully!")
                    selected_point = candidate
                    selected_partition = candidate_partition
                    break

            selected_point = candidate
            selected_partition = candidate_partition
            break

        if selected_point is None:
            _log(f"Found no unlocked points to compute, sleeping for 5 seconds...")
            # time.sleep(5)
            continue

        # Calculate energy for selected point
        _log(f"Calculating energy for point {selected_point.cnf.coords} (ID {selected_point.id} in partition {selected_partition})")
        energy = energy_calc.calculate_energy(selected_point.cnf)
        _log(f"Energy: {energy}")

        # Set the energy value in the correct partition
        selected_map_store = db.get_map_store_by_idx(selected_partition)
        selected_search_store = db.get_search_store_by_idx(selected_partition)

        selected_map_store.set_point_value(selected_point.id, energy)
        selected_search_store.add_to_search_frontier_by_id(search_id, selected_point.id)
        selected_map_store.unlock_point(selected_point.id)

        _log(f"Added point to search frontier and removed lock!")

        num_iters += 1
