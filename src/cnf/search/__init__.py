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
        search_store.add_search_start_point(search_id, sid)
        search_store.add_to_search_frontier_by_id(search_id, sid)
    
    for eid in end_point_ids:
        search_store.add_search_end_point(search_id, eid)

def explore_pt(partition_db: PartitionedDB, point_cnf: CrystalNormalForm, filters: list[SearchFilter] = None, log_lvl=1):
    def _log(msg, lvl = 1):
        if lvl > log_lvl:
            print(msg)
    if filters is None:
        filters = []

    point_partition = partition_db.get_partition_idx(point_cnf)

    point_map_store = partition_db.get_map_store(point_cnf)

    pt = point_map_store.get_point_by_cnf(point_cnf)
    nbs = NeighborFinder(pt.cnf).find_neighbors()
    all_nb_ids = []
    new_nb_ids = []
    existing_nb_ids = []
    filtered_ct = 0
    edges_added = 0

    for nb in nbs:
        nb_partition = partition_db.get_partition_idx(nb)
        nb_map_store = partition_db.get_map_store_by_idx(nb_partition)

        if filters:  # Only reconstruct if needed
            struct = nb.reconstruct()
            if not all([f.should_add_pt(nb, struct) for f in filters]):
                filtered_ct += 1
                continue
        
        nb_id = nb_map_store.add_point(nb)
        if nb_id is not None:
            new_nb_ids.append((nb_partition, nb_id))
        else:
            nb_id = nb_map_store.get_point_by_cnf(nb).id
            existing_nb_ids.append((nb_partition, nb_id))

        
        # If the neighbor is in the same partition as the point, add local edge
        if point_partition == nb_partition:
            edge_added = nb_map_store.add_connection_by_ids(pt.id, nb_id)
        else:
            point_map_store.add_connection_to_target_cnf(pt.id, nb)
            edge_added = nb_map_store.add_connection_to_target_cnf(nb_id, point_cnf)

        if edge_added:
            edges_added += 1

        all_nb_ids.append((nb_partition, nb_id))

    _log(f"Exploration: {len(new_nb_ids)} new neighbors, {len(existing_nb_ids)} existing, {edges_added} edges added, {filtered_ct} filtered")
    point_map_store.mark_point_explored(pt.id)
    return all_nb_ids, new_nb_ids

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
                               log_lvl=0):
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
    def _log(msg, lvl = 1):
        if lvl > log_lvl:
            print(msg)

    while True:
        _log(f"================ BEGINNING STEP {num_iters} ================")
        if num_iters > max_iters:
            _log(f"Reached {num_iters} iterations, quitting...")
            break
        _log(f"Endpoint is not yet in the frontier, continuing search...")

        random_read_store_idx = db.get_random_partition_idx()
        random_read_search_store = db.get_search_store_by_idx(random_read_store_idx)
        random_read_map_store = db.get_map_store_by_idx(random_read_store_idx)
        frontier_points = random_read_search_store.get_frontier_points_in_search(search_id, limit=frontier_limit)
        _log(f"Found {len(frontier_points)} frontier points to consider (limited to {frontier_limit})...")

        selected_point = None
        for frontier_point in frontier_points:
            _log(f"Attempting lock on frontier pt ID: {frontier_point.id}, CNF: {frontier_point.cnf.coords}")
            lock_acquired = random_read_map_store.lock_point(frontier_point.id)
            if not lock_acquired:
                _log(f"Failed to acquire lock (another worker got it first), trying next point...")
                continue  # Continue to next frontier point
            _log(f"Lock acquired successfully!")
            selected_point = frontier_point
            
            if not selected_point.explored:
                _log(f"Point has not been explored... computing neighbors and adding to the map...")
                _, new_nbs = explore_pt(db, selected_point.cnf, filters=search_filters, log_lvl=log_lvl)
                for nb in new_nbs:
                    partition, nb_id = nb
                    nb_store = db.get_search_store_by_idx(partition)
                    nb_store.add_to_search_frontier_by_id(search_id, nb_id)
                _log(f"Added {len(new_nbs)} neighbors to the map!")
            else:
                _log(f"Point ID {selected_point.id} has already been explored, marking as searched.")

            random_read_search_store.mark_point_searched_by_id(search_id, selected_point.id)
            random_read_search_store.remove_from_search_frontier_by_id(search_id, selected_point.id)
            break

        num_iters += 1
        
        if selected_point is None:
            _log(f"All tested frontier points locked, sleeping...")
            time.sleep(5)
            continue
        
        random_read_map_store.unlock_point(selected_point.id)
        _log(f"Removed lock!")
    