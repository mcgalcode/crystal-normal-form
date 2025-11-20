from ..crystal_normal_form import CrystalNormalForm
from ..db.crystal_map_store import CrystalMapStore
from ..db.search_store import SearchProcessStore
from ..db.setup import setup_cnf_db
from ..navigation.neighbor_finder import NeighborFinder
from ..navigation.search_filters import VolumeLimitFilter, SearchFilter
from ..calculation.base_calculator import BaseCalculator
import time
import math
import sqlite3

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
    return search_id

def explore_pt(store: CrystalMapStore, pt_id: int, filters: list[SearchFilter] = None):
    if filters is None:
        filters = []

    pt = store.get_point_by_id(pt_id)
    nbs = NeighborFinder(pt.cnf).find_neighbors()
    all_nb_ids = []
    new_nb_ids = []
    filtered_ct = 0
    for nb in nbs:
        struct = nb.reconstruct()
        if not all([f.should_add_pt(nb, struct) for f in filters]):
            filtered_ct += 1
            continue
        try:
            nb_id = store.add_point(nb)
            new_nb_ids.append(nb_id)
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" not in e.__repr__():
                raise e
            else:
                nb_id = store.get_point_by_cnf(nb).id
        store.add_connection_by_ids(pt_id, nb_id)
        all_nb_ids.append(nb_id)
    print(f"Skipped {filtered_ct} due to search filter constraints...")
    store.mark_point_explored(pt_id)
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
    