from ..calculation import BaseCalculator
from ..crystal_normal_form import CrystalNormalForm
from ..db import PartitionedDB, CrystalMapStore, SearchProcessStore
from ..utils.log import Logger
from ..navigation import find_neighbors
import math

FRONTIER_WIDTH = 0.002

def explore_pt_partition(partition_db: PartitionedDB, point_cnf: CrystalNormalForm, log_lvl=1):
    point_partition = partition_db.get_partition_idx(point_cnf)
    local_map_store = partition_db.get_map_store(point_cnf)
    pt = local_map_store.get_point_by_cnf(point_cnf)

    if pt.explored:
        return local_map_store.get_local_neighbors(pt.id)

    nbs = find_neighbors(pt.cnf)

    local_nb_ids = []
    local_edges = []   

    for partition_idx, cnfs in partition_db.partition_cnfs(nbs).items():
        
        # If these are local, insert them and add edges
        if partition_idx == point_partition:
            local_map_store.bulk_insert_points(cnfs)
            local_nb_ids = local_map_store.get_point_ids(cnfs)
            for nb_id in local_nb_ids:
                local_edges.append((pt.id, nb_id, None))

        # If these are in another partition, send to mailbox
        else:
            foreign_store = partition_db.get_search_store_by_idx(partition_idx)
            foreign_store.bulk_add_incoming_points(partition_db.search_id, cnfs)
            for cnf in cnfs:
                local_edges.append((pt.id, None, cnf))

    local_map_store.bulk_add_edges(local_edges)
    local_map_store.mark_point_explored(pt.id)
    
    return local_nb_ids

def process_cnf_batch(cnfs: list[CrystalNormalForm],
                      map_store: CrystalMapStore,
                      search_store: SearchProcessStore,
                      search_id: int,
                      calculator: BaseCalculator):
    map_store.bulk_insert_points(cnfs)
    all_ids = map_store.get_point_ids(cnfs)   
    return process_cnf_ids_batch(all_ids, map_store, search_store, search_id, calculator) 

def process_cnf_ids_batch(cnf_ids: list[int],
                          map_store: CrystalMapStore,
                          search_store: SearchProcessStore,
                          search_id: int,
                          calculator: BaseCalculator):
    searched_ids = search_store.get_searched_ids_intersecting_with(search_id, cnf_ids)
    new_frontier_ids = list(set(cnf_ids) - set(searched_ids))
    search_store.bulk_add_to_search_frontier_by_id(search_id, new_frontier_ids)
    new_frontier_cnfs = map_store.get_points_by_ids(new_frontier_ids)
    new_frontier_cnfs = [pt.cnf for pt in new_frontier_cnfs]
    for id, cnf in zip(new_frontier_ids, new_frontier_cnfs):
        energy = calculator.calculate_energy(cnf)
        map_store.set_point_value(id, energy)
    return new_frontier_ids


def waterfill_step(db: PartitionedDB,
                   partition_idx: int,
                   search_id: int,
                   logger: Logger,
                   calculator: BaseCalculator,
                   batch_size: int,
                   energy_limit: float,
                   log_lvl: int):
    """
    Takes a step during the waterfilling algorithm, which consists of:
        1. Process incoming nodes
            a. read all from incoming nodes tables
            b. add each node to the partition if it's not already there
            c. add each node to the search frontier if it's not already searched
            d. compute the energy of each of these nodes if it's not already computed
        2. Process frontier nodes
            a. Select X nodes from the frontier whose energy is below current water level
            b. If node is unexplored (meaning, neighbors have not been computed), explore it
            c. Send foreign neighbors to their respective partition inboxes
            d. For local nodes:
                i. add each node to the partition if it's not already there
                ii. add each node to the search frontier if it's not already searched
                iii. compute the energy of each of these nodes if it's not already computed
            e. Mark node as searched
    """
    search_store = db.get_search_store_by_idx(partition_idx)
    map_store = db.get_map_store_by_idx(partition_idx)

    all_new_frontier_ids = []

    # PROCESS INCOMING NODES

    incoming_cnfs = search_store.get_and_empty_incoming_points(search_id)
    mailbox_frontier_ids = process_cnf_batch(incoming_cnfs,
                                             map_store,
                                             search_store,
                                             search_id,
                                             calculator)

    all_new_frontier_ids = all_new_frontier_ids + mailbox_frontier_ids

    # PROCESS FRONTIER NODES
    frontier_points = search_store.get_frontier_points_in_search(
        search_id, limit=batch_size, max_energy=energy_limit
    )

    logger.debug(f"Found {len(frontier_points)} frontier points at/near water level (limit={batch_size}, max_energy={energy_limit})...")

    for frontier_point in frontier_points:
        logger.debug(f"Considering frontier pt ID: {frontier_point.id}, VALUE: {frontier_point.value}, CNF: {frontier_point.cnf.coords}")

        # Explore
        local_nb_ids = explore_pt_partition(db, frontier_point.cnf, log_lvl=log_lvl)
        logger.debug(f"Found {len(local_nb_ids)} local neighbors")
        # Calculate NBs + add them to frontier
        new_frontier_ids = process_cnf_ids_batch(local_nb_ids, map_store, search_store,search_id, calculator)
        logger.debug(f"Added {len(new_frontier_ids)} to the search frontier")
        all_new_frontier_ids = all_new_frontier_ids + new_frontier_ids

        # Mark this frontier point as searched (we're done with it)
        search_store.mark_point_searched_by_id(search_id, frontier_point.id)
        logger.debug(f"Marked frontier point as searched and removed from frontier")

    return all_new_frontier_ids


def continue_search_waterfill(search_id,
                              partitions_dir: str,
                              calculator: BaseCalculator,
                              max_iters: int = None,
                              batch_size: int = 100,
                              partition_range: list[int] = None,
                              log_lvl: int = 0):
    """Continue a water-filling search process with partitioned database.

    1. Track water_level = minimum frontier energy across all partitions
    2. Only explore frontier points below the water level
    3. When exploring a point:
       - Calculate energy for ALL new neighbors
       - Add all neighbors to frontier
       - Mark point as searched
    4. Water rises naturally as low-energy regions are exhausted

    Args:
        search_id: ID of the search process
        partitions_dir: Path to the partitions directory
        energy_calc: Calculator to compute point energies
        search_filters: Optional filters to apply to candidate points
        max_iters: Maximum iterations before stopping (default: infinite)
        frontier_limit: Max frontier points to consider per iteration (default: 100)
        log_lvl: Logging verbosity level
    """
    db = PartitionedDB(partitions_dir, search_id, partition_range)
    logger = Logger(log_lvl)
    if max_iters is None:
        max_iters = math.inf

    num_iters = 0
    while True:
        logger.debug(f"================ BEGINNING STEP {num_iters} ================")
        if num_iters > max_iters:
            logger.info(f"Reached {num_iters} iterations, quitting...")
            break

        if db.is_search_complete():
            logger.info(f"Search is complete! Worker retiring...")
            break

        water_level = db.get_current_water_level() + FRONTIER_WIDTH
        logger.debug(f"Current water level: {water_level}")

        # Choose a partition to operate on for this iteration
        partition_idx = db.get_random_partition_idx()

        waterfill_step(db,
                       partition_idx,
                       search_id,
                       logger,
                       calculator,
                       batch_size,
                       water_level,
                       log_lvl)

        # Update global water level based on current partition frontiers
        db.sync_control_water_level()

        num_iters += 1
