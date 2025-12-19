from ..crystal_normal_form import CrystalNormalForm
from ..calculation.grace import GraceCalculator
from ..db.crystal_map_store import CrystalMapStore
from ..db.partitioned_db import PartitionedDB
from ..db.search_store import SearchProcessStore
from ..navigation.neighbor_finder import NeighborFinder
from ..navigation.search_filters import VolumeLimitFilter, SearchFilter
from ..calculation.base_calculator import BaseCalculator
from ..utils.log import Logger
import time
import math

FRONTIER_WIDTH = 0.002 # eV

def explore_pt(map_store: CrystalMapStore, pt_id: int, filters: list[SearchFilter] = None, log_lvl=1):
    if filters is None:
        filters = []

    pt = map_store.get_point_by_id(pt_id)

    nbs = NeighborFinder.from_cnf(pt.cnf).find_neighbors(pt.cnf)
    
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
    logger = Logger(log_lvl)
    if filters is None:
        filters = []


    point_partition = partition_db.get_partition_idx(point_cnf)
    point_map_store = partition_db.get_map_store(point_cnf)
    pt = point_map_store.get_point_by_cnf(point_cnf)

    nbs = NeighborFinder.from_cnf(pt.cnf).find_neighbors(pt.cnf)

    all_nb_ids = []
    new_nb_ids = []
    existing_nb_ids = []
    filtered_ct = 0
    edges_added = 0
    
    edges_to_add_to_point_store = []   

    for nb in nbs:
        nb_partition = partition_db.get_partition_idx(nb)
        nb_map_store = partition_db.get_map_store_by_idx(nb_partition)

        if filters:
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


        if point_partition == nb_partition:
            edges_to_add_to_point_store.append((pt.id, nb_id, None))
        else:
            edges_to_add_to_point_store.append((pt.id, None, nb))


        all_nb_ids.append((nb_partition, nb_id))
    
    point_map_store.bulk_add_edges(edges_to_add_to_point_store)

    logger.info(f"Exploration: {len(new_nb_ids)} new neighbors, {len(existing_nb_ids)} existing, {edges_added} edges added, {filtered_ct} filtered")

    point_map_store.mark_point_explored(pt.id)

    return all_nb_ids, new_nb_ids

def continue_search_waterfill(search_id,
                               partitions_dir: str,
                               energy_calc: BaseCalculator,
                               search_filters: list[SearchFilter] = None,
                               max_iters: int = None,
                               frontier_limit: int = 100,
                               log_lvl: int = 0):
    """Continue a water-filling search process with partitioned database.

    True water-filling algorithm with global water level tracking:
    1. Track water_level = minimum frontier energy across all partitions
    2. Only explore frontier points at or near the water level
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
    db = PartitionedDB(partitions_dir)
    logger = Logger(log_lvl)
    if max_iters is None:
        max_iters = math.inf

    # Water level tracking
    water_level = None
    tolerance = FRONTIER_WIDTH

    num_iters = 0
    while True:
        logger.debug(f"================ BEGINNING STEP {num_iters} ================")
        if num_iters > max_iters:
            logger.info(f"Reached {num_iters} iterations, quitting...")
            break

        # Initialize or update water level if needed
        if water_level is None:
            logger.debug("Initializing water level...")
            water_level = db.get_current_water_level(search_id)
            logger.info(f"Initial water level: {water_level}")

        logger.debug(f"Current water level: {water_level}")

        # Get frontier points from a random partition, filtered by water level
        random_read_store_idx = db.get_random_partition_idx()
        random_read_search_store = db.get_search_store_by_idx(random_read_store_idx)
        random_read_map_store = db.get_map_store_by_idx(random_read_store_idx)

        max_energy = water_level + tolerance

        frontier_points = random_read_search_store.get_frontier_points_in_search(
            search_id, limit=frontier_limit, max_energy=max_energy
        )

        logger.debug(f"Found {len(frontier_points)} frontier points at/near water level (limit={frontier_limit}, max_energy={max_energy})...")

        # If no frontier points at current level, update water level
        if len(frontier_points) == 0:
            logger.debug("No frontier points at current water level, updating...")
            water_level = db.get_current_water_level(search_id)
            logger.info(f"Updated water level: {water_level}")
            continue

        for frontier_point in frontier_points:
            logger.debug(f"Considering frontier pt ID: {frontier_point.id}, VALUE: {frontier_point.value}, CNF: {frontier_point.cnf.coords}")

            # If frontier point has no energy, calculate it (happens for start points)
            if frontier_point.value is None:
                logger.debug(f"Frontier point has no energy, calculating...")
                energy = energy_calc.calculate_energy(frontier_point.cnf)
                logger.info(f"Energy: {energy} CNF={frontier_point.cnf.coords} (frontier point)")
                # Update in the partition where we read this point from
                random_read_map_store.set_point_value(frontier_point.id, energy)
                frontier_point.value = energy  # Update local copy

            # If frontier point not explored yet, explore it first
            if not frontier_point.explored:
                logger.debug(f"Frontier point not explored yet, exploring...")
                _, new_nb_ids, _ = explore_pt_partition(db, frontier_point.cnf, filters=search_filters, log_lvl=log_lvl)
                logger.debug(f"Explored point, found {len(new_nb_ids)} new neighbors")

                # Calculate energy for ALL new neighbors and add them to frontier
                for nb_partition, nb_id in new_nb_ids:
                    nb_map_store = db.get_map_store_by_idx(nb_partition)
                    nb_search_store = db.get_search_store_by_idx(nb_partition)
                    nb_point = nb_map_store.get_point_by_id(nb_id)

                    # Calculate energy for this neighbor
                    nb_energy = energy_calc.calculate_energy(nb_point.cnf)
                    logger.info(f"Energy: {nb_energy} CNF={nb_point.cnf.coords} (neighbor, ID {nb_id} in partition {nb_partition})")
                    nb_map_store.set_point_value(nb_id, nb_energy)

                    # Add to frontier (regardless of energy - water level will filter later)
                    nb_search_store.add_to_search_frontier_by_id(search_id, nb_id)

                logger.debug(f"Calculated energy for {len(new_nb_ids)} neighbors and added to frontier")

            # Mark this frontier point as searched (we're done with it)
            random_read_search_store.mark_point_searched_by_id(search_id, frontier_point.id)
            random_read_search_store.remove_from_search_frontier_by_id(search_id, frontier_point.id)
            logger.debug(f"Marked frontier point as searched and removed from frontier")

            # Process one frontier point per iteration
            break

        num_iters += 1
