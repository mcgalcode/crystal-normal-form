from ..calculation import BaseCalculator
from ..crystal_normal_form import CrystalNormalForm
from ..db import PartitionedDB, CrystalMapStore, SearchProcessStore
from ..utils.log import Logger
from ..navigation import find_neighbors
import math
import time
from collections import defaultdict

def explore_pt_partition(partition_db: PartitionedDB, point_cnf: CrystalNormalForm, log_lvl=1,
                         profile_timings: dict = None, profile_counts: dict = None):
    point_partition = partition_db.get_partition_idx(point_cnf)
    local_map_store = partition_db.get_map_store(point_cnf)
    pt = local_map_store.get_point_by_cnf(point_cnf)

    if pt.explored:
        return [nb.id for nb in local_map_store.get_local_neighbors(pt.id)]

    t0 = time.time()
    nbs = find_neighbors(pt.cnf)
    if profile_timings is not None:
        profile_timings['find_neighbors'] += time.time() - t0
        profile_counts['find_neighbors'] += 1

    local_nb_ids = []
    local_edges = []

    t0 = time.time()
    partitioned_cnfs = partition_db.partition_cnfs(nbs)
    if profile_timings is not None:
        profile_timings['partition_cnfs'] += time.time() - t0
        profile_counts['partition_cnfs'] += 1

    for partition_idx, cnfs in partitioned_cnfs.items():
        # If these are local, insert them and add edges
        if partition_idx == point_partition:
            t0 = time.time()
            local_map_store.bulk_insert_points(cnfs)
            if profile_timings is not None:
                profile_timings['write_local_partition'] += time.time() - t0
                profile_counts['write_local_partition'] += 1
                profile_counts['local_points_written'] += len(cnfs)

            local_nb_ids = local_map_store.get_point_ids(cnfs)
            for nb_id in local_nb_ids:
                local_edges.append((pt.id, nb_id, None))

        # If these are in another partition, send to mailbox
        else:
            t0 = time.time()
            partition_db.bulk_add_incoming_points(cnfs, partition_idx)
            if profile_timings is not None:
                profile_timings['write_remote_inbox'] += time.time() - t0
                profile_counts['write_remote_inbox'] += 1
                profile_counts['remote_points_written'] += len(cnfs)

            for cnf in cnfs:
                local_edges.append((pt.id, None, cnf))

    t0 = time.time()
    local_map_store.bulk_add_edges(local_edges)
    if profile_timings is not None:
        profile_timings['add_edges'] += time.time() - t0
        profile_counts['add_edges'] += 1

    local_map_store.mark_point_explored(pt.id)

    return local_nb_ids

def process_cnf_batch(cnfs: list[CrystalNormalForm],
                      map_store: CrystalMapStore,
                      search_store: SearchProcessStore,
                      search_id: int,
                      calculator: BaseCalculator,
                      profile_timings: dict = None,
                      profile_counts: dict = None):
    map_store.bulk_insert_points(cnfs)
    all_ids = map_store.get_point_ids(cnfs)
    return process_cnf_ids_batch(all_ids, map_store, search_store, search_id, calculator,
                                 profile_timings, profile_counts) 

def process_cnf_ids_batch(cnf_ids: list[int],
                          map_store: CrystalMapStore,
                          search_store: SearchProcessStore,
                          search_id: int,
                          calculator: BaseCalculator,
                          profile_timings: dict = None,
                          profile_counts: dict = None):
    searched_ids = search_store.get_searched_ids_intersecting_with(search_id, cnf_ids)
    new_frontier_ids = list(set(cnf_ids) - set(searched_ids))

    t0 = time.time()
    search_store.bulk_add_to_search_frontier_by_id(search_id, new_frontier_ids)
    if profile_timings is not None:
        profile_timings['add_to_frontier'] += time.time() - t0
        profile_counts['add_to_frontier'] += 1

    t0 = time.time()
    new_frontier_cnfs = map_store.get_points_by_ids(new_frontier_ids)
    if profile_timings is not None:
        profile_timings['get_points_by_ids'] += time.time() - t0
        profile_counts['get_points_by_ids'] += 1

    new_frontier_cnfs = [pt.cnf for pt in new_frontier_cnfs]
    for id, cnf in zip(new_frontier_ids, new_frontier_cnfs):
        t0 = time.time()
        energy = calculator.calculate_energy(cnf)
        if profile_timings is not None:
            profile_timings['calculate_energy'] += time.time() - t0
            profile_counts['calculate_energy'] += 1
        map_store.set_point_value(id, energy)
    return new_frontier_ids


def waterfill_step(db: PartitionedDB,
                   partition_idx: int,
                   search_id: int,
                   logger: Logger,
                   calculator: BaseCalculator,
                   batch_size: int,
                   energy_limit: float,
                   log_lvl: int,
                   profile_timings: dict = None,
                   profile_counts: dict = None):
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

    t0 = time.time()
    incoming_cnfs, incoming_ids = search_store.get_incoming_points(search_id, limit=1000)
    if profile_timings is not None:
        profile_timings['get_incoming'] += time.time() - t0
        profile_counts['get_incoming'] += 1

    logger.info(f"[partition={partition_idx}] Received {len(incoming_cnfs)} from other workers")
    if len(incoming_cnfs) > 0:
        mailbox_frontier_ids = process_cnf_batch(incoming_cnfs,
                                                map_store,
                                                search_store,
                                                search_id,
                                                calculator,
                                                profile_timings,
                                                profile_counts)

        all_new_frontier_ids = all_new_frontier_ids + mailbox_frontier_ids

        # Delete processed points from inbox after successful processing
        search_store.delete_incoming_points_by_ids(incoming_ids)
        logger.debug(f"Deleted {len(incoming_ids)} processed points from inbox")

    # PROCESS FRONTIER NODES
    t0 = time.time()
    frontier_points = search_store.get_frontier_points_in_search(
        search_id, limit=batch_size, max_energy=energy_limit
    )
    if profile_timings is not None:
        profile_timings['get_frontier'] += time.time() - t0
        profile_counts['get_frontier'] += 1

    logger.debug(f"Found {len(frontier_points)} frontier points at/near water level (limit={batch_size}, max_energy={energy_limit})...")

    for frontier_point in frontier_points:
        logger.debug(f"Searching frontier pt ID: {frontier_point.id}, VALUE: {frontier_point.value}, CNF: {frontier_point.cnf.coords}")

        # Explore (profiling happens inside explore_pt_partition)
        local_nb_ids = explore_pt_partition(db, frontier_point.cnf, log_lvl=log_lvl,
                                           profile_timings=profile_timings,
                                           profile_counts=profile_counts)

        # Calculate NBs + add them to frontier (profiling happens inside process_cnf_ids_batch)
        new_frontier_ids = process_cnf_ids_batch(local_nb_ids, map_store, search_store, search_id, calculator,
                                                 profile_timings, profile_counts)

        logger.debug(f"Added {len(new_frontier_ids)} to the search frontier")
        all_new_frontier_ids = all_new_frontier_ids + new_frontier_ids

        # Mark this frontier point as searched (we're done with it)
        search_store.mark_point_searched_by_id(search_id, frontier_point.id)

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

    # Profiling counters
    profile_timings = defaultdict(float)
    profile_counts = defaultdict(int)

    num_iters = 0
    while True:
        step_start = time.time()

        # Reload frontier_width from metadata file to allow dynamic updates
        frontier_width = db.reload_frontier_width()
        water_level = db.get_current_water_level() + frontier_width
        partition_idx = db.get_random_partition_idx()

        logger.debug(f"========== [partition {partition_idx}] BEGINNING STEP {num_iters} - water level: {water_level} ================")
        if num_iters > max_iters:
            logger.info(f"Reached {num_iters} iterations, quitting...")
            break

        if db.is_search_complete():
            logger.info(f"Search is complete! Worker retiring...")
            break

        # Choose a partition to operate on for this iteration

        new_ids = waterfill_step(db,
                       partition_idx,
                       search_id,
                       logger,
                       calculator,
                       batch_size,
                       water_level,
                       log_lvl,
                       profile_timings,
                       profile_counts)

        logger.info(f"Added {len(new_ids)} to the frontier during this step!")

        # Sync partition stats to metastore for monitoring
        db.gather_and_sync_partition_stats(partition_idx)

        # Update global water level based on current partition frontiers
        logger.info(f"Syncing water level to control plane!")
        db.sync_control_water_level()

        # Review partitions to see if endpoint is reached
        db.sync_search_completion_status()

        profile_timings['total_step'] += time.time() - step_start
        num_iters += 1

        # Log profiling information every 2 steps
        if num_iters % 2 == 0:
            logger.info("=" * 80)
            logger.info(f"PROFILING REPORT (after {num_iters} steps)")
            logger.info("=" * 80)

            total_time = profile_timings['total_step']

            # Define the key operations to report in order
            key_operations = [
                ('find_neighbors', 'Finding neighbors'),
                ('calculate_energy', 'Calculating energy'),
                ('get_frontier', 'Reading frontier from DB'),
                ('get_points_by_ids', 'Getting points by IDs'),
                ('add_to_frontier', 'Adding to frontier'),
                ('write_local_partition', 'Writing to local partition'),
                ('write_remote_inbox', 'Writing to remote mailboxes'),
                ('get_incoming', 'Reading from mailbox'),
            ]

            accounted_time = 0
            for key, label in key_operations:
                if key in profile_timings:
                    elapsed = profile_timings[key]
                    count = profile_counts[key]
                    pct = (elapsed / total_time * 100) if total_time > 0 else 0
                    accounted_time += elapsed
                    logger.info(f"  {label:30s}: {elapsed:8.3f}s ({pct:5.1f}%)")

            other_time = total_time - accounted_time
            other_pct = (other_time / total_time * 100) if total_time > 0 else 0
            logger.info(f"  {'Other operations':30s}: {other_time:8.3f}s ({other_pct:5.1f}%)")
            logger.info("-" * 80)
            logger.info(f"  {'TOTAL':30s}: {total_time:8.3f}s (100.0%)")
            logger.info("-" * 80)

            # Report on partition write statistics
            local_points = profile_counts.get('local_points_written', 0)
            remote_points = profile_counts.get('remote_points_written', 0)
            total_points = local_points + remote_points

            if total_points > 0:
                logger.info(f"  Local points written:  {local_points:8d} ({local_points/total_points*100:5.1f}% of points)")
                logger.info(f"  Remote points written: {remote_points:8d} ({remote_points/total_points*100:5.1f}% of points)")

            logger.info("=" * 80)
