#!/usr/bin/env python3
"""
Run water-filling pathfinding search on a partitioned CNF database.

Usage:
    cnf-pathfind-waterfill <partitions_dir>
    cnf-pathfind-waterfill <partitions_dir> --max-iters 1000 --log-lvl 1
"""

import argparse
import multiprocessing as mp
import math
from cnf.navigation.waterfill import continue_search_waterfill
from cnf.db.partitioned_db import PartitionedDB
from cnf.calculation.grace import GraceCalculator

def _continue_search_waterfill(search_proc_id, partitions_dir, log_lvl, max_iters, partition_range):
    """Worker function to run waterfill search on a subset of partitions."""
    continue_search_waterfill(
        search_proc_id,
        partitions_dir,
        GraceCalculator(),
        log_lvl=log_lvl,
        max_iters=max_iters,
        partition_range=partition_range
    )    

def main():
    parser = argparse.ArgumentParser(
        description='Run water-filling pathfinding search on partitioned CNF database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  %(prog)s ./search_db

  # Run with max 1000 iterations
  %(prog)s ./search_db --max-iters 1000

  # Run with minimal logging
  %(prog)s ./search_db --log-lvl 0

  # Run with custom settings
  %(prog)s ./search_db --max-iters 5000 --log-lvl 2
        """
    )

    parser.add_argument('partitions_dir',
                       help='Directory containing graph_partition_*.db files')
    parser.add_argument('--num-workers', type=int, default=1,
                        help="The number of worker processes to work on this search problem.")
    parser.add_argument('--max-iters', type=int, default=None,
                       help='Maximum number of iterations (default: no limit)')
    parser.add_argument('--log-lvl', type=int, default=2,
                       help='Logging level: 0=FATAL, 1=SEVERE, 2=WARN, 3=INFO, 4=DEBUG (default: 2)')

    args = parser.parse_args()

    print(f"Using partition directory {args.partitions_dir}...")
    print(f"Max iterations: {args.max_iters if args.max_iters else 'unlimited'}")
    print(f"Num. Workers: {args.num_workers}")
    print(f"Log level: {args.log_lvl}")
    print()

    search_proc_id = 1

    pdb = PartitionedDB(args.partitions_dir, search_proc_id)
    num_workers = args.num_workers
    num_partitions_per_worker = math.ceil(pdb.num_partitions / num_workers)
    all_partition_nums = list(range(pdb.num_partitions))
    partition_ranges = [all_partition_nums[start_idx:start_idx + num_partitions_per_worker] for start_idx in range(0, pdb.num_partitions, num_partitions_per_worker)]

    print(f"Starting {len(partition_ranges)} workers...")
    processes = []
    for i, partition_range in enumerate(partition_ranges):
        print(f"  Worker {i}: partitions {partition_range}")
        proc = mp.Process(
            target=_continue_search_waterfill,
            args=(search_proc_id, args.partitions_dir, args.log_lvl, args.max_iters, partition_range)
        )
        proc.start()
        processes.append(proc)

    print(f"\nAll workers launched. Waiting for completion...")
    print("(Press Ctrl+C to interrupt)")
    print()

    try:
        # Wait for all workers to complete
        for proc in processes:
            proc.join()
        print("\n✓ All workers completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user, terminating workers...")
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.join()
        exit(1)



if __name__ == "__main__":
    main()