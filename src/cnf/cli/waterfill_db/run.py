"""Run command for waterfill database search."""

import sys
import math
import multiprocessing as mp


def run_command(args):
    """Run waterfill search on a partitioned database."""
    from cnf.navigation.waterfill import continue_search_waterfill
    from cnf.db.partitioned_db import PartitionedDB
    from cnf.calculation.grace import GraceCalculator

    print(f"Using partition directory {args.partition_dir}...")
    print(f"Max iterations: {args.max_iters if args.max_iters else 'unlimited'}")
    print(f"Num. Workers: {args.num_workers}")
    print(f"Log level: {args.log_lvl}")
    print()

    search_proc_id = 1

    pdb = PartitionedDB(args.partition_dir, search_proc_id)
    num_workers = args.num_workers
    num_partitions_per_worker = math.ceil(pdb.num_partitions / num_workers)
    all_partition_nums = list(range(pdb.num_partitions))
    partition_ranges = [
        all_partition_nums[start_idx:start_idx + num_partitions_per_worker]
        for start_idx in range(0, pdb.num_partitions, num_partitions_per_worker)
    ]

    def _worker(search_proc_id, partitions_dir, log_lvl, max_iters, partition_range):
        continue_search_waterfill(
            search_proc_id,
            partitions_dir,
            GraceCalculator(),
            log_lvl=log_lvl,
            max_iters=max_iters,
            partition_range=partition_range
        )

    print(f"Starting {len(partition_ranges)} workers...")
    processes = []
    for i, partition_range in enumerate(partition_ranges):
        print(f"  Worker {i}: partitions {partition_range}")
        proc = mp.Process(
            target=_worker,
            args=(search_proc_id, args.partition_dir, args.log_lvl, args.max_iters, partition_range)
        )
        proc.start()
        processes.append(proc)

    print(f"\nAll workers launched. Waiting for completion...")
    print("(Press Ctrl+C to interrupt)")
    print()

    try:
        for proc in processes:
            proc.join()
        print("\nAll workers completed successfully!")
    except KeyboardInterrupt:
        print("\nInterrupted by user, terminating workers...")
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.join()
        sys.exit(1)


def register_parser(subparsers):
    """Register the run subcommand."""
    parser = subparsers.add_parser('run', help='Run waterfill search on the database')
    parser.add_argument('partition_dir', help='Directory containing partition files')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of worker processes (default: 1)')
    parser.add_argument('--max-iters', type=int, default=None, help='Maximum iterations (default: unlimited)')
    parser.add_argument('--log-lvl', type=int, default=2, help='Log level: 0=FATAL, 1=SEVERE, 2=WARN, 3=INFO, 4=DEBUG (default: 2)')
    parser.set_defaults(func=run_command)
