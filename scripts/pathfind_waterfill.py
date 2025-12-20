#!/usr/bin/env python3
"""
Run water-filling pathfinding search on a partitioned CNF database.

Usage:
    cnf-pathfind-waterfill <partitions_dir>
    cnf-pathfind-waterfill <partitions_dir> --max-iters 1000 --log-lvl 1
"""

import argparse
from cnf.search.waterfill import continue_search_waterfill
from cnf.db.partitioned_db import PartitionedDB
from cnf.navigation.search_filters import VolumeLimitFilter, MinDistanceFilter
from cnf.calculation.grace import GraceCalculator


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
    parser.add_argument('--max-iters', type=int, default=None,
                       help='Maximum number of iterations (default: no limit)')
    parser.add_argument('--log-lvl', type=int, default=2,
                       help='Logging level: 0=FATAL, 1=SEVERE, 2=WARN, 3=INFO, 4=DEBUG (default: 2)')

    args = parser.parse_args()

    print(f"Using partition directory {args.partitions_dir}...")
    print(f"Max iterations: {args.max_iters if args.max_iters else 'unlimited'}")
    print(f"Log level: {args.log_lvl}")
    print()

    search_proc_id = 1

    continue_search_waterfill(
        search_proc_id,
        args.partitions_dir,
        GraceCalculator(),
        log_lvl=args.log_lvl,
        max_iters=args.max_iters
    )

if __name__ == "__main__":
    main()