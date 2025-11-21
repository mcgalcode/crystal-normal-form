#!/usr/bin/env python3
"""
CLI tool to interrogate partitioned CNF database state.

Usage:
    python scripts/db_status_partitioned.py <partition_dir>                  # Show stats once
    python scripts/db_status_partitioned.py <partition_dir> --watch          # Live updating dashboard
"""

import argparse
import sqlite3
import time
import sys
import os
import glob
from datetime import datetime


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def get_partitioned_stats(partition_dir):
    """Get aggregated statistics across all partition databases."""
    # Find all partition files
    pattern = os.path.join(partition_dir, "graph_partition_*.db")
    db_files = sorted(glob.glob(pattern))

    if not db_files:
        raise ValueError(f"No partition files found in {partition_dir}")

    stats = {
        'total_points': 0,
        'points_with_energy': 0,
        'total_edges': 0,
        'explored_points': 0,
        'global_min_energy': None,
        'global_max_energy': None,
        'locked_points': 0,
        'total_frontier_points': 0,
        'num_partitions': len(db_files),
        'per_partition': []
    }

    for db_file in db_files:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()

        # Per-partition stats
        partition_stats = {
            'name': os.path.basename(db_file),
            'total_points': 0,
            'explored_points': 0,
            'total_edges': 0,
            'locked_points': 0,
            'frontier_points': 0
        }

        # Total points
        result = cur.execute("SELECT COUNT(*) FROM point").fetchone()
        partition_stats['total_points'] = result[0]
        stats['total_points'] += result[0]

        # Points with energy
        result = cur.execute("SELECT COUNT(*) FROM point WHERE value IS NOT NULL").fetchone()
        stats['points_with_energy'] += result[0]

        # Total edges
        result = cur.execute("SELECT COUNT(*) FROM edge").fetchone()
        partition_stats['total_edges'] = result[0]
        stats['total_edges'] += result[0]

        # Explored points
        result = cur.execute("SELECT COUNT(*) FROM point WHERE explored = 1").fetchone()
        partition_stats['explored_points'] = result[0]
        stats['explored_points'] += result[0]

        # Energy range
        result = cur.execute("SELECT MIN(value), MAX(value) FROM point WHERE value IS NOT NULL").fetchone()
        if result[0] is not None:
            if stats['global_min_energy'] is None:
                stats['global_min_energy'] = result[0]
                stats['global_max_energy'] = result[1]
            else:
                stats['global_min_energy'] = min(stats['global_min_energy'], result[0])
                stats['global_max_energy'] = max(stats['global_max_energy'], result[1])

        result = cur.execute("SELECT COUNT(*) FROM lock").fetchone()
        partition_stats['locked_points'] = result[0]
        stats['locked_points'] += result[0]

        # Frontier points (check if search_frontier table exists)
        try:
            result = cur.execute("SELECT COUNT(*) FROM search_frontier_member").fetchone()
            partition_stats['frontier_points'] = result[0]
            stats['total_frontier_points'] += result[0]
        except sqlite3.OperationalError:
            # Table doesn't exist in this partition
            partition_stats['frontier_points'] = 0

        stats['per_partition'].append(partition_stats)
        conn.close()

    return stats


def display_stats(stats, rates=None):
    """Display statistics in same format as original db_status.py."""
    if rates is None:
        rates = {}

    print("=" * 60)
    print(f"GLOBAL STATISTICS (aggregated across {stats['num_partitions']} partitions):")
    print("=" * 60)

    total_points_rate = f" ({rates.get('total_points', 0):+.1f} pts/s)" if 'total_points' in rates else ""
    explored_rate = f" ({rates.get('explored_points', 0):+.1f} pts/s)" if 'explored_points' in rates else ""
    edges_rate = f" ({rates.get('total_edges', 0):+.1f} edges/s)" if 'total_edges' in rates else ""
    frontier_rate = f" ({rates.get('total_frontier_points', 0):+.1f} pts/s)" if 'total_frontier_points' in rates else ""

    print(f"  Total Points:              {stats['total_points']:,}{total_points_rate}")
    print(f"  Neighbors Found (explored):{stats['explored_points']:,}{explored_rate}")
    print(f"  Points with Energy:        {stats['points_with_energy']:,}")
    print(f"  Total Edges:               {stats['total_edges']:,}{edges_rate}")
    print(f"  Locked Points:             {stats['locked_points']:,}")
    print(f"  Frontier Points:           {stats['total_frontier_points']:,}{frontier_rate}")

    if stats['global_min_energy'] is not None:
        print(f"  Global Energy Range:       [{stats['global_min_energy']:.4f}, {stats['global_max_energy']:.4f}]")
    else:
        print(f"  Global Energy Range:       No energies calculated")

    print("=" * 60)

    # Per-partition breakdown
    if 'per_partition' in stats and stats['per_partition']:
        print()
        print("PER-PARTITION BREAKDOWN:")
        print("=" * 75)
        print(f"{'Partition':<30} {'Points':>10} {'Explored':>10} {'Frontier':>10} {'Edges':>10}")
        print("-" * 75)
        for p in stats['per_partition']:
            print(f"{p['name']:<30} {p['total_points']:>10,} {p['explored_points']:>10,} {p['frontier_points']:>10,} {p['total_edges']:>10,}")
        print("=" * 75)


def watch_mode(partition_dir, interval=1.0):
    """Live updating dashboard mode."""
    try:
        iteration = 0
        prev_stats = None
        prev_time = None

        while True:

            stats = get_partitioned_stats(partition_dir)
            current_time = time.time()

            # Calculate rates
            rates = {}
            if prev_stats is not None and prev_time is not None:
                time_delta = current_time - prev_time
                if time_delta > 0:
                    rates['total_points'] = (stats['total_points'] - prev_stats['total_points']) / time_delta
                    rates['explored_points'] = (stats['explored_points'] - prev_stats['explored_points']) / time_delta
                    rates['total_edges'] = (stats['total_edges'] - prev_stats['total_edges']) / time_delta
                    rates['total_frontier_points'] = (stats['total_frontier_points'] - prev_stats['total_frontier_points']) / time_delta

            # Display header
            clear_screen()
            print(f"CNF Partitioned Database Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Partition Directory: {partition_dir}")
            print(f"Update #{iteration} (refreshing every {interval}s, Ctrl+C to exit)")
            print()

            display_stats(stats, rates=rates)

            print()
            print(f"Last update: {datetime.now().strftime('%H:%M:%S')}")

            prev_stats = stats.copy()
            prev_time = current_time
            iteration += 1

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Interrogate partitioned CNF database state',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('partition_dir',
                        help='Directory containing graph_partition_*.db files')
    parser.add_argument('--watch', action='store_true',
                        help='Live updating dashboard mode')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Update interval in seconds for watch mode (default: 1.0)')

    args = parser.parse_args()

    if not os.path.isdir(args.partition_dir):
        print(f"ERROR: {args.partition_dir} is not a directory")
        sys.exit(1)

    if args.watch:
        watch_mode(args.partition_dir, interval=args.interval)
    else:
        stats = get_partitioned_stats(args.partition_dir)
        display_stats(stats)


if __name__ == '__main__':
    main()
