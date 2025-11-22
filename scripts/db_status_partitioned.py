#!/usr/bin/env python3
"""
CLI tool to interrogate partitioned CNF database state.

Usage:
    python scripts/db_status_partitioned.py <partition_dir>                     # Show stats once
    python scripts/db_status_partitioned.py <partition_dir> --watch             # Live updating dashboard
    python scripts/db_status_partitioned.py <partition_dir> --global-only       # Show only global stats
    python scripts/db_status_partitioned.py <partition_dir> --partitions-only   # Show only partition breakdown
    python scripts/db_status_partitioned.py <partition_dir> --search 2          # Monitor search process #2
    python scripts/db_status_partitioned.py <partition_dir> --num-partitions 50 # Sample 50 random partitions
    python scripts/db_status_partitioned.py <partition_dir> --summary           # Show summary statistics (avg/median/mode)
"""

import argparse
import sqlite3
import time
import sys
import os
import glob
import statistics
from datetime import datetime


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def get_partitioned_stats(partition_dir, search_id=1, sample_partitions=None, db_files_list=None):
    """Get aggregated statistics across all or sampled partition databases.

    Args:
        partition_dir: Directory containing partition databases
        search_id: Search process ID to query
        sample_partitions: Number of partitions to randomly sample (None = all)
        db_files_list: Pre-selected list of db files to use (overrides sampling)
    """
    # Use provided db_files_list if available, otherwise find and sample
    if db_files_list is not None:
        db_files = db_files_list
        # Find total partitions for scaling calculation
        pattern = os.path.join(partition_dir, "graph_partition_*.db")
        all_files = sorted(glob.glob(pattern))
        total_partitions = len(all_files)
        is_sampled = len(db_files) < total_partitions
        scaling_factor = total_partitions / len(db_files) if is_sampled else 1.0
    else:
        # Find all partition files
        pattern = os.path.join(partition_dir, "graph_partition_*.db")
        db_files = sorted(glob.glob(pattern))

        if not db_files:
            raise ValueError(f"No partition files found in {partition_dir}")

        # Sample partitions if requested
        total_partitions = len(db_files)
        if sample_partitions is not None and sample_partitions < total_partitions:
            import random
            db_files = random.sample(db_files, sample_partitions)
            db_files = sorted(db_files)  # Sort sampled files for consistent display
            is_sampled = True
            scaling_factor = total_partitions / len(db_files)
        else:
            is_sampled = False
            scaling_factor = 1.0

    stats = {
        'total_points': 0,
        'points_with_energy': 0,
        'total_edges': 0,
        'explored_points': 0,
        'global_min_energy': None,
        'global_max_energy': None,
        'locked_points': 0,
        'total_frontier_points': 0,
        'searched_points': 0,
        'searched_min_energy': None,
        'searched_max_energy': None,
        'frontier_min_energy': None,
        'frontier_max_energy': None,
        'num_partitions': len(db_files),
        'total_partitions': total_partitions,
        'is_sampled': is_sampled,
        'scaling_factor': scaling_factor,
        'per_partition': [],
        'start_points': [],
        'current_water_level': None
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
            'frontier_points': 0,
            'searched_points': 0,
            'points_with_energy': 0
        }

        # Total points
        result = cur.execute("SELECT COUNT(*) FROM point").fetchone()
        partition_stats['total_points'] = result[0]
        stats['total_points'] += result[0]

        # Points with energy
        result = cur.execute("SELECT COUNT(*) FROM point WHERE value IS NOT NULL").fetchone()
        partition_stats['points_with_energy'] = result[0]
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
            result = cur.execute("SELECT COUNT(*) FROM search_frontier_member WHERE search_id = ?", (search_id,)).fetchone()
            partition_stats['frontier_points'] = result[0]
            stats['total_frontier_points'] += result[0]
        except sqlite3.OperationalError:
            # Table doesn't exist in this partition
            partition_stats['frontier_points'] = 0

        # Searched points count
        try:
            result = cur.execute("SELECT COUNT(*) FROM searched_point WHERE search_id = ?", (search_id,)).fetchone()
            partition_stats['searched_points'] = result[0]
            stats['searched_points'] += result[0]
        except sqlite3.OperationalError:
            pass

        # Energy range of searched points (below water surface)
        try:
            result = cur.execute(
                """SELECT MIN(pt.value), MAX(pt.value)
                   FROM searched_point AS sp
                   JOIN point AS pt ON pt.id = sp.point_id
                   WHERE sp.search_id = ? AND pt.value IS NOT NULL""",
                (search_id,)
            ).fetchone()
            if result[0] is not None:
                if stats['searched_min_energy'] is None:
                    stats['searched_min_energy'] = result[0]
                    stats['searched_max_energy'] = result[1]
                else:
                    stats['searched_min_energy'] = min(stats['searched_min_energy'], result[0])
                    stats['searched_max_energy'] = max(stats['searched_max_energy'], result[1])
        except sqlite3.OperationalError:
            pass

        # Energy range of frontier points (water surface)
        try:
            result = cur.execute(
                """SELECT MIN(pt.value), MAX(pt.value)
                   FROM search_frontier_member AS fm
                   JOIN point AS pt ON pt.id = fm.point_id
                   WHERE fm.search_id = ? AND pt.value IS NOT NULL""",
                (search_id,)
            ).fetchone()
            if result[0] is not None:
                if stats['frontier_min_energy'] is None:
                    stats['frontier_min_energy'] = result[0]
                    stats['frontier_max_energy'] = result[1]
                else:
                    stats['frontier_min_energy'] = min(stats['frontier_min_energy'], result[0])
                    stats['frontier_max_energy'] = max(stats['frontier_max_energy'], result[1])
        except sqlite3.OperationalError:
            pass

        # Get start points with their energies
        try:
            result = cur.execute(
                """SELECT pt.id, pt.value, pt.cnf
                   FROM search_start_point AS sp
                   JOIN point AS pt ON pt.id = sp.start_point_id
                   WHERE sp.search_id = ?""",
                (search_id,)
            ).fetchall()
            for row in result:
                stats['start_points'].append({
                    'id': row[0],
                    'energy': row[1],
                    'cnf': row[2],
                    'partition': os.path.basename(db_file)
                })
        except sqlite3.OperationalError as e:
            # Silently skip if table doesn't exist
            pass

        stats['per_partition'].append(partition_stats)
        conn.close()

    # Current water level is the minimum frontier energy
    stats['current_water_level'] = stats['frontier_min_energy']

    # If sampling, scale up count statistics to estimate totals
    if is_sampled:
        stats['total_points'] = int(stats['total_points'] * scaling_factor)
        stats['points_with_energy'] = int(stats['points_with_energy'] * scaling_factor)
        stats['total_edges'] = int(stats['total_edges'] * scaling_factor)
        stats['explored_points'] = int(stats['explored_points'] * scaling_factor)
        stats['locked_points'] = int(stats['locked_points'] * scaling_factor)
        stats['total_frontier_points'] = int(stats['total_frontier_points'] * scaling_factor)
        stats['searched_points'] = int(stats['searched_points'] * scaling_factor)
        # Note: Energy min/max values are NOT scaled as they represent actual energy values

    return stats


def format_value(value, decimals=8):
    """Format a numeric value, handling None."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def display_stats(stats, rates=None, show_global=True, show_partitions=True, show_summary=False):
    """Display statistics in same format as original db_status.py."""
    if rates is None:
        rates = {}

    if show_global:
        print("=" * 60)
        if stats['is_sampled']:
            print(f"GLOBAL STATISTICS (sampled from {stats['num_partitions']} of {stats['total_partitions']} partitions):")
            print(f"⚠️  Estimated totals (scaled by {stats['scaling_factor']:.1f}x)")
        else:
            print(f"GLOBAL STATISTICS (aggregated across {stats['num_partitions']} partitions):")
        print("=" * 60)

        total_points_rate = f" ({rates.get('total_points', 0):+.1f} pts/s)" if 'total_points' in rates else ""
        energy_rate = f" ({rates.get('points_with_energy', 0):+.1f} pts/s)" if 'points_with_energy' in rates else ""
        explored_rate = f" ({rates.get('explored_points', 0):+.1f} pts/s)" if 'explored_points' in rates else ""
        edges_rate = f" ({rates.get('total_edges', 0):+.1f} edges/s)" if 'total_edges' in rates else ""
        frontier_rate = f" ({rates.get('total_frontier_points', 0):+.1f} pts/s)" if 'total_frontier_points' in rates else ""
        searched_rate = f" ({rates.get('searched_points', 0):+.1f} pts/s)" if 'searched_points' in rates else ""

        print(f"  Total Points:              {stats['total_points']:,}{total_points_rate}")
        print(f"  Neighbors Found (explored):{stats['explored_points']:,}{explored_rate}")
        print(f"  Points with Energy:        {stats['points_with_energy']:,}{energy_rate}")
        print(f"  Total Edges:               {stats['total_edges']:,}{edges_rate}")
        print(f"  Locked Points:             {stats['locked_points']:,}")
        print(f"  Frontier Points:           {stats['total_frontier_points']:,}{frontier_rate}")
        print(f"  Searched Points:           {stats['searched_points']:,}{searched_rate}")

        if stats['global_min_energy'] is not None:
            print(f"  Global Energy Range:       [{stats['global_min_energy']:.8f}, {stats['global_max_energy']:.8f}]")
        else:
            print(f"  Global Energy Range:       No energies calculated")

        print()
        print("START POINTS (where water began):")
        if stats['start_points']:
            for i, sp in enumerate(stats['start_points'], 1):
                energy_str = format_value(sp['energy']) if sp['energy'] is not None else "Not calculated"
                print(f"  Start Point {i}:            {energy_str}")
        else:
            print("  No start points found")

        print()
        print("CURRENT WATER LEVEL:")
        if stats['current_water_level'] is not None:
            tolerance = 0.001  # Same as in the algorithm (1 meV)
            max_threshold = stats['current_water_level'] + tolerance
            print(f"  Lowest Frontier Energy:    {format_value(stats['current_water_level'])}")
            print(f"  Max Threshold (+1meV):     {format_value(max_threshold)}")
            print(f"  (Algorithm explores points with energy <= {format_value(max_threshold)})")
        else:
            print(f"  No frontier points with energy")

        print()
        print("SEARCHED REGION (below water surface):")
        print(f"  Energy Range:              {format_value(stats['searched_min_energy'])} to {format_value(stats['searched_max_energy'])}")
        searched_range = stats['searched_max_energy'] - stats['searched_min_energy'] if stats['searched_max_energy'] and stats['searched_min_energy'] else None
        print(f"  Range Width:               {format_value(searched_range)}")

        print()
        print("FRONTIER (water surface - being explored):")
        print(f"  Energy Range:              {format_value(stats['frontier_min_energy'])} to {format_value(stats['frontier_max_energy'])}")
        frontier_range = stats['frontier_max_energy'] - stats['frontier_min_energy'] if stats['frontier_max_energy'] and stats['frontier_min_energy'] else None
        print(f"  Range Width:               {format_value(frontier_range)}")

        print("=" * 60)

    # Summary statistics helper function
    def print_summary_stats():
        if 'per_partition' in stats and len(stats['per_partition']) > 0:
            print()
            print("SUMMARY STATISTICS (across sampled partitions):")
            print("-" * 105)

            # Extract values for each column
            columns = {
                'Points': [p['total_points'] for p in stats['per_partition']],
                'W/Energy': [p['points_with_energy'] for p in stats['per_partition']],
                'Explored': [p['explored_points'] for p in stats['per_partition']],
                'Searched': [p['searched_points'] for p in stats['per_partition']],
                'Frontier': [p['frontier_points'] for p in stats['per_partition']],
                'Edges': [p['total_edges'] for p in stats['per_partition']]
            }

            # Calculate and display statistics
            print(f"{'Statistic':<30} {'Points':>10} {'W/Energy':>10} {'Explored':>10} {'Searched':>10} {'Frontier':>10} {'Edges':>10}")
            print("-" * 105)

            # Average
            averages = {name: statistics.mean(values) for name, values in columns.items()}
            print(f"{'Average':<30} {averages['Points']:>10.1f} {averages['W/Energy']:>10.1f} {averages['Explored']:>10.1f} {averages['Searched']:>10.1f} {averages['Frontier']:>10.1f} {averages['Edges']:>10.1f}")

            # Median
            medians = {name: statistics.median(values) for name, values in columns.items()}
            print(f"{'Median':<30} {medians['Points']:>10.0f} {medians['W/Energy']:>10.0f} {medians['Explored']:>10.0f} {medians['Searched']:>10.0f} {medians['Frontier']:>10.0f} {medians['Edges']:>10.0f}")

            # Mode (with error handling for no unique mode)
            modes = {}
            for name, values in columns.items():
                try:
                    modes[name] = statistics.mode(values)
                except statistics.StatisticsError:
                    # No unique mode
                    modes[name] = None

            mode_str = f"{'Mode':<30}"
            for col_name in ['Points', 'W/Energy', 'Explored', 'Searched', 'Frontier', 'Edges']:
                if modes[col_name] is not None:
                    mode_str += f" {modes[col_name]:>10.0f}"
                else:
                    mode_str += f" {'N/A':>10}"
            print(mode_str)

            print("=" * 105)

    # Show summary in global-only mode (below global stats)
    if show_summary and show_global and not show_partitions:
        print_summary_stats()

    # Per-partition breakdown
    if show_partitions and 'per_partition' in stats and stats['per_partition']:
        # Show summary above partition table if requested
        if show_summary:
            print_summary_stats()

        print()
        print("PER-PARTITION BREAKDOWN:")
        print("=" * 105)
        print(f"{'Partition':<30} {'Points':>10} {'W/Energy':>10} {'Explored':>10} {'Searched':>10} {'Frontier':>10} {'Edges':>10}")
        print("-" * 105)
        for p in stats['per_partition']:
            print(f"{p['name']:<30} {p['total_points']:>10,} {p['points_with_energy']:>10,} {p['explored_points']:>10,} {p['searched_points']:>10,} {p['frontier_points']:>10,} {p['total_edges']:>10,}")
        print("=" * 105)


def watch_mode(partition_dir, interval=1.0, search_id=1, show_global=True, show_partitions=True, sample_partitions=None, show_summary=False):
    """Live updating dashboard mode."""
    try:
        # Sample partition files once before entering loop (for consistency across refreshes)
        sampled_db_files = None
        if sample_partitions is not None:
            import random
            pattern = os.path.join(partition_dir, "graph_partition_*.db")
            all_db_files = sorted(glob.glob(pattern))

            if not all_db_files:
                raise ValueError(f"No partition files found in {partition_dir}")

            if sample_partitions < len(all_db_files):
                sampled_db_files = random.sample(all_db_files, sample_partitions)
                sampled_db_files = sorted(sampled_db_files)  # Sort for consistent display

        iteration = 0
        prev_stats = None
        prev_time = None

        while True:
            # Query FIRST (so screen isn't blank while waiting)
            start_time = time.time()
            stats = get_partitioned_stats(partition_dir, search_id=search_id,
                                         sample_partitions=sample_partitions,
                                         db_files_list=sampled_db_files)
            query_time = time.time() - start_time
            current_time = time.time()

            # Calculate rates
            rates = {}
            if prev_stats is not None and prev_time is not None:
                time_delta = current_time - prev_time
                if time_delta > 0:
                    rates['total_points'] = (stats['total_points'] - prev_stats['total_points']) / time_delta
                    rates['points_with_energy'] = (stats['points_with_energy'] - prev_stats['points_with_energy']) / time_delta
                    rates['explored_points'] = (stats['explored_points'] - prev_stats['explored_points']) / time_delta
                    rates['total_edges'] = (stats['total_edges'] - prev_stats['total_edges']) / time_delta
                    rates['total_frontier_points'] = (stats['total_frontier_points'] - prev_stats['total_frontier_points']) / time_delta
                    rates['searched_points'] = (stats['searched_points'] - prev_stats['searched_points']) / time_delta

            # THEN clear and display
            clear_screen()
            print(f"CNF Partitioned Database Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Partition Directory: {partition_dir}")
            print(f"Update #{iteration} (refreshing every {interval}s, Ctrl+C to exit)")
            print()

            display_stats(stats, rates=rates, show_global=show_global, show_partitions=show_partitions, show_summary=show_summary)

            print()
            print(f"Refreshing every {interval}s... (query took {query_time:.2f}s, iteration #{iteration})")
            print("Press Ctrl+C to stop")

            # If queries are slow, warn user
            if query_time > interval * 0.8:
                print(f"\n⚠️  Warning: Queries taking {query_time:.2f}s (longer than refresh interval)")
                print(f"   Consider using --interval {max(2.0, query_time * 1.5):.1f} for smoother updates")

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
    parser.add_argument('--search', '-s', type=int, default=1, metavar='ID',
                        help='Search process ID to monitor (default: 1)')
    parser.add_argument('--num-partitions', '-n', type=int, default=None, metavar='N',
                        help='Number of partitions to randomly sample (default: all)')
    parser.add_argument('--global-only', action='store_true',
                        help='Show only global statistics (hide partition breakdown)')
    parser.add_argument('--partitions-only', action='store_true',
                        help='Show only partition breakdown (hide global statistics)')
    parser.add_argument('--summary', action='store_true',
                        help='Show summary statistics (avg, median, mode) for partition columns')

    args = parser.parse_args()

    if not os.path.isdir(args.partition_dir):
        print(f"ERROR: {args.partition_dir} is not a directory")
        sys.exit(1)

    # Determine what to show
    show_global = not args.partitions_only
    show_partitions = not args.global_only

    if args.watch:
        watch_mode(args.partition_dir, interval=args.interval, search_id=args.search,
                   show_global=show_global, show_partitions=show_partitions,
                   sample_partitions=args.num_partitions, show_summary=args.summary)
    else:
        stats = get_partitioned_stats(args.partition_dir, search_id=args.search,
                                     sample_partitions=args.num_partitions)
        display_stats(stats, show_global=show_global, show_partitions=show_partitions, show_summary=args.summary)


if __name__ == '__main__':
    main()
