#!/usr/bin/env python3
"""
CLI tool to interrogate partitioned CNF database state.

Usage:
    python scripts/db_status_partitioned.py <partition_dir> --global             # Show only global stats
    python scripts/db_status_partitioned.py <partition_dir> --partitions         # Show only partition breakdown
    python scripts/db_status_partitioned.py <partition_dir> --completion         # Check if endpoints reached
    python scripts/db_status_partitioned.py <partition_dir> --global --watch     # Live updating global dashboard
    python scripts/db_status_partitioned.py <partition_dir> --partitions --watch # Live updating partition dashboard
    python scripts/db_status_partitioned.py <partition_dir> --completion --watch # Monitor completion status
    python scripts/db_status_partitioned.py <partition_dir> --global --search 2  # Monitor search process #2
    python scripts/db_status_partitioned.py <partition_dir> --partitions --num-partitions 50  # Sample 50 partitions
    python scripts/db_status_partitioned.py <partition_dir> --partitions --summary  # Show summary statistics
"""

import argparse
import sqlite3
import time
import sys
import os
import glob
import statistics
import re
from datetime import datetime
from cnf.search import FRONTIER_WIDTH
from cnf.db.crystal_map_store import CrystalMapStore
from cnf.db.partitioned_db import PARTITION_SUFFIX, PartitionedDB
from cnf.db.meta_store import MetaStore
from cnf.db.constants import META_DB_NAME


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
        pattern = os.path.join(partition_dir, f"*{PARTITION_SUFFIX}")
        all_files = sorted(glob.glob(pattern))
        total_partitions = len(all_files)
        is_sampled = len(db_files) < total_partitions
        scaling_factor = total_partitions / len(db_files) if is_sampled else 1.0
    else:
        # Find all partition files
        pattern = os.path.join(partition_dir, f"*{PARTITION_SUFFIX}")
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
        'total_frontier_points': 0,
        'searched_points': 0,
        'num_partitions': len(db_files),
        'total_partitions': total_partitions,
        'is_sampled': is_sampled,
        'scaling_factor': scaling_factor,
        'per_partition': [],
        'start_points': []
    }

    cmap_store = CrystalMapStore.from_file(db_files[0])
    stats['metadata'] = cmap_store.get_metadata()

    # Connect to meta store to get water levels
    meta_db_path = os.path.join(partition_dir, META_DB_NAME)
    if os.path.exists(meta_db_path):
        meta_store = MetaStore.from_file(meta_db_path)
        stats['global_water_level'] = meta_store.get_global_water_level(search_id)
        stats['partition_water_levels'] = {}
        for i in range(total_partitions):
            partition_level = meta_store.get_partition_water_level(search_id, i)
            stats['partition_water_levels'][i] = partition_level
    else:
        stats['global_water_level'] = None
        stats['partition_water_levels'] = {}

    for db_file in db_files:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()

        # Extract partition number from filename (e.g., "0.partition.db" -> 0)
        match = re.search(r'(\d+)\.partition\.db', os.path.basename(db_file))
        partition_idx = int(match.group(1)) if match else None

        # Per-partition stats
        partition_stats = {
            'name': os.path.basename(db_file),
            'partition_idx': partition_idx,
            'total_points': 0,
            'explored_points': 0,
            'total_edges': 0,
            'frontier_points': 0,
            'searched_points': 0,
            'points_with_energy': 0,
            'incoming_points': 0,
            'water_level': stats['partition_water_levels'].get(partition_idx) if partition_idx is not None else None
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

        # Frontier points (OPEN status in search_point_status table)
        try:
            result = cur.execute("SELECT COUNT(*) FROM search_point_status WHERE search_id = ? AND point_status = 'OPEN'", (search_id,)).fetchone()
            partition_stats['frontier_points'] = result[0]
            stats['total_frontier_points'] += result[0]
        except sqlite3.OperationalError:
            # Table doesn't exist in this partition
            partition_stats['frontier_points'] = 0

        # Searched points count (CLOSED status in search_point_status table)
        try:
            result = cur.execute("SELECT COUNT(*) FROM search_point_status WHERE search_id = ? AND point_status = 'CLOSED'", (search_id,)).fetchone()
            partition_stats['searched_points'] = result[0]
            stats['searched_points'] += result[0]
        except sqlite3.OperationalError:
            pass

        # Incoming points in mailbox (waiting to be processed)
        try:
            result = cur.execute("SELECT COUNT(*) FROM incoming_point WHERE search_id = ?", (search_id,)).fetchone()
            partition_stats['incoming_points'] = result[0]
            if 'total_incoming_points' not in stats:
                stats['total_incoming_points'] = 0
            stats['total_incoming_points'] += result[0]
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

    # Deduplicate start points by CNF (same start point may be in multiple partitions)
    unique_start_points = {}
    for sp in stats['start_points']:
        cnf_key = sp['cnf']
        # Keep the one with energy if available, otherwise keep first encountered
        if cnf_key not in unique_start_points:
            unique_start_points[cnf_key] = sp
        elif sp['energy'] is not None and unique_start_points[cnf_key]['energy'] is None:
            unique_start_points[cnf_key] = sp
    stats['start_points'] = list(unique_start_points.values())

    # If sampling, scale up count statistics to estimate totals
    if is_sampled:
        stats['total_points'] = int(stats['total_points'] * scaling_factor)
        stats['points_with_energy'] = int(stats['points_with_energy'] * scaling_factor)
        stats['total_edges'] = int(stats['total_edges'] * scaling_factor)
        stats['explored_points'] = int(stats['explored_points'] * scaling_factor)
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

        incoming_rate = f" ({rates.get('total_incoming_points', 0):+.1f} pts/s)" if 'total_incoming_points' in rates else ""

        print(f"  Total Points:              {stats['total_points']:,}{total_points_rate}")
        print(f"  Neighbors Found (explored):{stats['explored_points']:,}{explored_rate}")
        print(f"  Points with Energy:        {stats['points_with_energy']:,}{energy_rate}")
        print(f"  Total Edges:               {stats['total_edges']:,}{edges_rate}")
        print(f"  Frontier Points:           {stats['total_frontier_points']:,}{frontier_rate}")
        print(f"  Searched Points:           {stats['searched_points']:,}{searched_rate}")
        print(f"  Incoming Points (Inbox):   {stats.get('total_incoming_points', 0):,}{incoming_rate}")

        if stats['global_min_energy'] is not None:
            print(f"  Global Energy Range:       [{stats['global_min_energy']:.8f}, {stats['global_max_energy']:.8f}]")
        else:
            print(f"  Global Energy Range:       No energies calculated")
        
        print()
        print("Metadata:")
        print(f"  xi: {stats['metadata'].xi}")
        print(f"  delta: {stats['metadata'].delta}")
        print(f"  elements: {stats['metadata'].element_list}")

        print()
        print("START POINTS (where water began):")
        if stats['start_points']:
            for i, sp in enumerate(stats['start_points'], 1):
                energy_str = format_value(sp['energy']) if sp['energy'] is not None else "Not calculated"
                print(f"  Start Point {i}:            {energy_str}")
        else:
            print("  No start points found")

        print()
        print("CURRENT WATER LEVEL (from Meta Store):")
        if stats.get('global_water_level') is not None:
            tolerance = FRONTIER_WIDTH  # Same as in the algorithm
            max_threshold = stats['global_water_level'] + tolerance
            print(f"  Global Water Level:        {format_value(stats['global_water_level'])}")
            print(f"  Max Threshold (+{FRONTIER_WIDTH}):  {format_value(max_threshold)}")
            print(f"  (Algorithm explores points with energy <= {format_value(max_threshold)})")
        else:
            print(f"  No water level data in meta store")

        print("=" * 60)

    # Summary statistics helper function
    def print_summary_stats():
        if 'per_partition' in stats and len(stats['per_partition']) > 0:
            print()
            print("SUMMARY STATISTICS (across sampled partitions):")
            print("-" * 117)

            # Extract values for each column
            columns = {
                'Points': [p['total_points'] for p in stats['per_partition']],
                'W/Energy': [p['points_with_energy'] for p in stats['per_partition']],
                'Explored': [p['explored_points'] for p in stats['per_partition']],
                'Searched': [p['searched_points'] for p in stats['per_partition']],
                'Frontier': [p['frontier_points'] for p in stats['per_partition']],
                'Inbox': [p['incoming_points'] for p in stats['per_partition']],
                'Water Lvl': [p.get('water_level') for p in stats['per_partition'] if p.get('water_level') is not None],
                'Edges': [p['total_edges'] for p in stats['per_partition']]
            }

            # Calculate and display statistics
            print(f"{'Statistic':<30} {'Points':>10} {'W/Energy':>10} {'Explored':>10} {'Searched':>10} {'Frontier':>10} {'Inbox':>10} {'Water Lvl':>12} {'Edges':>10}")
            print("-" * 135)

            # Average
            averages = {name: statistics.mean(values) if len(values) > 0 else 0 for name, values in columns.items()}
            water_avg = f"{averages.get('Water Lvl', 0):.6f}" if averages.get('Water Lvl', 0) > 0 else "N/A"
            print(f"{'Average':<30} {averages['Points']:>10.1f} {averages['W/Energy']:>10.1f} {averages['Explored']:>10.1f} {averages['Searched']:>10.1f} {averages['Frontier']:>10.1f} {averages['Inbox']:>10.1f} {water_avg:>12} {averages['Edges']:>10.1f}")

            # Median
            medians = {name: statistics.median(values) if len(values) > 0 else 0 for name, values in columns.items()}
            water_med = f"{medians.get('Water Lvl', 0):.6f}" if medians.get('Water Lvl', 0) > 0 else "N/A"
            print(f"{'Median':<30} {medians['Points']:>10.0f} {medians['W/Energy']:>10.0f} {medians['Explored']:>10.0f} {medians['Searched']:>10.0f} {medians['Frontier']:>10.0f} {medians['Inbox']:>10.0f} {water_med:>12} {medians['Edges']:>10.0f}")

            # Mode (with error handling for no unique mode)
            modes = {}
            for name, values in columns.items():
                try:
                    modes[name] = statistics.mode(values)
                except statistics.StatisticsError:
                    # No unique mode
                    modes[name] = None

            mode_str = f"{'Mode':<30}"
            for col_name in ['Points', 'W/Energy', 'Explored', 'Searched', 'Frontier', 'Inbox']:
                if modes.get(col_name) is not None:
                    mode_str += f" {modes[col_name]:>10.0f}"
                else:
                    mode_str += f" {'N/A':>10}"
            # Water level - special handling
            if modes.get('Water Lvl') is not None:
                mode_str += f" {modes['Water Lvl']:>12.6f}"
            else:
                mode_str += f" {'N/A':>12}"
            # Edges
            if modes.get('Edges') is not None:
                mode_str += f" {modes['Edges']:>10.0f}"
            else:
                mode_str += f" {'N/A':>10}"
            print(mode_str)

            print("=" * 135)

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
        print("=" * 135)
        print(f"{'Partition':<30} {'Points':>10} {'W/Energy':>10} {'Explored':>10} {'Searched':>10} {'Frontier':>10} {'Inbox':>10} {'Water Lvl':>12} {'Edges':>10}")
        print("-" * 135)
        for p in stats['per_partition']:
            water_lvl_str = format_value(p.get('water_level'), decimals=6) if p.get('water_level') is not None else "N/A"
            print(f"{p['name']:<30} {p['total_points']:>10,} {p['points_with_energy']:>10,} {p['explored_points']:>10,} {p['searched_points']:>10,} {p['frontier_points']:>10,} {p['incoming_points']:>10,} {water_lvl_str:>12} {p['total_edges']:>10,}")
        print("=" * 135)


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
                    rates['total_incoming_points'] = (stats.get('total_incoming_points', 0) - prev_stats.get('total_incoming_points', 0)) / time_delta

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


def gather_completion_data(partition_dir: str, search_id: int = 1):
    """Gather completion data for endpoints. Returns (completion_found, endpoint_data_list, num_endpoints)."""
    db = PartitionedDB(partition_dir, search_id)

    # Get endpoints from all partitions (endpoints are only in their correct partitions)
    endpoints = []
    for partition_idx in db.partition_range:
        search_store = db.get_search_store_by_idx(partition_idx)
        partition_endpoints = search_store.get_search_endpoints(search_id)
        endpoints.extend(partition_endpoints)

    if not endpoints:
        return None, None, 0

    endpoint_data_list = []
    completion_found = False

    for i, endpoint_pt in enumerate(endpoints, 1):
        endpoint_cnf = endpoint_pt.cnf

        # Determine which partition this endpoint belongs to
        partition_idx = db.get_partition_idx(endpoint_cnf)
        map_store = db.get_map_store_by_idx(partition_idx)
        search_store = db.get_search_store_by_idx(partition_idx)

        endpoint_data = {
            'index': i,
            'coords': endpoint_cnf.coords,
            'partition': partition_idx,
            'found': False
        }

        # Check if endpoint exists in the database
        point = map_store.get_point_by_cnf(endpoint_cnf)
        if point:
            endpoint_data['point_id'] = point.id
            endpoint_data['found'] = True

            # Check (a): Is it in the frontier?
            in_frontier_results = search_store.get_endpoint_ids_in_frontier(search_id)
            in_frontier = point.id in in_frontier_results
            endpoint_data['in_frontier'] = in_frontier

            # Check (b): Does it have connections?
            neighbors = map_store.get_neighbor_cnfs(point.id)
            num_neighbors = len(neighbors) if neighbors else 0
            endpoint_data['num_neighbors'] = num_neighbors

            # Check (c): Is it in the searched table?
            searched_points = search_store.get_searched_points_in_search(search_id)
            is_searched = any(sp.id == point.id for sp in searched_points)
            endpoint_data['is_searched'] = is_searched

            # Additional info
            endpoint_data['explored'] = point.explored
            endpoint_data['has_value'] = point.value is not None
            endpoint_data['value'] = point.value

            # Check completion criteria
            if in_frontier or num_neighbors > 0:
                endpoint_data['reached'] = True
                completion_found = True
            else:
                endpoint_data['reached'] = False

        endpoint_data_list.append(endpoint_data)

    return completion_found, endpoint_data_list, len(endpoints)


def display_completion_status(completion_found, endpoint_data_list, num_endpoints, search_id):
    """Display completion status from gathered data."""
    if endpoint_data_list is None:
        print(f"Error: No endpoints found for search_id {search_id}")
        return

    print("=" * 70)
    print(f"SEARCH COMPLETION CHECK (Search ID: {search_id})")
    print("=" * 70)
    print(f"Checking {num_endpoints} endpoint(s)...")
    print()

    for endpoint_data in endpoint_data_list:
        print(f"Endpoint {endpoint_data['index']}:")
        print(f"  CNF coords: {endpoint_data['coords']}")
        print(f"  Partition:  {endpoint_data['partition']}")

        if not endpoint_data['found']:
            print(f"  Status: Not found in database")
        else:
            print(f"  Point ID:    {endpoint_data['point_id']}")

            # Show value if computed
            if endpoint_data['has_value']:
                print(f"  Value:       {endpoint_data['value']:.8f}")
            else:
                print(f"  Value:       Not computed")

            print(f"  In frontier: {endpoint_data['in_frontier']}")
            print(f"  Searched:    {endpoint_data['is_searched']}")
            print(f"  Explored:    {endpoint_data['explored']}")
            print(f"  Connections: {endpoint_data['num_neighbors']}")

            if endpoint_data['reached']:
                print(f"  Status:      ✓ REACHED!")
            else:
                print(f"  Status:      ✗ Not yet reached")

        print()

    print("=" * 70)
    if completion_found:
        print("✓ SUCCESS: At least one endpoint has been reached!")
        print("  The search has found a path.")
    else:
        print("✗ INCOMPLETE: No endpoints reached yet")
        print("  Continue searching...")
    print("=" * 70)


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
    # Mutually exclusive group for display mode
    display_group = parser.add_mutually_exclusive_group(required=True)
    display_group.add_argument('--global', action='store_true',
                               help='Show only global statistics')
    display_group.add_argument('--partitions', action='store_true',
                               help='Show only partition breakdown')
    display_group.add_argument('--completion', action='store_true',
                               help='Show search completion status (check if endpoints reached)')

    parser.add_argument('--summary', action='store_true',
                        help='Show summary statistics (avg, median, mode) for partition columns (only with --partitions)')

    args = parser.parse_args()

    if not os.path.isdir(args.partition_dir):
        print(f"ERROR: {args.partition_dir} is not a directory")
        sys.exit(1)

    # Determine what to show based on mutually exclusive flags
    # Note: 'global' is a Python keyword, so use getattr
    if getattr(args, 'global', False):
        show_global = True
        show_partitions = False
        show_completion = False
    elif args.completion:
        show_global = False
        show_partitions = False
        show_completion = True
    else:  # args.partitions
        show_global = False
        show_partitions = True
        show_completion = False

    # Handle completion mode
    if show_completion:
        if args.watch:
            # Watch mode for completion
            try:
                iteration = 0
                while True:
                    # Query first
                    start_time = time.time()
                    completion_found, endpoint_data_list, num_endpoints = gather_completion_data(
                        args.partition_dir, args.search
                    )
                    query_time = time.time() - start_time

                    # Clear and display
                    clear_screen()
                    print(f"Search Completion Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Partition Directory: {args.partition_dir}")
                    print(f"Update #{iteration} (refreshing every {args.interval}s, Ctrl+C to exit)")
                    print()

                    display_completion_status(completion_found, endpoint_data_list, num_endpoints, args.search)

                    print()
                    print(f"Refreshing every {args.interval}s... (query took {query_time:.2f}s)")
                    print("Press Ctrl+C to stop")

                    iteration += 1
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                sys.exit(0)
        else:
            # One-time completion check
            completion_found, endpoint_data_list, num_endpoints = gather_completion_data(
                args.partition_dir, args.search
            )
            display_completion_status(completion_found, endpoint_data_list, num_endpoints, args.search)
            sys.exit(0 if completion_found else 1)

    # Regular stats mode (global or partitions)
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
