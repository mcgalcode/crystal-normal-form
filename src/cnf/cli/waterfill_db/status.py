"""Status command for waterfill database."""

import sys
import os
import time
import glob
import statistics
from datetime import datetime

from cnf.db.crystal_map_store import CrystalMapStore
from cnf.db.search_store import SearchProcessStore
from cnf.db.partitioned_db import PARTITION_SUFFIX, PartitionedDB
from cnf.db.meta_store import MetaStore
from cnf.db.constants import META_DB_NAME


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def format_value(value, decimals=8):
    """Format a numeric value, handling None."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def get_stats_from_metastore(partition_dir, search_id=1):
    """Get statistics from metastore (fast - no partition scanning)."""
    meta_db_path = os.path.join(partition_dir, META_DB_NAME)
    meta_store = MetaStore.from_file(meta_db_path)

    partition_stats_list = meta_store.get_all_partition_stats(search_id)

    pattern = os.path.join(partition_dir, f"*{PARTITION_SUFFIX}")
    db_files = sorted(glob.glob(pattern))
    if not db_files:
        raise ValueError(f"No partition files found in {partition_dir}")

    cmap_store = CrystalMapStore.from_file(db_files[0])
    metadata = cmap_store.get_metadata()

    db = PartitionedDB(partition_dir, search_id)
    frontier_width = db.reload_frontier_width()

    stats = {
        'total_points': 0,
        'points_with_energy': 0,
        'total_edges': 0,
        'explored_points': 0,
        'global_min_energy': None,
        'global_max_energy': None,
        'global_max_searched_energy': None,
        'total_frontier_points': 0,
        'searched_points': 0,
        'total_incoming_points': 0,
        'num_partitions': len(partition_stats_list),
        'total_partitions': len(db_files),
        'is_sampled': False,
        'scaling_factor': 1.0,
        'per_partition': [],
        'start_points': [],
        'metadata': metadata,
        'global_water_level': meta_store.get_global_water_level(search_id),
        'partition_water_levels': {},
        'frontier_width': frontier_width
    }

    for pstats in partition_stats_list:
        partition_idx = pstats['partition_number']

        stats['total_points'] += pstats['total_points']
        stats['points_with_energy'] += pstats['points_with_energy']
        stats['explored_points'] += pstats['explored_points']
        stats['total_edges'] += pstats['total_edges']
        stats['total_frontier_points'] += pstats['frontier_points']
        stats['searched_points'] += pstats['searched_points']
        stats['total_incoming_points'] += pstats['inbox_size']

        if pstats['min_energy'] is not None:
            if stats['global_min_energy'] is None:
                stats['global_min_energy'] = pstats['min_energy']
                stats['global_max_energy'] = pstats['max_energy']
            else:
                stats['global_min_energy'] = min(stats['global_min_energy'], pstats['min_energy'])
                stats['global_max_energy'] = max(stats['global_max_energy'], pstats['max_energy'])

        if pstats['max_searched_energy'] is not None:
            if stats['global_max_searched_energy'] is None:
                stats['global_max_searched_energy'] = pstats['max_searched_energy']
            else:
                stats['global_max_searched_energy'] = max(stats['global_max_searched_energy'], pstats['max_searched_energy'])

        stats['per_partition'].append({
            'name': f"{partition_idx}.partition.db",
            'partition_idx': partition_idx,
            'total_points': pstats['total_points'],
            'explored_points': pstats['explored_points'],
            'total_edges': pstats['total_edges'],
            'frontier_points': pstats['frontier_points'],
            'searched_points': pstats['searched_points'],
            'points_with_energy': pstats['points_with_energy'],
            'incoming_points': pstats['inbox_size'],
            'water_level': pstats['min_frontier_energy']
        })

        stats['partition_water_levels'][partition_idx] = pstats['min_frontier_energy']

    search_store = SearchProcessStore.from_file(db_files[0])
    start_points = search_store.get_search_startpoints(search_id)
    for sp in start_points:
        stats['start_points'].append({
            'id': sp.id,
            'energy': sp.value,
            'cnf': sp.cnf,
            'partition': os.path.basename(db_files[0])
        })

    return stats


def display_stats(stats, rates=None, show_global=True, show_partitions=True, show_summary=False):
    """Display statistics."""
    if rates is None:
        rates = {}

    if show_global:
        print("=" * 60)
        if stats['is_sampled']:
            print(f"GLOBAL STATISTICS (sampled from {stats['num_partitions']} of {stats['total_partitions']} partitions):")
            print(f"  Estimated totals (scaled by {stats['scaling_factor']:.1f}x)")
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
            tolerance = stats['frontier_width']
            max_threshold = stats['global_water_level'] + tolerance
            print(f"  Global Water Level:        {format_value(stats['global_water_level'])}")
            print(f"  Max Threshold (+{stats['frontier_width']}):  {format_value(max_threshold)}")
            print(f"  (Algorithm explores points with energy <= {format_value(max_threshold)})")

            if stats.get('global_max_searched_energy') is not None:
                max_searched = stats['global_max_searched_energy']
                print(f"\n  High Water Mark (max searched): {format_value(max_searched)}")
                if max_searched > stats['global_water_level']:
                    diff = max_searched - stats['global_water_level']
                    print(f"  Water flowing DOWNHILL (descended {format_value(diff, 6)} from peak)")
                else:
                    print(f"  Water still rising")
        else:
            print(f"  No water level data in meta store")

        print("=" * 60)

    def print_summary_stats():
        if 'per_partition' in stats and len(stats['per_partition']) > 0:
            print()
            print("SUMMARY STATISTICS (across sampled partitions):")
            print("-" * 117)

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

            print(f"{'Statistic':<30} {'Points':>10} {'W/Energy':>10} {'Explored':>10} {'Searched':>10} {'Frontier':>10} {'Inbox':>10} {'Water Lvl':>12} {'Edges':>10}")
            print("-" * 135)

            averages = {name: statistics.mean(values) if len(values) > 0 else 0 for name, values in columns.items()}
            water_avg = f"{averages.get('Water Lvl', 0):.6f}" if averages.get('Water Lvl', 0) > 0 else "N/A"
            print(f"{'Average':<30} {averages['Points']:>10.1f} {averages['W/Energy']:>10.1f} {averages['Explored']:>10.1f} {averages['Searched']:>10.1f} {averages['Frontier']:>10.1f} {averages['Inbox']:>10.1f} {water_avg:>12} {averages['Edges']:>10.1f}")

            medians = {name: statistics.median(values) if len(values) > 0 else 0 for name, values in columns.items()}
            water_med = f"{medians.get('Water Lvl', 0):.6f}" if medians.get('Water Lvl', 0) > 0 else "N/A"
            print(f"{'Median':<30} {medians['Points']:>10.0f} {medians['W/Energy']:>10.0f} {medians['Explored']:>10.0f} {medians['Searched']:>10.0f} {medians['Frontier']:>10.0f} {medians['Inbox']:>10.0f} {water_med:>12} {medians['Edges']:>10.0f}")

            print("=" * 135)

    if show_summary and show_global and not show_partitions:
        print_summary_stats()

    if show_partitions and 'per_partition' in stats and stats['per_partition']:
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
        iteration = 0
        initial_stats = None
        initial_time = None

        while True:
            start_time = time.time()
            stats = get_stats_from_metastore(partition_dir, search_id=search_id)
            query_time = time.time() - start_time
            current_time = time.time()

            rates = {}
            if initial_stats is None:
                initial_stats = stats.copy()
                initial_time = current_time
            else:
                time_delta = current_time - initial_time
                if time_delta > 0:
                    rates['total_points'] = (stats['total_points'] - initial_stats['total_points']) / time_delta
                    rates['points_with_energy'] = (stats['points_with_energy'] - initial_stats['points_with_energy']) / time_delta
                    rates['explored_points'] = (stats['explored_points'] - initial_stats['explored_points']) / time_delta
                    rates['total_edges'] = (stats['total_edges'] - initial_stats['total_edges']) / time_delta
                    rates['total_frontier_points'] = (stats['total_frontier_points'] - initial_stats['total_frontier_points']) / time_delta
                    rates['searched_points'] = (stats['searched_points'] - initial_stats['searched_points']) / time_delta
                    rates['total_incoming_points'] = (stats.get('total_incoming_points', 0) - initial_stats.get('total_incoming_points', 0)) / time_delta

            clear_screen()
            print(f"CNF Partitioned Database Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Partition Directory: {partition_dir}")
            if initial_stats is not None:
                elapsed = current_time - initial_time
                print(f"Update #{iteration} (refreshing every {interval}s, running for {elapsed:.1f}s, Ctrl+C to exit)")
                print(f"Rates shown are averages since monitoring started")
            else:
                print(f"Update #{iteration} (refreshing every {interval}s, Ctrl+C to exit)")
            print()

            display_stats(stats, rates=rates, show_global=show_global, show_partitions=show_partitions, show_summary=show_summary)

            print()
            print(f"Refreshing every {interval}s... (query took {query_time:.2f}s, iteration #{iteration})")
            print("Press Ctrl+C to stop")

            if query_time > interval * 0.8:
                print(f"\nWarning: Queries taking {query_time:.2f}s (longer than refresh interval)")
                print(f"   Consider using --interval {max(2.0, query_time * 1.5):.1f} for smoother updates")

            iteration += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        sys.exit(0)


def status_command(args):
    """Show database statistics."""
    partition_dir = args.partition_dir
    search_id = args.search_id

    if not os.path.isdir(partition_dir):
        print(f"Error: {partition_dir} is not a directory")
        sys.exit(1)

    show_global = args.show_global and not args.partitions
    show_partitions = args.partitions

    if args.watch:
        watch_mode(
            partition_dir,
            interval=args.interval,
            search_id=search_id,
            show_global=show_global,
            show_partitions=show_partitions,
            sample_partitions=args.num_partitions,
            show_summary=args.summary
        )
    else:
        stats = get_stats_from_metastore(partition_dir, search_id=search_id)
        display_stats(stats, show_global=show_global, show_partitions=show_partitions, show_summary=args.summary)


def register_parser(subparsers):
    """Register the status subcommand."""
    parser = subparsers.add_parser('status', help='Show database statistics')
    parser.add_argument('partition_dir', help='Directory containing partition files')
    parser.add_argument('--search-id', '-s', type=int, default=1, help='Search process ID (default: 1)')
    parser.add_argument('--watch', '-w', action='store_true', help='Live updating dashboard mode')
    parser.add_argument('--interval', '-i', type=float, default=1.0, help='Update interval in seconds (default: 1.0)')
    parser.add_argument('--num-partitions', '-n', type=int, help='Number of partitions to sample')
    parser.add_argument('--summary', action='store_true', help='Show summary statistics')
    status_group = parser.add_mutually_exclusive_group()
    status_group.add_argument('--global', dest='show_global', action='store_true', default=True, help='Show global statistics (default)')
    status_group.add_argument('--partitions', action='store_true', help='Show partition breakdown')
    parser.set_defaults(func=status_command)
