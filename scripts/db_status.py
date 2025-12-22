#!/usr/bin/env python3
"""
CLI tool to interrogate CNF database state.

Usage:
    python scripts/db_status.py <db_file>                    # Show stats once
    python scripts/db_status.py <db_file> --watch            # Live updating dashboard
    python scripts/db_status.py <db_file> --search <id>      # Show search process info
"""

import argparse
import sqlite3
import time
import sys
import os
from datetime import datetime


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def get_db_stats(db_file, search_id=None):
    """Get statistics about the database."""
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    stats = {}

    # Total points
    result = cur.execute("SELECT COUNT(*) FROM point").fetchone()
    stats['total_points'] = result[0]

    # Points with values (energy calculated)
    result = cur.execute("SELECT COUNT(*) FROM point WHERE value IS NOT NULL").fetchone()
    stats['points_with_energy'] = result[0]

    # Total edges
    result = cur.execute("SELECT COUNT(*) FROM edge").fetchone()
    stats['total_edges'] = result[0]

    # Explored points (neighbors found, globally)
    result = cur.execute("SELECT COUNT(*) FROM point WHERE explored = 1").fetchone()
    stats['explored_points'] = result[0]

    # Global energy range (all points with calculated energy)
    result = cur.execute("SELECT MIN(value), MAX(value) FROM point WHERE value IS NOT NULL").fetchone()
    stats['global_min_energy'] = result[0]
    stats['global_max_energy'] = result[1]

    if search_id is not None:
        # Frontier size for specific search
        result = cur.execute(
            "SELECT COUNT(*) FROM search_frontier_member WHERE search_id = ?",
            (search_id,)
        ).fetchone()
        stats['frontier_size'] = result[0]

        # Searched points for specific search
        result = cur.execute(
            "SELECT COUNT(*) FROM searched_point WHERE search_id = ?",
            (search_id,)
        ).fetchone()
        stats['searched_points'] = result[0]

        # Energy range of searched points
        result = cur.execute(
            """SELECT MIN(pt.value), MAX(pt.value)
               FROM searched_point AS sp
               JOIN point AS pt ON pt.id = sp.point_id
               WHERE sp.search_id = ? AND pt.value IS NOT NULL""",
            (search_id,)
        ).fetchone()
        stats['searched_min_energy'] = result[0]
        stats['searched_max_energy'] = result[1]

        # Energy range of frontier points
        result = cur.execute(
            """SELECT MIN(pt.value), MAX(pt.value)
               FROM search_frontier_member AS fm
               JOIN point AS pt ON pt.id = fm.point_id
               WHERE fm.search_id = ? AND pt.value IS NOT NULL""",
            (search_id,)
        ).fetchone()
        stats['frontier_min_energy'] = result[0]
        stats['frontier_max_energy'] = result[1]

    conn.close()
    return stats


def get_search_info(db_file, search_id):
    """Get information about a specific search process."""
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    # Search description
    result = cur.execute(
        "SELECT description FROM search_process WHERE id = ?",
        (search_id,)
    ).fetchone()

    if result is None:
        conn.close()
        return None

    description = result[0]

    # Start points
    start_points = cur.execute(
        """SELECT pt.id, pt.cnf, pt.value
           FROM search_start_point AS ssp
           JOIN point AS pt ON pt.id = ssp.start_point_id
           WHERE ssp.search_id = ?""",
        (search_id,)
    ).fetchall()

    # End points
    end_points = cur.execute(
        """SELECT pt.id, pt.cnf, pt.value
           FROM search_end_point AS sep
           JOIN point AS pt ON pt.id = sep.end_point_id
           WHERE sep.search_id = ?""",
        (search_id,)
    ).fetchall()

    # Check if any endpoints are in frontier
    endpoints_in_frontier = cur.execute(
        """SELECT COUNT(*)
           FROM search_frontier_member AS fm
           JOIN search_end_point AS sep ON sep.end_point_id = fm.point_id
           WHERE fm.search_id = ? AND sep.search_id = ?""",
        (search_id, search_id)
    ).fetchone()[0]

    conn.close()

    return {
        'description': description,
        'start_points': start_points,
        'end_points': end_points,
        'endpoints_in_frontier': endpoints_in_frontier
    }


def format_value(value, decimals=6):
    """Format a numeric value, handling None."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def display_stats(stats, search_id=None, rates=None):
    """Display statistics in a nice format."""
    if rates is None:
        rates = {}

    print("=" * 70)
    print(f"  CNF Database Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    print("GLOBAL STATISTICS:")
    total_points_rate = f" ({rates.get('total_points', 0):+.1f} pts/s)" if 'total_points' in rates else ""
    explored_rate = f" ({rates.get('explored_points', 0):+.1f} pts/s)" if 'explored_points' in rates else ""
    edges_rate = f" ({rates.get('total_edges', 0):+.1f} edges/s)" if 'total_edges' in rates else ""
    print(f"  Total Points:              {stats['total_points']:,}{total_points_rate}")
    print(f"  Points with Energy:        {stats['points_with_energy']:,}")
    print(f"  Neighbors Found (explored):{stats['explored_points']:,}{explored_rate}")
    print(f"  Total Edges:               {stats['total_edges']:,}{edges_rate}")
    print()

    print("GLOBAL ENERGY RANGE (all points ever discovered):")
    print(f"  Low Tide (min):            {format_value(stats['global_min_energy'])}")
    print(f"  High Tide (max):           {format_value(stats['global_max_energy'])}")
    energy_range = stats['global_max_energy'] - stats['global_min_energy'] if stats['global_max_energy'] and stats['global_min_energy'] else None
    print(f"  Total Range:               {format_value(energy_range)}")
    print()

    if search_id is not None:
        frontier_rate = f" ({rates.get('frontier_size', 0):+.1f} pts/s)" if 'frontier_size' in rates else ""
        searched_rate = f" ({rates.get('searched_points', 0):+.1f} pts/s)" if 'searched_points' in rates else ""
        print(f"SEARCH PROCESS #{search_id}:")
        print(f"  Frontier Size:             {stats['frontier_size']:,}{frontier_rate}")
        print(f"  Searched Points:           {stats['searched_points']:,}{searched_rate}")
        print()

        print(f"SEARCHED REGION (below water surface):")
        print(f"  Energy Range:              {format_value(stats['searched_min_energy'])} to {format_value(stats['searched_max_energy'])}")
        searched_range = stats['searched_max_energy'] - stats['searched_min_energy'] if stats['searched_max_energy'] and stats['searched_min_energy'] else None
        print(f"  Range Width:               {format_value(searched_range)}")
        print()

        print(f"FRONTIER (water surface - being explored):")
        print(f"  Energy Range:              {format_value(stats['frontier_min_energy'])} to {format_value(stats['frontier_max_energy'])}")
        frontier_range = stats['frontier_max_energy'] - stats['frontier_min_energy'] if stats['frontier_max_energy'] and stats['frontier_min_energy'] else None
        print(f"  Range Width:               {format_value(frontier_range)}")
        print()


def display_search_info(db_file, search_id):
    """Display information about a search process."""
    info = get_search_info(db_file, search_id)

    if info is None:
        print(f"Error: Search process #{search_id} not found")
        return

    print("=" * 60)
    print(f"  Search Process #{search_id}")
    print("=" * 60)
    print()

    print(f"Description: {info['description']}")
    print()

    print(f"Status: {'COMPLETE - Endpoint reached!' if info['endpoints_in_frontier'] > 0 else 'In progress...'}")
    print()

    print(f"START POINTS ({len(info['start_points'])}):")
    for point_id, cnf, value in info['start_points']:
        print(f"  Point #{point_id}:")
        print(f"    Energy: {format_value(value)}")
        print(f"    CNF:    {cnf}")
    print()

    print(f"END POINTS ({len(info['end_points'])}):")
    for point_id, cnf, value in info['end_points']:
        in_frontier = "✓ IN FRONTIER!" if info['endpoints_in_frontier'] > 0 else ""
        print(f"  Point #{point_id}: {in_frontier}")
        print(f"    Energy: {format_value(value)}")
        print(f"    CNF:    {cnf}")
    print()


def watch_mode(db_file, search_id=None, interval=1.0):
    """Continuously update and display stats."""
    try:
        iteration = 0
        prev_stats = None
        prev_time = None

        while True:
            # Query FIRST (so screen isn't blank while waiting)
            start_time = time.time()
            stats = get_db_stats(db_file, search_id)
            query_time = time.time() - start_time
            current_time = time.time()

            # Calculate rates if we have previous data
            rates = {}
            if prev_stats is not None and prev_time is not None:
                time_delta = current_time - prev_time
                if time_delta > 0:
                    rates['total_points'] = (stats['total_points'] - prev_stats['total_points']) / time_delta
                    rates['explored_points'] = (stats['explored_points'] - prev_stats['explored_points']) / time_delta
                    rates['total_edges'] = (stats['total_edges'] - prev_stats['total_edges']) / time_delta
                    if search_id is not None:
                        rates['frontier_size'] = (stats['frontier_size'] - prev_stats['frontier_size']) / time_delta
                        rates['searched_points'] = (stats['searched_points'] - prev_stats['searched_points']) / time_delta

            # THEN clear and display
            clear_screen()
            display_stats(stats, search_id, rates=rates)

            # Show refresh info with query timing
            iteration += 1
            print(f"Refreshing every {interval}s... (query took {query_time:.2f}s, iteration #{iteration})")
            print("Press Ctrl+C to stop")

            # If queries are slow, warn user
            if query_time > interval * 0.8:
                print(f"\n⚠️  Warning: Queries taking {query_time:.2f}s (longer than refresh interval)")
                print(f"   Consider using --interval {max(2.0, query_time * 1.5):.1f} for smoother updates")

            # Save current stats for next iteration
            prev_stats = stats.copy()
            prev_time = current_time

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nStopped watching.")
        sys.exit(0)



def main():
    parser = argparse.ArgumentParser(
        description="Interrogate CNF database state",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s test_search_db                      # Show stats once
  %(prog)s test_search_db --watch              # Live updating dashboard
  %(prog)s test_search_db --watch --search 1   # Watch search #1
  %(prog)s test_search_db --search 1           # Show search #1 info
        """
    )

    parser.add_argument('db_file', help='Path to the CNF database file')
    parser.add_argument('--watch', '-w', action='store_true',
                       help='Continuously update stats (like watch command)')
    parser.add_argument('--search', '-s', type=int, metavar='ID',
                       help='Show/monitor specific search process')
    parser.add_argument('--interval', '-i', type=float, default=1.0,
                       help='Update interval in seconds for watch mode (default: 1.0)')

    args = parser.parse_args()

    # Check if database exists
    if not os.path.exists(args.db_file):
        print(f"Error: Database file '{args.db_file}' not found")
        sys.exit(1)

    if args.watch:
        # Watch mode
        watch_mode(args.db_file, args.search, args.interval)
    elif args.search is not None:
        # Display search info
        display_search_info(args.db_file, args.search)
        print()
        # Also show stats for this search
        stats = get_db_stats(args.db_file, args.search)
        display_stats(stats, args.search)
    else:
        # One-time stats display
        stats = get_db_stats(args.db_file)
        display_stats(stats)


if __name__ == '__main__':
    main()
