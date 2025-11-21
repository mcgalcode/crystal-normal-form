#!/usr/bin/env python3
"""
Check if search completion criteria are met in a partitioned database.

A search is considered complete if any endpoint is either:
- In the search frontier (we've reached it during exploration)
- Connected to another point (we've found it as a neighbor)

Usage:
    python scripts/check_search_completion.py <partitions_dir> [--search-id ID]
    python scripts/check_search_completion.py <partitions_dir> --watch
"""

import sys
import argparse
import time
import os
from datetime import datetime
from cnf.db.partitioned_db import PartitionedDB


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')

def gather_completion_data(partitions_dir: str, search_id: int = 1):
    """Gather completion data without printing. Returns (completion_found, endpoint_data_list, num_endpoints)."""

    db = PartitionedDB(partitions_dir)

    # Get endpoints from one of the partitions (they should all have the same search metadata)
    search_store = db.get_search_store_by_idx(0)
    endpoints = search_store.get_search_endpoints(search_id)

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
        # try:
        point = map_store.get_point_by_cnf(endpoint_cnf)
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

        # except Exception as e:
        #     print(e.__repr__())
        #     endpoint_data['error'] = f"{type(e).__name__}"

        endpoint_data_list.append(endpoint_data)

    return completion_found, endpoint_data_list, len(endpoints)


def display_completion_status(completion_found, endpoint_data_list, num_endpoints, search_id):
    """Display completion status from gathered data."""

    if endpoint_data_list is None:
        print(f"Error: No endpoints found for search_id {search_id}")
        return

    print(f"Checking {num_endpoints} endpoint(s) for search_id {search_id}...")
    print("=" * 70)
    print()

    for endpoint_data in endpoint_data_list:
        print(f"Endpoint {endpoint_data['index']}:")
        print(f"  CNF coords: {endpoint_data['coords']}")
        print(f"  Partition:  {endpoint_data['partition']}")

        if not endpoint_data['found']:
            print(f"  Status: Not found in database ({endpoint_data.get('error', 'Unknown error')})")
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


def check_search_completion(partitions_dir: str, search_id: int = 1):
    """Check if any search endpoint has been reached."""
    completion_found, endpoint_data_list, num_endpoints = gather_completion_data(partitions_dir, search_id)

    if endpoint_data_list is None:
        print(f"Error: No endpoints found for search_id {search_id}")
        return False

    display_completion_status(completion_found, endpoint_data_list, num_endpoints, search_id)
    return completion_found


def watch_mode(partitions_dir: str, search_id: int = 1, interval: float = 1.0):
    """Continuously monitor search completion status."""
    try:
        iteration = 0

        while True:
            # Query FIRST (before clearing screen to avoid flicker)
            completion_found, endpoint_data_list, num_endpoints = gather_completion_data(partitions_dir, search_id)

            # THEN clear screen and display
            clear_screen()
            print(f"Search Completion Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Partitions Directory: {partitions_dir}")
            print(f"Search ID: {search_id}")
            print(f"Update #{iteration} (refreshing every {interval}s, Ctrl+C to exit)")
            print()

            # Display completion status
            display_completion_status(completion_found, endpoint_data_list, num_endpoints, search_id)

            print()
            print(f"Last update: {datetime.now().strftime('%H:%M:%S')}")

            iteration += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Check if search completion criteria are met',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check search in partitions directory
  python scripts/check_search_completion.py ./my_partitions

  # Check specific search ID
  python scripts/check_search_completion.py ./my_partitions --search-id 2

  # Watch mode - continuously monitor until completion
  python scripts/check_search_completion.py ./my_partitions --watch
        """
    )

    parser.add_argument('partitions_dir',
                       help='Directory containing graph_partition_*.db files')
    parser.add_argument('--search-id', type=int, default=1,
                       help='Search process ID to check (default: 1)')
    parser.add_argument('--watch', '-w', action='store_true',
                       help='Continuously monitor until search completes')
    parser.add_argument('--interval', '-i', type=float, default=1.0,
                       help='Update interval in seconds for watch mode (default: 1.0)')

    args = parser.parse_args()

    try:
        if args.watch:
            watch_mode(args.partitions_dir, args.search_id, args.interval)
        else:
            completed = check_search_completion(args.partitions_dir, args.search_id)
            sys.exit(0 if completed else 1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)

if __name__ == '__main__':
    main()
