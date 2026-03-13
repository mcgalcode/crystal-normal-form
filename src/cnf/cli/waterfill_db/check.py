"""Check command for waterfill database completion."""

import sys
import os
import time
from datetime import datetime

from cnf.db.partitioned_db import PartitionedDB


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def gather_completion_data(partition_dir: str, search_id: int = 1):
    """Gather completion data for endpoints."""
    db = PartitionedDB(partition_dir, search_id)

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

        partition_idx = db.get_partition_idx(endpoint_cnf)
        map_store = db.get_map_store_by_idx(partition_idx)
        search_store = db.get_search_store_by_idx(partition_idx)

        endpoint_data = {
            'index': i,
            'coords': endpoint_cnf.coords,
            'partition': partition_idx,
            'found': False
        }

        point = map_store.get_point_by_cnf(endpoint_cnf)
        if point:
            endpoint_data['point_id'] = point.id
            endpoint_data['found'] = True

            in_frontier_results = search_store.get_endpoint_ids_in_frontier(search_id)
            in_frontier = point.id in in_frontier_results
            endpoint_data['in_frontier'] = in_frontier

            neighbors = map_store.get_neighbor_cnfs(point.id)
            num_neighbors = len(neighbors) if neighbors else 0
            endpoint_data['num_neighbors'] = num_neighbors

            searched_points = search_store.get_searched_points_in_search(search_id)
            is_searched = any(sp.id == point.id for sp in searched_points)
            endpoint_data['is_searched'] = is_searched

            endpoint_data['explored'] = point.explored
            endpoint_data['has_value'] = point.value is not None
            endpoint_data['value'] = point.value

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

            if endpoint_data['has_value']:
                print(f"  Value:       {endpoint_data['value']:.8f}")
            else:
                print(f"  Value:       Not computed")

            print(f"  In frontier: {endpoint_data['in_frontier']}")
            print(f"  Searched:    {endpoint_data['is_searched']}")
            print(f"  Explored:    {endpoint_data['explored']}")
            print(f"  Connections: {endpoint_data['num_neighbors']}")

            if endpoint_data['reached']:
                print(f"  Status:      REACHED!")
            else:
                print(f"  Status:      Not yet reached")

        print()

    print("=" * 70)
    if completion_found:
        print("SUCCESS: At least one endpoint has been reached!")
        print("  The search has found a path.")
    else:
        print("INCOMPLETE: No endpoints reached yet")
        print("  Continue searching...")
    print("=" * 70)


def watch_mode(partition_dir: str, search_id: int = 1, interval: float = 1.0):
    """Continuously monitor search completion status."""
    try:
        iteration = 0

        while True:
            completion_found, endpoint_data_list, num_endpoints = gather_completion_data(partition_dir, search_id)

            clear_screen()
            print(f"Search Completion Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Partitions Directory: {partition_dir}")
            print(f"Search ID: {search_id}")
            print(f"Update #{iteration} (refreshing every {interval}s, Ctrl+C to exit)")
            print()

            display_completion_status(completion_found, endpoint_data_list, num_endpoints, search_id)

            print()
            print(f"Last update: {datetime.now().strftime('%H:%M:%S')}")

            iteration += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        sys.exit(0)


def check_command(args):
    """Check if search has reached endpoints."""
    try:
        if args.watch:
            watch_mode(args.partition_dir, args.search_id, args.interval)
        else:
            completion_found, endpoint_data_list, num_endpoints = gather_completion_data(
                args.partition_dir, args.search_id
            )
            display_completion_status(completion_found, endpoint_data_list, num_endpoints, args.search_id)
            sys.exit(0 if completion_found else 1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


def register_parser(subparsers):
    """Register the check subcommand."""
    parser = subparsers.add_parser('check', help='Check if search has reached endpoints')
    parser.add_argument('partition_dir', help='Directory containing partition files')
    parser.add_argument('--search-id', type=int, default=1, help='Search process ID (default: 1)')
    parser.add_argument('--watch', '-w', action='store_true', help='Continuously monitor until completion')
    parser.add_argument('--interval', '-i', type=float, default=1.0, help='Update interval in seconds (default: 1.0)')
    parser.set_defaults(func=check_command)
