"""Rust A* pathfinding between two crystal structures"""

import sys
import json
import argparse
import numpy as np
import rust_cnf
from pymatgen.core import Structure

from cnf import UnitCell
from cnf.cnf_constructor import CNFConstructor
from cnf.navigation.utils import get_endpoints_from_pmg_structs
from cnf.navigation.astar import astar_pathfind as astar_pathfind_py, squared_euclidean_heuristic, min_distance_filter


def pathfind_and_save(start_cif, end_cif, output_json, xi=0.2, delta=30, min_distance=0.0, max_iterations=100000):
    """Run pathfinding between two CIF structures and save result to JSON

    Args:
        start_cif: Path to starting structure CIF file
        end_cif: Path to ending structure CIF file
        output_json: Path to output JSON file
        xi: Lattice step size (default: 0.2)
        delta: Integer discretization factor (default: 30)
        min_distance: Minimum allowed pairwise distance for filtering (default: 0.0)
        max_iterations: Maximum pathfinding iterations (default: 100000)
    """

    start_struct = Structure.from_file(start_cif)
    end_struct = Structure.from_file(end_cif)

    print(f"\n=== Loading Structures ===")
    print(f"Starting: {start_cif}")
    print(f"  Composition: {start_struct.composition}, {len(start_struct)} atoms")
    print(f"  Lattice: a={start_struct.lattice.a:.3f}, b={start_struct.lattice.b:.3f}, c={start_struct.lattice.c:.3f}")
    print(f"Ending: {end_cif}")
    print(f"  Composition: {end_struct.composition}, {len(end_struct)} atoms")
    print(f"  Lattice: a={end_struct.lattice.a:.3f}, b={end_struct.lattice.b:.3f}, c={end_struct.lattice.c:.3f}")

    start_cells, goal_cells = get_endpoints_from_pmg_structs(start_struct, end_struct)

    print(f"\n=== Endpoints ===")
    print(f"Number of start cells: {len(start_cells)}")
    print(f"Number of goal cells: {len(goal_cells)}")

    # Convert unit cells to CNFs and deduplicate
    start_cnfs = list(set([cell.to_cnf(xi=xi, delta=delta) for cell in start_cells]))
    goal_cnfs = list(set([cell.to_cnf(xi=xi, delta=delta) for cell in goal_cells]))

    print(f"Unique start CNFs: {len(start_cnfs)}")
    for sc in start_cnfs:
        print(f"    {sc.coords}")
    print(f"Unique goal CNFs: {len(goal_cnfs)}")
    for ec in goal_cnfs:
        print(f"    {ec.coords}")

    # Convert CNFs to the format expected by Rust pathfinding
    start_points = []
    for cnf in start_cnfs:
        vonorms = list([int(i) for i in cnf.lattice_normal_form.vonorms.tuple])
        coords = list(cnf.motif_normal_form.coord_list)
        start_points.append((vonorms, coords))

    goal_points = []
    for cnf in goal_cnfs:
        vonorms = list([int(i) for i in cnf.lattice_normal_form.vonorms.tuple])
        coords = list(cnf.motif_normal_form.coord_list)
        goal_points.append((vonorms, coords))

    # Get n_atoms and elements from first start point
    first_cnf = start_cnfs[0]
    elements = first_cnf.motif_normal_form.elements
    n_atoms = len(elements)

    print(f"\n=== Running A* pathfinding ===")
    print(f"Elements: {elements}")
    print(f"N atoms: {n_atoms}")
    print(f"Xi: {xi}, Delta: {delta}")
    print(f"Start points: {len(start_points)}")
    print(f"Goal points: {len(goal_points)}")

    # Call Rust pathfinding with verbose output
    path = rust_cnf.astar_pathfind_rust(
        start_points,
        goal_points,
        elements,
        n_atoms,
        xi,
        delta,
        min_distance=min_distance,
        max_iterations=max_iterations,
        verbose=True
    )

    # path = astar_pathfind_py(
    #     start_cnfs,
    #     goal_cnfs,
    #     squared_euclidean_heuristic,
    #     min_distance_filter(min_distance),
    #     max_iterations=max_iterations,
    #     verbose=True
    # )

    if path is None:
        print("❌ No path found!")
        return False

    print(f"\n✅ Path found with {len(path)} steps!")

    # Path from Python A* is flat tuples (vonorms + coords concatenated)
    # Split them into (vonorms, coords) pairs for validation and output
    path_split = []
    for flat_tuple in path:
        vonorms = list(flat_tuple[:7])
        coords = list(flat_tuple[7:])
        path_split.append((vonorms, coords))

    # Show path summary
    print(f"\nPath summary:")
    print(f"  First step vonorms: {path_split[0][0]}")
    print(f"  First step coords:  {path_split[0][1]}")
    print(f"  Last step vonorms:  {path_split[-1][0]}")
    print(f"  Last step coords:   {path_split[-1][1]}")

    # Verify path starts at one of the start states
    path_start = path_split[0]
    start_matches = any(
        path_start[0] == s[0] and path_start[1] == s[1]
        for s in start_points
    )

    if start_matches:
        print("✅ Path starts at a valid start state")
    else:
        print(f"❌ Path doesn't start at any start state!")
        return False

    # Verify path ends at one of the goal states
    path_goal = path_split[-1]
    goal_matches = any(
        path_goal[0] == g[0] and path_goal[1] == g[1]
        for g in goal_points
    )

    if goal_matches:
        print("✅ Path ends at a valid goal state")
    else:
        print(f"❌ Path doesn't end at any goal state!")
        return False

    # Prepare JSON output
    output_data = {
        "metadata": {
            "xi": xi,
            "delta": delta,
            "elements": elements,
            "n_atoms": n_atoms,
            "min_distance": min_distance,
            "path_length": len(path),
            "start_cif": start_cif,
            "end_cif": end_cif
        },
        "path": [
            {
                "vonorms": vonorms,
                "coords": coords
            }
            for vonorms, coords in path_split
        ]
    }

    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Path saved to {output_json}")

    return True

def main():
    """Main entry point for the cnf-pathfind script"""
    parser = argparse.ArgumentParser(
        description="Run A* pathfinding between two crystal structures and save the path to JSON"
    )
    parser.add_argument("start_cif", help="Path to starting structure CIF file")
    parser.add_argument("end_cif", help="Path to ending structure CIF file")
    parser.add_argument("output_json", help="Path to output JSON file")
    parser.add_argument("--xi", type=float, default=0.2, help="Lattice step size (default: 0.2)")
    parser.add_argument("--delta", type=int, default=30, help="Integer discretization factor (default: 30)")
    parser.add_argument("--min-distance", type=float, default=0.0,
                        help="Minimum allowed pairwise distance for filtering in Angstroms (default: 0.0)")
    parser.add_argument("--max-iterations", type=int, default=100000,
                        help="Maximum pathfinding iterations (default: 100000)")

    args = parser.parse_args()

    success = pathfind_and_save(
        args.start_cif,
        args.end_cif,
        args.output_json,
        xi=args.xi,
        delta=args.delta,
        min_distance=args.min_distance,
        max_iterations=args.max_iterations
    )

    if success:
        print("\n✅ Pathfinding completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Pathfinding failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
