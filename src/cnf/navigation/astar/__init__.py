"""A* pathfinding for CNF navigation with customizable heuristics and filters"""

import os
import json
import rust_cnf

from typing import Callable

from cnf import CrystalNormalForm
from ..search_filters import FilterSet, MinDistanceFilter
from pymatgen.core import Structure
from ..endpoints import get_endpoint_unit_cells
from .core import astar_pathfind
from .bidirectional import bidirectional_astar_pathfind
from .heuristics import manhattan_distance

USE_RUST = os.getenv('USE_RUST') == '1'

# Type aliases for clarity
FilterFunc = Callable[[CrystalNormalForm], bool]


def pathfind_and_save(start_cif,
                      end_cif,
                      output_json,
                      xi=0.2,
                      delta=30,
                      min_distance=0.0,
                      max_iterations=100000,
                      use_python=False,
                      bidirectional=False,
                      beam_width=None):
    """Run pathfinding between two CIF structures and save result to JSON

    Args:
        start_cif: Path to starting structure CIF file
        end_cif: Path to ending structure CIF file
        output_json: Path to output JSON file
        xi: Lattice step size (default: 0.2)
        delta: Integer discretization factor (default: 30)
        min_distance: Minimum allowed pairwise distance for filtering (default: 0.0)
        max_iterations: Maximum pathfinding iterations (default: 100000)
        use_python: Use Python A* implementation instead of Rust (default: False)
        bidirectional: Use bidirectional A* (only with use_python=True) (default: False)
        beam_width: Beam width for beam search (only with use_python=True, bidirectional=False) (default: None)
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

    start_cells, goal_cells = get_endpoint_unit_cells(start_struct, end_struct)

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
    print(f"Implementation: {'Python' if use_python else 'Rust'}")
    print(f"Elements: {elements}")
    print(f"N atoms: {n_atoms}")
    print(f"Xi: {xi}, Delta: {delta}")
    print(f"Start points: {len(start_points)}")
    print(f"Goal points: {len(goal_points)}")

    search_state = None  # Will be set if using Python non-bidirectional A*

    if use_python:
        # Use Python A* implementation
        heuristic = manhattan_distance
        filter_set = FilterSet([MinDistanceFilter(min_distance)], use_structs = not USE_RUST)
        # filter_set = FilterSet([EnergyFilter.from_cnfs(start_cnfs + goal_cnfs)])
        if bidirectional:
            path = bidirectional_astar_pathfind(
                start_cnfs,
                goal_cnfs,
                heuristic,
                filter_set,
                max_iterations=max_iterations,
                beam_width=beam_width,
                verbose=True
            )
        else:
            search_state = astar_pathfind(
                start_cnfs,
                goal_cnfs,
                heuristic,
                filter_set,
                max_iterations=max_iterations,
                beam_width=beam_width,
                verbose=True
            )
            path = search_state.path

            # finished = False
            # round = 1
            # # while not finished:
            # #     print(f"Starting round {round}!")
            # #     search_state = astar_pathfind(
            # #         start_cnfs,
            # #         goal_cnfs,
            # #         heuristic,
            # #         filter_set,
            # #         max_iterations=max_iterations,
            # #         beam_width=beam_width,
            # #         verbose=True
            # #     )
            # #     if search_state.found_goal:
            # #         finished = True
            # #         path = search_state.path
            # #     round += 1
            # #     start_cnfs = [heapq.heappop(search_state.open_set).point for _ in range(1)]
            # #     start_cnfs = [CrystalNormalForm.from_tuple(sc, elements=elements, xi=xi, delta=delta) for sc in start_cnfs]
            # forward_results = []
            # backward_results = []
            # forward_goals = goal_cnfs
            # forward_starts = start_cnfs

            # while not finished:
            #     forward_result = astar_pathfind(
            #         forward_starts,
            #         forward_goals,
            #         heuristic,
            #         filter_set,
            #         max_iterations=max_iterations,
            #         beam_width=beam_width,
            #         greedy=True,
            #         verbose=True
            #     )
            #     forward_results.append(forward_result)

            #     if forward_result.found_goal:
            #         finished = True
            #         break

            #     backward_goals = [heapq.heappop(forward_result.open_set).point for _ in range(1)]
            #     backward_goals = [CrystalNormalForm.from_tuple(bg, elements=elements, xi=xi, delta=delta) for bg in backward_goals]
            #     backward_starts = forward_goals

            #     print("Beginning backward search!")
            #     backward_result = astar_pathfind(
            #         backward_starts,
            #         backward_goals,
            #         heuristic,
            #         filter_set,
            #         max_iterations=100,
            #         beam_width=beam_width,
            #         greedy=True,
            #         verbose=True
            #     )
            #     backward_results.append(backward_result)
            #     if backward_result.found_goal:
            #         finished = True
            #         break

            #     round += 1
            #     forward_goals = [heapq.heappop(backward_result.open_set).point for _ in range(1)]
            #     forward_goals = [CrystalNormalForm.from_tuple(fg, elements=elements, xi=xi, delta=delta) for fg in forward_goals]                    
            #     forward_starts = backward_goals
                
        
    else:
        # Use Rust A* implementation (default)
        if bidirectional:
            path = rust_cnf.bidirectional_astar_pathfind_rust(
                start_points,
                goal_points,
                elements,
                n_atoms,
                xi,
                delta,
                min_distance=min_distance,
                max_iterations=max_iterations,
                beam_width=beam_width if beam_width is not None else 0,
                verbose=True
            )
        else:
            path = rust_cnf.astar_pathfind_rust(
                start_points,
                goal_points,
                elements,
                n_atoms,
                xi,
                delta,
                min_distance=min_distance,
                max_iterations=max_iterations,
                beam_width=beam_width if beam_width is not None else 0,
                verbose=True
            )

    # Prepare output metadata
    output_data = {
        "metadata": {
            "xi": xi,
            "delta": delta,
            "elements": elements,
            "n_atoms": n_atoms,
            "min_distance": min_distance,
            "start_cif": start_cif,
            "end_cif": end_cif,
            "found_goal": path is not None
        }
    }

    if search_state is not None:
        output_data["metadata"]["iterations"] = search_state.iterations
        output_data["metadata"]["max_iterations_reached"] = search_state.max_iterations_reached
        # output_data["search_state"] = search_state.to_dict()
        print(f"\nSearch state: {search_state.frontier_stats()}")

    if path is None:
        print("❌ No path found!")
        output_data["metadata"]["path_length"] = 0
        output_data["path"] = []

        # Save state even when no path found
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Search state saved to {output_json}")
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

    output_data["metadata"]["path_length"] = len(path)
    output_data["path"] = [
        {
            "vonorms": vonorms,
            "coords": coords
        }
        for vonorms, coords in path_split
    ]

    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Path saved to {output_json}")

    return True