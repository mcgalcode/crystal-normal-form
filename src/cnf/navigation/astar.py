"""A* pathfinding for CNF navigation with customizable heuristics and filters"""

import os
import heapq
import time
import json
import rust_cnf
import numpy as np

from typing import List, Optional, Callable, Set
from dataclasses import dataclass, field

from functools import cache

from cnf import CrystalNormalForm
from ..utils.pdd import pdd_for_cnfs, pdd_amd_for_cnfs
from .utils import no_atoms_closer_than
from .neighbor_finder import NeighborFinder
from .search_filters import EnergyFilter, FilterSet, MinDistanceFilter

from pymatgen.core import Structure

from .endpoints import get_endpoint_unit_cells

USE_RUST = os.getenv('USE_RUST') == '1'

@dataclass(order=True)
class AStarNode:
    """Node in the A* search priority queue"""
    f_score: float
    g_score: float = field(compare=False)
    h_score: float = field(compare=True)
    point: tuple = field(compare=False)
    counter: int = field(compare=False)  # For tie-breaking


# Type aliases for clarity
HeuristicFunc = Callable[[CrystalNormalForm, List[CrystalNormalForm]], float]
FilterFunc = Callable[[CrystalNormalForm], bool]

def pdd_heuristic(cnf: tuple, goals: list[CrystalNormalForm]) -> float:
    xi = goals[0].xi
    delta = goals[0].delta
    els = goals[0].elements

    pt = CrystalNormalForm.from_tuple(cnf, els, xi, delta)
    dists = [pdd_for_cnfs(pt, g, k=20) for g in goals]
    return (min(dists) * 100) ** 3

def squared_euclidean_heuristic(cnf: tuple, goals: List[CrystalNormalForm]) -> float:
    """
    Compute squared Euclidean distance heuristic to nearest goal.

    This is inadmissible (overestimates) but works well in practice for CNF navigation.
    Uses sum of squared differences WITHOUT sqrt.

    Args:
        cnf: Current CNF state as flat tuple (vonorms + coords concatenated)
        goals: List of goal CNF states

    Returns:
        Minimum squared Euclidean distance to any goal
    """
    vonorms = np.array(cnf[:7])
    coords = np.array(cnf[7:])

    min_dist_sq = float('inf')

    for goal in goals:
        goal_vonorms = np.array(goal.lattice_normal_form.vonorms.tuple)
        goal_coords = np.array(goal.motif_normal_form.coord_list)

        # Squared distance in vonorm space
        vonorm_dist_sq = np.sum((vonorms - goal_vonorms) ** 2)

        # Squared distance in coord space
        coord_dist_sq = np.sum((coords - goal_coords) ** 2)

        # Combined squared distance (no sqrt)
        dist_sq = vonorm_dist_sq + coord_dist_sq
        min_dist_sq = min(min_dist_sq, dist_sq)

    return min_dist_sq

def manhattan_distance(cnf: tuple, goals: List[CrystalNormalForm]) -> float:
    """
    Compute squared Euclidean distance heuristic to nearest goal.

    This is inadmissible (overestimates) but works well in practice for CNF navigation.
    Uses sum of squared differences WITHOUT sqrt.

    Args:
        cnf: Current CNF state as flat tuple (vonorms + coords concatenated)
        goals: List of goal CNF states

    Returns:
        Minimum squared Euclidean distance to any goal
    """

    manhattan_dist = float('inf')
    current_coords = np.array(cnf)

    for goal in goals:
        goal_coords = np.array(goal.coords)
        curr_dist = np.sum(np.abs(current_coords - goal_coords))
        manhattan_dist = min(manhattan_dist, curr_dist)

    return manhattan_dist * 2


def astar_pathfind(
    start_cnfs: List[CrystalNormalForm],
    goal_cnfs: List[CrystalNormalForm],
    heuristic: Optional[HeuristicFunc] = None,
    filter_set: Optional[FilterSet] = None,
    max_iterations: int = 100000,
    beam_width: Optional[int] = None,
    verbose: bool = False
) -> Optional[List[CrystalNormalForm]]:
    """
    A* pathfinding between CNF states with customizable heuristic and filtering.

    Args:
        start_cnfs: List of starting CNF states
        goal_cnfs: List of goal CNF states
        heuristic: Heuristic function h(cnf, goals) -> float
                   If None, uses squared Euclidean distance
        filter_set: FilterSet to exclude invalid neighbors
                     If None, no filtering is applied
        max_iterations: Maximum number of iterations (0 for unlimited)
        beam_width: Maximum size of open set (beam search). If None, no limit (standard A*)
        verbose: Print progress every 100 iterations

    Returns:
        List of CNF states forming the path from start to goal, or None if no path found
    """
    if not start_cnfs:
        raise ValueError("start_cnfs cannot be empty")
    if not goal_cnfs:
        raise ValueError("goal_cnfs cannot be empty")
    
    goal_cnfs = tuple(goal_cnfs)

    # Use default heuristic if not provided
    if heuristic is None:
        heuristic = pdd_heuristic

    # Check if CNF is a goal
    def is_goal(cnf_point: tuple) -> bool:
        return any(cnf_point == goal.coords for goal in goal_cnfs)
    
    xi = start_cnfs[0].xi
    delta = start_cnfs[0].delta
    elements = start_cnfs[0].elements

    # Initialize search structures
    open_set = []  # Priority queue (heap)
    closed_set: Set[tuple] = set()  # Visited states
    came_from = {}  # Parent pointers for path reconstruction
    g_score = {}  # Cost from start to each node
    counter = 0  # For tie-breaking in priority queue

    neighbor_finder = NeighborFinder.from_cnf(start_cnfs[0])

    # Add all start states to open set
    for start_cnf in start_cnfs:
        start_key = start_cnf.coords
        g_score[start_key] = 0.0
        h = heuristic(start_key, goal_cnfs)
        f = 0.0 + h

        heapq.heappush(open_set, AStarNode(
            f_score=f,
            g_score=0.0,
            h_score=h,
            point=start_key,
            counter=counter
        ))
        counter += 1

    iterations = 0

    if verbose:
        print(f"Starting A* search with {len(start_cnfs)} start states and {len(goal_cnfs)} goal states")

    ts = time.perf_counter_ns()
    while open_set:
        iterations += 1

        if max_iterations > 0 and iterations > max_iterations:
            if verbose:
                print(f"Reached max iterations ({max_iterations})")
            return None

        if verbose and iterations % 5 == 0:
            te = time.perf_counter_ns()
            elapsed = round((te - ts) / 1e9, 3)

            print(f"Step {iterations}:  open={len(open_set)}, closed={len(closed_set)}, f={open_set[0].f_score:.2f}, g={open_set[0].g_score:.2f}, h={open_set[0].h_score:.6f}, Elapsed: {elapsed:.2f}s")


        # Pop node with lowest f_score
        current_node = heapq.heappop(open_set)

        current_point = current_node.point

        # Check if we reached a goal
        goal_reached = is_goal(current_point)

        if goal_reached:
            if verbose:
                print(f"\n✅ Found goal after {iterations} iterations!")
            return _reconstruct_path(current_point, came_from)

        # Mark as visited
        closed_set.add(current_point)

        # Get neighbors
        neighbors = neighbor_finder.find_neighbor_tuples(current_point)

        if iterations == 1 and verbose:
            print(f"First iteration: found {len(neighbors)} raw neighbors")

        # Apply filter if provided
        if filter_set is not None:
            neighbors = [CrystalNormalForm.from_tuple(n, elements, xi, delta) for n in neighbors]
            neighbors, _ = filter_set.filter_cnfs(neighbors)
            neighbors = [nb.coords for nb in neighbors]

        # Sort neighbors for deterministic behavior (avoids hash randomization issues)
        neighbors = sorted(neighbors)

        # Explore neighbors
        for neighbor_point in neighbors:
            in_closed = neighbor_point in closed_set
            if in_closed:
                continue

            # Edge cost: uniform cost 1
            edge_cost = 1.0
            tentative_g = current_node.g_score + edge_cost

            # Check if this is a better path
            current_g = g_score.get(neighbor_point, float('inf'))

            if tentative_g < current_g:
                # Update path
                came_from[neighbor_point] = current_point
                g_score[neighbor_point] = tentative_g

                # Calculate f_score
                h = heuristic(neighbor_point, goal_cnfs)

                # f = tentative_g + h
                f = h

                # Add to open set
                heapq.heappush(open_set, AStarNode(
                    f_score=f,
                    g_score=tentative_g,
                    h_score=h,
                    point=neighbor_point,
                    counter=counter
                ))

                counter += 1

        # Beam search: prune open set if it exceeds beam_width
        if beam_width is not None and len(open_set) > beam_width:
            # Keep only the beam_width nodes with lowest f-score
            open_set = heapq.nsmallest(beam_width, open_set)
            heapq.heapify(open_set)

    if verbose:
        print(f"\n❌ No path found after {iterations} iterations")

    return None


def _reconstruct_path(
    goal_point: tuple,
    came_from: dict
) -> List[tuple]:
    """
    Reconstruct the path from start to goal.

    Args:
        goal_point: The goal CNF tuple that was reached
        came_from: Dictionary mapping tuple keys to parent tuple keys

    Returns:
        List of CNF tuples forming the path from start to goal
    """

    path = [goal_point]
    current_point = goal_point

    while current_point in came_from:
        parent_point = came_from[current_point]
        path.append(parent_point)
        current_point = parent_point

    # Reverse to get path from start to goal
    path.reverse()

    return path


def bidirectional_astar_pathfind(
    start_cnfs: List[CrystalNormalForm],
    goal_cnfs: List[CrystalNormalForm],
    heuristic: Optional[HeuristicFunc] = None,
    filter_set: Optional[FilterSet] = None,
    max_iterations: int = 100000,
    beam_width: Optional[int] = None,
    verbose: bool = False
) -> Optional[List[CrystalNormalForm]]:
    """
    Bidirectional A* pathfinding - searches from both start and goal simultaneously.

    Args:
        start_cnfs: List of starting CNF states
        goal_cnfs: List of goal CNF states
        heuristic: Heuristic function h(cnf, goals) -> float
        filter_func: Filter function f(cnf) -> bool to exclude invalid neighbors
        max_iterations: Maximum number of iterations (0 for unlimited)
        beam_width: Maximum size of each open set (beam search). If None, no limit (standard A*)
        verbose: Print progress every 100 iterations

    Returns:
        List of CNF states forming the path from start to goal, or None if no path found
    """
    if not start_cnfs:
        raise ValueError("start_cnfs cannot be empty")
    if not goal_cnfs:
        raise ValueError("goal_cnfs cannot be empty")

    # Use default heuristic if not provided
    if heuristic is None:
        heuristic = squared_euclidean_heuristic

    xi = start_cnfs[0].xi
    delta = start_cnfs[0].delta
    elements = start_cnfs[0].elements

    # Forward search structures
    forward_open = []
    forward_closed: Set[tuple] = set()
    forward_came_from = {}
    forward_g_score = {}
    forward_counter = 0

    # Backward search structures
    backward_open = []
    backward_closed: Set[tuple] = set()
    backward_came_from = {}
    backward_g_score = {}
    backward_counter = 0

    neighbor_finder = NeighborFinder.from_cnf(start_cnfs[0])

    # Initialize forward search with all start states
    for start_cnf in start_cnfs:
        start_key = start_cnf.coords
        forward_g_score[start_key] = 0.0
        h = heuristic(start_key, goal_cnfs)
        f = 0.0 + h
        heapq.heappush(forward_open, AStarNode(
            f_score=f,
            g_score=0.0,
            h_score=h,
            point=start_key,
            counter=forward_counter
        ))
        forward_counter += 1

    # Initialize backward search with all goal states
    for goal_cnf in goal_cnfs:
        goal_key = goal_cnf.coords
        backward_g_score[goal_key] = 0.0
        h = heuristic(goal_key, start_cnfs)  # Heuristic to starts
        f = 0.0 + h
        heapq.heappush(backward_open, AStarNode(
            f_score=f,
            g_score=0.0,
            h_score=h,
            point=goal_key,
            counter=backward_counter
        ))
        backward_counter += 1

    iterations = 0
    best_path_length = float('inf')
    meeting_point = None

    if verbose:
        print(f"Starting bidirectional A* search:")
        print(f"  {len(start_cnfs)} start states, {len(goal_cnfs)} goal states")

    import time as time_module
    ts = time_module.perf_counter_ns()

    # Track g and h for logging
    last_forward_g = 0.0
    last_forward_h = 0.0
    last_backward_g = 0.0
    last_backward_h = 0.0

    while forward_open and backward_open:
        iterations += 1

        if max_iterations > 0 and iterations > max_iterations:
            if verbose:
                print(f"Reached max iterations ({max_iterations})")
            break

        if verbose and iterations % 5 == 0:
            te = time_module.perf_counter_ns()
            elapsed = round((te - ts) / 1e9, 3)
            print(f"Step {iterations}: "
                  f"fwd_open={len(forward_open)}, fwd_closed={len(forward_closed)}, fwd_g={last_forward_g:.1f}, fwd_h={last_forward_h:.4f}, "
                  f"bwd_open={len(backward_open)}, bwd_closed={len(backward_closed)}, bwd_g={last_backward_g:.1f}, bwd_h={last_backward_h:.4f}, "
                  f"elapsed={elapsed:.2f}s")

        # Expand from forward search
        if forward_open:
            current_node = heapq.heappop(forward_open)
            last_forward_g = current_node.g_score
            last_forward_h = current_node.h_score
            current_point = current_node.point

            # Check if we've met the backward search
            if current_point in backward_closed:
                path_length = forward_g_score[current_point] + backward_g_score[current_point]
                if path_length < best_path_length:
                    best_path_length = path_length
                    meeting_point = current_point
                    if verbose:
                        print(f"\n✅ Found meeting point at iteration {iterations}! Path length: {path_length}")
                    # Reconstruct and return path immediately
                    return _reconstruct_bidirectional_path(
                        meeting_point, forward_came_from, backward_came_from
                    )

            # Skip if already visited in forward search
            if current_point in forward_closed:
                continue

            forward_closed.add(current_point)

            # Expand neighbors
            neighbors = neighbor_finder.find_neighbor_tuples(current_point)

            # Apply filter if provided
            if filter_set is not None:
                neighbors = [CrystalNormalForm.from_tuple(n, elements, xi, delta) for n in neighbors]
                neighbors, _ = filter_set.filter_cnfs(neighbors)
                neighbors = [nb.coords for nb in neighbors]

            neighbors = sorted(neighbors)  # Deterministic ordering

            for neighbor_point in neighbors:
                if neighbor_point in forward_closed:
                    continue

                edge_cost = 1.0
                tentative_g = current_node.g_score + edge_cost
                current_g = forward_g_score.get(neighbor_point, float('inf'))

                if tentative_g < current_g:
                    forward_came_from[neighbor_point] = current_point
                    forward_g_score[neighbor_point] = tentative_g
                    h = heuristic(neighbor_point, goal_cnfs)
                    f = tentative_g + h

                    heapq.heappush(forward_open, AStarNode(
                        f_score=f,
                        g_score=tentative_g,
                        h_score=h,
                        point=neighbor_point,
                        counter=forward_counter
                    ))
                    forward_counter += 1

            # Beam search: prune forward open set if it exceeds beam_width
            if beam_width is not None and len(forward_open) > beam_width:
                forward_open = heapq.nsmallest(beam_width, forward_open)
                heapq.heapify(forward_open)

        # Expand from backward search
        if backward_open:
            current_node = heapq.heappop(backward_open)
            last_backward_g = current_node.g_score
            last_backward_h = current_node.h_score
            current_point = current_node.point

            # Check if we've met the forward search
            if current_point in forward_closed:
                path_length = forward_g_score[current_point] + backward_g_score[current_point]
                if path_length < best_path_length:
                    best_path_length = path_length
                    meeting_point = current_point
                    if verbose:
                        print(f"\n✅ Found meeting point at iteration {iterations}! Path length: {path_length}")
                    # Reconstruct and return path immediately
                    return _reconstruct_bidirectional_path(
                        meeting_point, forward_came_from, backward_came_from
                    )

            # Skip if already visited in backward search
            if current_point in backward_closed:
                continue

            backward_closed.add(current_point)

            # Expand neighbors
            neighbors = neighbor_finder.find_neighbor_tuples(current_point)

            # Apply filter if provided
            if filter_set is not None:
                neighbors = [CrystalNormalForm.from_tuple(n, elements, xi, delta) for n in neighbors]
                neighbors, _ = filter_set.filter_cnfs(neighbors)
                neighbors = [nb.coords for nb in neighbors]


            neighbors = sorted(neighbors)  # Deterministic ordering

            for neighbor_point in neighbors:
                if neighbor_point in backward_closed:
                    continue

                edge_cost = 1.0
                tentative_g = current_node.g_score + edge_cost
                current_g = backward_g_score.get(neighbor_point, float('inf'))

                if tentative_g < current_g:
                    backward_came_from[neighbor_point] = current_point
                    backward_g_score[neighbor_point] = tentative_g
                    h = heuristic(neighbor_point, start_cnfs)  # Heuristic to starts
                    f = tentative_g + h

                    heapq.heappush(backward_open, AStarNode(
                        f_score=f,
                        g_score=tentative_g,
                        h_score=h,
                        point=neighbor_point,
                        counter=backward_counter
                    ))
                    backward_counter += 1

            # Beam search: prune backward open set if it exceeds beam_width
            if beam_width is not None and len(backward_open) > beam_width:
                backward_open = heapq.nsmallest(beam_width, backward_open)
                heapq.heapify(backward_open)

    if meeting_point:
        if verbose:
            print(f"\n✅ Found path with length {best_path_length}")
        return _reconstruct_bidirectional_path(meeting_point, forward_came_from, backward_came_from)

    if verbose:
        print(f"\n❌ No path found after {iterations} iterations")

    return None


def _reconstruct_bidirectional_path(
    meeting_point: tuple,
    forward_came_from: dict,
    backward_came_from: dict
) -> List[tuple]:
    """
    Reconstruct path from bidirectional search.

    Args:
        meeting_point: The point where forward and backward searches met
        forward_came_from: Dictionary of parent pointers from forward search
        backward_came_from: Dictionary of parent pointers from backward search

    Returns:
        Complete path from start to goal
    """
    # Build forward path (start to meeting point)
    forward_path = []
    current = meeting_point
    while current in forward_came_from:
        forward_path.append(current)
        current = forward_came_from[current]
    forward_path.append(current)  # Add the start node
    forward_path.reverse()

    # Build backward path (meeting point to goal)
    backward_path = []
    current = meeting_point
    while current in backward_came_from:
        current = backward_came_from[current]
        backward_path.append(current)

    # Combine: forward path + backward path (excluding duplicate meeting point)
    complete_path = forward_path + backward_path

    return complete_path


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

    if use_python:
        # Use Python A* implementation
        heuristic = pdd_heuristic
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
            path = astar_pathfind(
                start_cnfs,
                goal_cnfs,
                heuristic,
                filter_set,
                max_iterations=max_iterations,
                beam_width=beam_width,
                verbose=True
            )
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