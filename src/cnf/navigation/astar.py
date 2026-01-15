"""A* pathfinding for CNF navigation with customizable heuristics and filters"""

import heapq
import time
from typing import List, Optional, Callable, Set
from dataclasses import dataclass, field
import numpy as np

from cnf import CrystalNormalForm
from ..utils.pdd import pdd_for_cnfs
from .utils import no_atoms_closer_than
from .neighbor_finder import NeighborFinder


@dataclass(order=True)
class AStarNode:
    """Node in the A* search priority queue"""
    f_score: float
    g_score: float = field(compare=False)
    h_score: float = field(compare=False)
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
    return 100*min(dists) ** 2

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


def min_distance_filter(min_distance: float) -> FilterFunc:
    """
    Create a filter function that checks minimum pairwise distances.

    Uses fast Rust implementation when USE_RUST=1, otherwise falls back to Python.

    Args:
        min_distance: Minimum allowed pairwise distance in Angstroms

    Returns:
        Filter function that returns True if CNF passes, False otherwise
    """
    import os
    use_rust = os.getenv('USE_RUST') == '1'

    if use_rust:
        # Return a batched filter that processes all neighbors at once in Rust
        def filter_func_rust_batch(neighbors, xi, delta, elements) -> list:
            """Batch filter using Rust - processes all neighbors at once"""
            import rust_cnf
            n_atoms = len(elements)

            # Convert neighbors to list of lists (concatenated vonorms + coords)
            neighbor_list = [list(n) for n in neighbors]

            # Filter in Rust (returns filtered list)
            filtered = rust_cnf.filter_neighbors_by_min_distance_rust(
                neighbor_list,
                n_atoms,
                xi,
                delta,
                min_distance
            )

            # Convert back to tuples
            return [tuple(n) for n in filtered]

        # Mark this as a batch filter for astar to handle differently
        filter_func_rust_batch._is_batch_filter = True
        return filter_func_rust_batch
    else:
        # Python fallback - filters one neighbor at a time
        def filter_func(cnf_tuple, xi, delta, elements) -> bool:
            cnf = CrystalNormalForm.from_tuple(cnf_tuple, elements, xi, delta)
            return no_atoms_closer_than(cnf, min_distance)

        return filter_func


def astar_pathfind(
    start_cnfs: List[CrystalNormalForm],
    goal_cnfs: List[CrystalNormalForm],
    heuristic: Optional[HeuristicFunc] = None,
    filter_func: Optional[FilterFunc] = None,
    max_iterations: int = 100000,
    verbose: bool = False
) -> Optional[List[CrystalNormalForm]]:
    """
    A* pathfinding between CNF states with customizable heuristic and filtering.

    Args:
        start_cnfs: List of starting CNF states
        goal_cnfs: List of goal CNF states
        heuristic: Heuristic function h(cnf, goals) -> float
                   If None, uses squared Euclidean distance
        filter_func: Filter function f(cnf) -> bool to exclude invalid neighbors
                     If None, no filtering is applied
        max_iterations: Maximum number of iterations (0 for unlimited)
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

    # Performance instrumentation
    time_neighbor_finding = 0.0
    time_filtering = 0.0
    time_heuristic = 0.0
    time_heap_ops = 0.0
    time_closed_check = 0.0
    time_goal_check = 0.0
    time_other = 0.0

    total_neighbors_generated = 0
    total_neighbors_filtered_out = 0

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
            total_instrumented_time = (time_neighbor_finding + time_filtering + time_heuristic +
                                      time_heap_ops + time_closed_check + time_goal_check + time_other)

            print(f"Step {iterations}:  open={len(open_set)}, closed={len(closed_set)}, f_score={open_set[0].f_score:.2f}, h={open_set[0].h_score:.2f}, Elapsed: {elapsed:.2f}s")

            # if iterations >= 100 and total_instrumented_time > 0:
            #     print(f"  Time breakdown:")
            #     print(f"    Neighbor finding: {time_neighbor_finding/1e6:7.1f} ms ({100*time_neighbor_finding/total_instrumented_time:5.1f}%)")
            #     print(f"    Filtering:        {time_filtering/1e6:7.1f} ms ({100*time_filtering/total_instrumented_time:5.1f}%)")
            #     print(f"    Heuristic calc:   {time_heuristic/1e6:7.1f} ms ({100*time_heuristic/total_instrumented_time:5.1f}%)")
            #     print(f"    Heap operations:  {time_heap_ops/1e6:7.1f} ms ({100*time_heap_ops/total_instrumented_time:5.1f}%)")
            #     print(f"    Closed set check: {time_closed_check/1e6:7.1f} ms ({100*time_closed_check/total_instrumented_time:5.1f}%)")
            #     print(f"    Goal check:       {time_goal_check/1e6:7.1f} ms ({100*time_goal_check/total_instrumented_time:5.1f}%)")
            #     print(f"    Other:            {time_other/1e6:7.1f} ms ({100*time_other/total_instrumented_time:5.1f}%)")

            #     avg_neighbors = total_neighbors_generated / iterations if iterations > 0 else 0
            #     print(f"  Neighbor stats:")
            #     print(f"    Total generated: {total_neighbors_generated}")
            #     print(f"    Total filtered:  {total_neighbors_filtered_out}")
            #     print(f"    Avg per iter:    {avg_neighbors:.1f}")

        # Pop node with lowest f_score
        t_start = time.perf_counter_ns()
        current_node = heapq.heappop(open_set)
        time_heap_ops += time.perf_counter_ns() - t_start

        current_point = current_node.point

        # Check if we reached a goal
        t_start = time.perf_counter_ns()
        goal_reached = is_goal(current_point)
        time_goal_check += time.perf_counter_ns() - t_start

        if goal_reached:
            if verbose:
                print(f"\n✅ Found goal after {iterations} iterations!")
            return _reconstruct_path(current_point, came_from)

        # Mark as visited
        t_start = time.perf_counter_ns()
        closed_set.add(current_point)
        time_other += time.perf_counter_ns() - t_start

        # Get neighbors
        t_start = time.perf_counter_ns()
        neighbors = neighbor_finder.find_neighbor_tuples(current_point)
        time_neighbor_finding += time.perf_counter_ns() - t_start

        total_neighbors_generated += len(neighbors)

        if iterations == 1 and verbose:
            print(f"First iteration: found {len(neighbors)} raw neighbors")

        # Apply filter if provided
        if filter_func is not None:
            neighbors_before = len(neighbors)
            t_start = time.perf_counter_ns()

            # Check if this is a batch filter (processes all neighbors at once)
            if hasattr(filter_func, '_is_batch_filter') and filter_func._is_batch_filter:
                # Batch filtering (Rust)
                neighbors = filter_func(neighbors, xi, delta, elements)
            else:
                # Single-item filtering (Python)
                neighbors = [n for n in neighbors if filter_func(n, xi, delta, elements)]

            time_filtering += time.perf_counter_ns() - t_start

            total_neighbors_filtered_out += neighbors_before - len(neighbors)

            if iterations == 1 and verbose:
                print(f"After filtering: {len(neighbors)} neighbors remain")

        # Sort neighbors for deterministic behavior (avoids hash randomization issues)
        neighbors = sorted(neighbors)

        # Explore neighbors
        for neighbor_point in neighbors:

            t_start = time.perf_counter_ns()
            in_closed = neighbor_point in closed_set
            time_closed_check += time.perf_counter_ns() - t_start

            if in_closed:
                continue

            # Edge cost: uniform cost 1
            t_start = time.perf_counter_ns()
            edge_cost = 1.0
            tentative_g = current_node.g_score + edge_cost

            # Check if this is a better path
            current_g = g_score.get(neighbor_point, float('inf'))
            time_other += time.perf_counter_ns() - t_start

            if tentative_g < current_g:
                t_start = time.perf_counter_ns()
                # Update path
                came_from[neighbor_point] = current_point
                g_score[neighbor_point] = tentative_g
                time_other += time.perf_counter_ns() - t_start

                # Calculate f_score
                t_start = time.perf_counter_ns()
                h = heuristic(neighbor_point, goal_cnfs)
                time_heuristic += time.perf_counter_ns() - t_start

                t_start = time.perf_counter_ns()
                f = tentative_g + h
                time_other += time.perf_counter_ns() - t_start

                # Add to open set
                t_start = time.perf_counter_ns()
                heapq.heappush(open_set, AStarNode(
                    f_score=f,
                    g_score=tentative_g,
                    h_score=h,
                    point=neighbor_point,
                    counter=counter
                ))
                time_heap_ops += time.perf_counter_ns() - t_start

                counter += 1

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
