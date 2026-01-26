"""A* pathfinding for CNF navigation with customizable heuristics and filters"""

import heapq
from typing import List, Optional

from cnf import CrystalNormalForm
from ..neighbor_finder import NeighborFinder
from ..search_filters import FilterSet
from .search_state import AStarSearchState
from .core import HeuristicFunc, add_start_points, process_node

def bidirectional_astar_pathfind(
    start_cnfs: List[CrystalNormalForm],
    goal_cnfs: List[CrystalNormalForm],
    heuristic: Optional[HeuristicFunc] = None,
    filter_set: Optional[FilterSet] = None,
    max_iterations: int = 100000,
    beam_width: Optional[int] = None,
    greedy: bool = False,
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

    xi = start_cnfs[0].xi
    delta = start_cnfs[0].delta
    elements = start_cnfs[0].elements

    # Forward search structures
    forward_search_state = AStarSearchState(
        xi=xi,
        delta=delta,
        elements=elements
    )

    # Backward search structures
    backward_search_state = AStarSearchState(
        xi=xi,
        delta=delta,
        elements=elements
    )

    neighbor_finder = NeighborFinder.from_cnf(start_cnfs[0])

    add_start_points(start_cnfs, goal_cnfs, heuristic, forward_search_state)
    add_start_points(goal_cnfs, start_cnfs, heuristic, backward_search_state)
    # Initialize forward search with all start states

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

    while forward_search_state.open_set and backward_search_state.open_set:
        forward_search_state.iterations += 1

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
        if forward_search_state.open_set:
            current_node = heapq.heappop(forward_search_state.open_set)
            last_forward_g = current_node.g_score
            last_forward_h = current_node.h_score
            current_point = current_node.point

            # Check if we've met the backward search
            if current_point in backward_search_state.closed_set:
                path_length = forward_search_state.g_score[current_point] + backward_search_state.g_score[current_point]
                if path_length < best_path_length:
                    best_path_length = path_length
                    meeting_point = current_point
                    if verbose:
                        print(f"\n✅ Found meeting point at iteration {forward_search_state.iterations}! Path length: {path_length}")
                    # Reconstruct and return pa
                    # th immediately
                    return _reconstruct_bidirectional_path(
                        meeting_point, forward_search_state.came_from, backward_search_state.came_from
                    )

            process_node(current_node, forward_search_state, neighbor_finder, filter_set, heuristic, greedy, beam_width)

        # Expand from backward search
        if backward_search_state.open_set:
            current_node = heapq.heappop(backward_search_state.open_set)
            last_backward_g = current_node.g_score
            last_backward_h = current_node.h_score
            current_point = current_node.point

            # Check if we've met the forward search
            if current_point in forward_search_state.closed_set:
                path_length = forward_search_state.g_score[current_point] + backward_search_state.g_score[current_point]
                if path_length < best_path_length:
                    best_path_length = path_length
                    meeting_point = current_point
                    if verbose:
                        print(f"\n✅ Found meeting point at iteration {forward_search_state.iterations}! Path length: {path_length}")
                    # Reconstruct and return path immediately
                    return _reconstruct_bidirectional_path(
                        meeting_point, forward_search_state.came_from, backward_search_state.came_from
                    )

            process_node(current_node, backward_search_state, neighbor_finder, filter_set, heuristic, greedy, beam_width)


    if meeting_point:
        if verbose:
            print(f"\n✅ Found path with length {best_path_length}")
        return _reconstruct_bidirectional_path(meeting_point, forward_search_state.came_from, backward_search_state.came_from)

    if verbose:
        print(f"\n❌ No path found after {forward_search_state.iterations} iterations")

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