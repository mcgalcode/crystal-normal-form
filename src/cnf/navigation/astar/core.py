"""A* pathfinding for CNF navigation with customizable heuristics and filters"""

import heapq
import time
import random

from typing import List, Optional, Callable

from cnf import CrystalNormalForm
from ..neighbor_finder import NeighborFinder
from ..search_filters import FilterSet

from .search_state import AStarSearchState
from .node import AStarNode

HeuristicFunc = Callable[[CrystalNormalForm, List[CrystalNormalForm]], float]

def add_start_points(start_cnfs: list[CrystalNormalForm],
                     goal_cnfs: list[CrystalNormalForm],
                     heuristic: HeuristicFunc,
                     search_state: AStarSearchState):
    # Add all start states to open set
    for start_cnf in start_cnfs:
        start_key = start_cnf.coords
        search_state.g_score[start_key] = 0.0
        h = heuristic(start_key, goal_cnfs)
        f = 0.0 + h

        heapq.heappush(search_state.open_set, AStarNode(
            f_score=f,
            g_score=0.0,
            h_score=h,
            point=start_key,
        ))

def process_node(node: AStarNode,
                 search_state: AStarSearchState,
                 neighbor_finder: NeighborFinder,
                 filter_set: FilterSet,
                 heuristic: HeuristicFunc,
                 greedy: bool,
                 beam_width: int,
                 goal_cnfs: list[CrystalNormalForm],
                 dropped: set = None,
                 dropout: float = None):
    # Mark as visited
    current_point = node.point
    search_state.closed_set.add(current_point)

    # Get neighbors
    neighbors = neighbor_finder.find_neighbor_tuples(current_point)

    if dropped is not None:
        neighbors = [n for n in neighbors if n not in dropped]
    
    if dropout is not None:
        goal_tups = [goal.coords for goal in goal_cnfs]
        new_dropped = []
        keep = []
        for nb in neighbors:

            if nb in goal_tups:
                keep.append(nb)
                continue

            if random.random() < dropout:
                new_dropped.append(nb)
            else:
                keep.append(nb)

        neighbors = keep
        dropped.update(new_dropped)

    # Apply filter if provided
    if filter_set is not None:
        neighbors = [CrystalNormalForm.from_tuple(n, search_state.elements, search_state.xi, search_state.delta) for n in neighbors]
        neighbors, _ = filter_set.filter_cnfs(neighbors)
        neighbors = [nb.coords for nb in neighbors]
    # Sort neighbors for deterministic behavior (avoids hash randomization issues)
    neighbors = sorted(neighbors)

    # Explore neighbors
    for neighbor_point in neighbors:
        in_closed = neighbor_point in search_state.closed_set
        if in_closed:
            continue

        # Edge cost: uniform cost 1
        edge_cost = 1.0
        tentative_g = node.g_score + edge_cost

        # Check if this is a better path
        current_g = search_state.g_score.get(neighbor_point, float('inf'))

        if tentative_g < current_g:
            # Update path
            search_state.came_from[neighbor_point] = current_point
            search_state.g_score[neighbor_point] = tentative_g

            # Calculate f_score
            h = heuristic(neighbor_point, goal_cnfs)

            if greedy:
                f = h
            else:
                f = tentative_g + h

            # Add to open set
            heapq.heappush(search_state.open_set, AStarNode(
                f_score=f,
                g_score=tentative_g,
                h_score=h,
                point=neighbor_point,
            ))

    # Beam search: prune open set if it exceeds beam_width
    if beam_width is not None and len(search_state.open_set) > beam_width:
        search_state.open_set = heapq.nsmallest(beam_width, search_state.open_set)
        heapq.heapify(search_state.open_set)

def astar_pathfind(
    start_cnfs: List[CrystalNormalForm],
    goal_cnfs: List[CrystalNormalForm],
    heuristic: Optional[HeuristicFunc] = None,
    filter_set: Optional[FilterSet] = None,
    max_iterations: int = 100000,
    beam_width: Optional[int] = None,
    greedy: bool = False,
    dropout: float = 0,
    verbose: bool = False,
    speak_freq = 5,
) -> AStarSearchState:
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
        AStarSearchState containing the complete search state:
        - path: List of point tuples if goal was found, None otherwise
        - open_set, closed_set, came_from, g_score: Full search state
        - found_goal: Whether a goal was reached
        - max_iterations_reached: Whether search stopped due to iteration limit
        Use state.get_top_frontier_cnfs(n) to get best frontier points for reverse search.
    """
    if not start_cnfs:
        raise ValueError("start_cnfs cannot be empty")
    if not goal_cnfs:
        raise ValueError("goal_cnfs cannot be empty")
    
    def is_goal(cnf_point: tuple) -> bool:
        return any(cnf_point == goal.coords for goal in goal_cnfs)
    
    xi = start_cnfs[0].xi
    delta = start_cnfs[0].delta
    elements = start_cnfs[0].elements

    # Initialize search structure
    search_state = AStarSearchState(
        xi=xi,
        delta=delta,
        elements=elements
    )


    neighbor_finder = NeighborFinder.from_cnf(start_cnfs[0])
    add_start_points(start_cnfs, goal_cnfs, heuristic, search_state)

    if verbose:
        print(f"Starting A* search with {len(start_cnfs)} start states and {len(goal_cnfs)} goal states")

    ts = time.perf_counter_ns()

    dropped = set()

    while search_state.open_set:
        search_state.iterations += 1

        if max_iterations > 0 and search_state.iterations > max_iterations:
            search_state.max_iterations_reached = True
            if verbose:
                print(f"Reached max iterations ({max_iterations})")
            return search_state

        if verbose and search_state.iterations % speak_freq == 0:
            te = time.perf_counter_ns()
            elapsed = round((te - ts) / 1e9, 3)
            print(f"Step {search_state.iterations}:  open={len(search_state.open_set)}, closed={len(search_state.closed_set)}, f={search_state.open_set[0].f_score:.2f}, g={search_state.open_set[0].g_score:.2f}, h={search_state.open_set[0].h_score:.6f}, Elapsed: {elapsed:.2f}s")

        # Pop node with lowest f_score
        current_node = heapq.heappop(search_state.open_set)
        current_point = current_node.point
        goal_reached = is_goal(current_point)

        if goal_reached:
            if verbose:
                print(f"\n✅ Found goal after {search_state.iterations} iterations!")
            path = _reconstruct_path(current_point, search_state.came_from)
            search_state.found_goal = True
            search_state.path = path
            return search_state
            
        process_node(current_node, search_state, neighbor_finder, filter_set, heuristic, greedy, beam_width, goal_cnfs, dropped=dropped, dropout=dropout)
        

    if verbose:
        print(f"\n❌ No path found after {search_state.iterations} iterations")

    return search_state


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
