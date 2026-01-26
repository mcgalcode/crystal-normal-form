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
from ...utils.pdd import pdd_for_cnfs, pdd_amd_for_cnfs
from ..utils import no_atoms_closer_than
from ..neighbor_finder import NeighborFinder
from ..search_filters import EnergyFilter, FilterSet, MinDistanceFilter

from pymatgen.core import Structure

from ..endpoints import get_endpoint_unit_cells

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


@dataclass
class AStarSearchState:
    """Complete state of an A* search, for saving/resuming or extracting frontier points"""
    open_set: list[AStarNode]  # List of AStarNode
    closed_set: Set[tuple]
    came_from: dict  # point -> parent point
    g_score: dict  # point -> cost from start
    iterations: int
    path: Optional[List[tuple]]  # The path if found, None otherwise
    found_goal: bool
    max_iterations_reached: bool

    # Store CNF metadata for reconstructing CNF objects
    xi: float = field(default=0.2)
    delta: int = field(default=30)
    elements: tuple = field(default_factory=tuple)

    def get_top_frontier_points(self, n: int = 10, by: str = 'h_score') -> List[tuple]:
        """
        Get the n best points from the open set.

        Args:
            n: Number of points to return
            by: Sort criterion - 'h_score' (closest to goal), 'f_score' (best overall),
                or 'g_score' (most explored)

        Returns:
            List of point tuples (vonorms + coords concatenated)
        """
        if not self.open_set:
            return []

        if by == 'h_score':
            sorted_nodes = sorted(self.open_set, key=lambda x: x.h_score)
        elif by == 'f_score':
            sorted_nodes = sorted(self.open_set, key=lambda x: x.f_score)
        elif by == 'g_score':
            sorted_nodes = sorted(self.open_set, key=lambda x: -x.g_score)  # Higher g = more explored
        else:
            raise ValueError(f"Unknown sort criterion: {by}")

        return [node.point for node in sorted_nodes[:n]]

    def get_top_frontier_cnfs(self, n: int = 10, by: str = 'h_score') -> List['CrystalNormalForm']:
        """
        Get the n best points from the open set as CNF objects.

        Args:
            n: Number of CNFs to return
            by: Sort criterion - 'h_score', 'f_score', or 'g_score'

        Returns:
            List of CrystalNormalForm objects
        """
        points = self.get_top_frontier_points(n, by)
        return [
            CrystalNormalForm.from_tuple(pt, self.elements, self.xi, self.delta)
            for pt in points
        ]

    def frontier_stats(self) -> dict:
        """Get statistics about the current frontier (open set)"""
        if not self.open_set:
            return {'size': 0}

        h_scores = [n.h_score for n in self.open_set]
        g_scores = [n.g_score for n in self.open_set]
        f_scores = [n.f_score for n in self.open_set]

        return {
            'size': len(self.open_set),
            'closed_size': len(self.closed_set),
            'h_min': min(h_scores),
            'h_max': max(h_scores),
            'h_mean': sum(h_scores) / len(h_scores),
            'g_min': min(g_scores),
            'g_max': max(g_scores),
            'g_mean': sum(g_scores) / len(g_scores),
            'f_min': min(f_scores),
            'f_max': max(f_scores),
        }

    def to_dict(self) -> dict:
        """Serialize the search state to a dictionary for JSON saving"""
        return {
            'open_set': [
                {
                    'f_score': node.f_score,
                    'g_score': node.g_score,
                    'h_score': node.h_score,
                    'point': list(node.point),
                    'counter': node.counter
                }
                for node in self.open_set
            ],
            'closed_set': [list(pt) for pt in self.closed_set],
            'came_from': {str(list(k)): list(v) for k, v in self.came_from.items()},
            'g_score': {str(list(k)): v for k, v in self.g_score.items()},
            'iterations': self.iterations,
            'path': [list(pt) for pt in self.path] if self.path else None,
            'found_goal': self.found_goal,
            'max_iterations_reached': self.max_iterations_reached,
            'xi': self.xi,
            'delta': self.delta,
            'elements': list(self.elements)
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AStarSearchState':
        """Deserialize a search state from a dictionary"""
        open_set = [
            AStarNode(
                f_score=node['f_score'],
                g_score=node['g_score'],
                h_score=node['h_score'],
                point=tuple(node['point']),
                counter=node['counter']
            )
            for node in data['open_set']
        ]
        heapq.heapify(open_set)

        closed_set = {tuple(pt) for pt in data['closed_set']}
        came_from = {tuple(json.loads(k)): tuple(v) for k, v in data['came_from'].items()}
        g_score = {tuple(json.loads(k)): v for k, v in data['g_score'].items()}
        path = [tuple(pt) for pt in data['path']] if data['path'] else None

        return cls(
            open_set=open_set,
            closed_set=closed_set,
            came_from=came_from,
            g_score=g_score,
            iterations=data['iterations'],
            path=path,
            found_goal=data['found_goal'],
            max_iterations_reached=data['max_iterations_reached'],
            xi=data['xi'],
            delta=data['delta'],
            elements=tuple(data['elements'])
        )


def astar_pathfind(
    start_cnfs: List[CrystalNormalForm],
    goal_cnfs: List[CrystalNormalForm],
    heuristic: Optional[HeuristicFunc] = None,
    filter_set: Optional[FilterSet] = None,
    max_iterations: int = 100000,
    beam_width: Optional[int] = None,
    greedy: bool = False,
    verbose: bool = False
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
            return AStarSearchState(
                open_set=open_set,
                closed_set=closed_set,
                came_from=came_from,
                g_score=g_score,
                iterations=iterations,
                path=None,
                found_goal=False,
                max_iterations_reached=True,
                xi=xi,
                delta=delta,
                elements=elements
            )

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
            path = _reconstruct_path(current_point, came_from)
            return AStarSearchState(
                open_set=open_set,
                closed_set=closed_set,
                came_from=came_from,
                g_score=g_score,
                iterations=iterations,
                path=path,
                found_goal=True,
                max_iterations_reached=False,
                xi=xi,
                delta=delta,
                elements=elements
            )

        # Mark as visited
        closed_set.add(current_point)

        # Get neighbors
        neighbors = neighbor_finder.find_neighbor_tuples(current_point)

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

                if greedy:
                    f = h
                else:
                    f = tentative_g + h

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

    return AStarSearchState(
        open_set=open_set,
        closed_set=closed_set,
        came_from=came_from,
        g_score=g_score,
        iterations=iterations,
        path=None,
        found_goal=False,
        max_iterations_reached=False,
        xi=xi,
        delta=delta,
        elements=elements
    )


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
                    # Reconstruct and return pa
                    # th immediately
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


def load_search_state(json_path: str) -> Optional[AStarSearchState]:
    """
    Load a saved search state from a JSON file.

    Args:
        json_path: Path to the JSON file containing the saved state

    Returns:
        AStarSearchState if the file contains a search_state, None otherwise
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    if 'search_state' not in data:
        return None

    return AStarSearchState.from_dict(data['search_state'])


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