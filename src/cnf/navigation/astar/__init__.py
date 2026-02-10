"""A* pathfinding for CNF navigation with customizable heuristics and filters"""

import os
import json
import rust_cnf

from typing import Callable
from dataclasses import asdict

from cnf import CrystalNormalForm
from ..search_filters import FilterSet, MinDistanceFilter
from pymatgen.core import Structure
from ..endpoints import get_endpoint_unit_cells
from .core import astar_pathfind
from .search_result import PathSearchResult
from .bidirectional import bidirectional_astar_pathfind
from .heuristics import manhattan_distance, pdd_amd_heuristic, pdd_heuristic, pdd_and_manhattan, UnimodularManhattanHeuristic

USE_RUST = os.getenv('USE_RUST') == '1'

# Type aliases for clarity
FilterFunc = Callable[[CrystalNormalForm], bool]

def astar_rust(
    start_cnfs: list[CrystalNormalForm],
    goal_cnfs: list[CrystalNormalForm],
    min_distance: float = 0.0,
    max_iterations: int = 100000,
    beam_width: int = None,
    greedy: bool = False,
    dropout: float = 0,
    verbose: bool = False,
    bidirectional: bool = False,
    speak_freq = 5,
    heuristic_mode: str = "manhattan",
    heuristic_weight: float = 0.5,
):
    """Run A* pathfinding using the Rust implementation.

    Args:
        start_cnfs: Starting CNF structures.
        goal_cnfs: Goal CNF structures.
        min_distance: Minimum allowed pairwise atomic distance (Angstroms).
        max_iterations: Maximum search iterations (0 for unlimited).
        beam_width: Max open-set size for beam search (None for unlimited).
        greedy: If True, use greedy best-first (f=h) instead of A* (f=g+h).
        dropout: Probability of permanently dropping a neighbor (0.0-1.0).
            Goal neighbors are never dropped.
        verbose: Print progress during search.
        bidirectional: Use bidirectional A* (ignores heuristic settings).
        speak_freq: Print progress every N iterations.
        heuristic_mode: Which heuristic to use. One of:
            - "manhattan": Plain L1 distance with 10x scaling. Fast but
              overestimates at Voronoi class boundaries.
            - "unimodular_light": Pre-computes 168 vonorm-permutation-derived
              goal variants (one matrix per permutation). Low overhead.
            - "unimodular_partial": One unimodular matrix per (zero_set,
              conorm_perm) across all coforms. Good accuracy/speed trade-off.
            - "unimodular_full": All unimodular matrices across all coforms.
              Tightest heuristic but slowest precomputation.
        heuristic_weight: Multiplier on the unimodular heuristic value.
            Only used when heuristic_mode is not "manhattan". Lower values
            make the search more exploratory; higher values more greedy.

    Returns:
        Tuple of (path, iterations) where path is a list of flat coordinate
        vectors (vonorms + coords) or None, and iterations is the count.
    """
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

    n_atoms = len(start_cnfs[0].elements)

    if bidirectional:
        return rust_cnf.bidirectional_astar_pathfind_rust(
            start_points,
            goal_points,
            start_cnfs[0].elements,
            n_atoms,
            start_cnfs[0].xi,
            start_cnfs[0].delta,
            min_distance=min_distance,
            max_iterations=max_iterations,
            beam_width=beam_width if beam_width is not None else 0,
            verbose=True
        )       
    else:
        return rust_cnf.astar_pathfind_rust(
            start_points,
            goal_points,
            start_cnfs[0].elements,
            n_atoms,
            start_cnfs[0].xi,
            start_cnfs[0].delta,
            min_distance,
            max_iterations,
            beam_width if beam_width is not None else 0,
            dropout,
            greedy,
            verbose,
            speak_freq,
            heuristic_mode,
            heuristic_weight,
        )

def pathfind_and_save(start_cif,
                      end_cif,
                      output_json,
                      xi=0.2,
                      delta=30,
                      min_distance=0.0,
                      max_iterations=100000,
                      use_python=False,
                      bidirectional=False,
                      greedy=False,
                      beam_width=None,
                      dropout=0.0,
                      speak_freq=5,
                      user_metadata=None,
                      verbose=False,
                      heuristic=None,
                      heuristic_mode="manhattan",
                      heuristic_weight=0.5):
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
        dropout: Probability of dropping a neighbor (0.0 to 1.0). Dropped neighbors are excluded
                 from consideration for the rest of the search. Goal neighbors are never dropped.
                 Only supported with Rust implementation (default: 0.0)
    """

    if user_metadata is None:
        user_metadata = {}

    start_struct = Structure.from_file(start_cif)
    end_struct = Structure.from_file(end_cif)

    start_cells, goal_cells = get_endpoint_unit_cells(start_struct, end_struct)

    if verbose:
        print(f"\n=== Endpoints ===")
        print(f"Number of start cells: {len(start_cells)}")
        print(f"Number of goal cells: {len(goal_cells)}")

    # Convert unit cells to CNFs and deduplicate
    start_cnfs = list(set([cell.to_cnf(xi=xi, delta=delta) for cell in start_cells]))
    goal_cnfs = list(set([cell.to_cnf(xi=xi, delta=delta) for cell in goal_cells]))

    if verbose:
        print(f"Unique start CNFs: {len(start_cnfs)}")
        for sc in start_cnfs:
            print(f"    {sc.coords}")
        print(f"Unique goal CNFs: {len(goal_cnfs)}")
        for ec in goal_cnfs:
            print(f"    {ec.coords}")



    # Get n_atoms and elements from first start point
    first_cnf = start_cnfs[0]
    elements = first_cnf.motif_normal_form.elements
    n_atoms = len(elements)

    if verbose:
        print(f"\n=== Running A* pathfinding ===")
        print(f"Implementation: {'Python' if use_python else 'Rust'}")
        print(f"Elements: {elements}")
        print(f"N atoms: {n_atoms}")
        print(f"Xi: {xi}, Delta: {delta}")
        print(f"Start points: {len(start_cnfs)}")
        print(f"Goal points: {len(goal_cnfs)}")

    search_state = None  # Will be set if using Python non-bidirectional A*
    max_iterations_reached = None
    if use_python:

        if heuristic_mode == "manhattan":
            heuristic = manhattan_distance
        elif heuristic_mode == "unimodular_light":
            heuristic = UnimodularManhattanHeuristic(weight=heuristic_weight, full=False, partial=False)
        elif heuristic_mode == "unimodular_partial":
            heuristic = UnimodularManhattanHeuristic(weight=heuristic_weight, full=False, partial=True)
        elif heuristic_mode == "unimodular_full":
            heuristic = UnimodularManhattanHeuristic(weight=heuristic_weight, full=True, partial=False)
        

        filter_set = FilterSet([MinDistanceFilter(min_distance)], use_structs = not USE_RUST)
        
        if bidirectional:
            path = bidirectional_astar_pathfind(
                start_cnfs,
                goal_cnfs,
                heuristic,
                filter_set,
                max_iterations=max_iterations,
                beam_width=beam_width,
                greedy=greedy,
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
                greedy=greedy,
                verbose=True,
                dropout=dropout,
            )
            path = search_state.path
            max_iterations_reached = search_state.max_iterations_reached
            num_iterations = search_state.iterations
    else:
        # Use Rust A* implementation (default)
        if bidirectional:
            path = rust_cnf.bidirectional_astar_pathfind_rust(
                start_cnfs,
                goal_cnfs,
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
            path, num_iterations = astar_rust(
                start_cnfs,
                goal_cnfs,
                min_distance=min_distance,
                max_iterations=max_iterations,
                beam_width=beam_width,
                greedy=greedy,
                dropout=dropout,
                verbose=verbose,
                speak_freq=speak_freq,
                heuristic_mode=heuristic_mode,
                heuristic_weight=heuristic_weight,
            )
            max_iterations_reached = num_iterations == max_iterations

    metadata= {
        "start_cif": str(start_cif),
        "end_cif": str(end_cif),
        "min_distance": min_distance,
        **user_metadata,
    }

    result = PathSearchResult(
        path=path,
        max_iterations_reached=max_iterations_reached,
        num_iterations=num_iterations,
        xi=xi,
        delta=delta,
        elements=elements,
        n_atoms=len(elements),
        
        greedy=greedy,
        beam_width=beam_width,
        dropout=dropout,
        max_iterations=max_iterations,

        metadata=metadata
    ) 


    if verbose:
        if path is None:
            print("❌ No path found!")
        else:
            print(f"\n✅ Path found with {len(path)} steps!")


    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(asdict(result), f, indent=2)

    if verbose:
        print(f"\n💾 Path saved to {output_json}")

    return result