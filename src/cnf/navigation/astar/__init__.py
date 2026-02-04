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
from .heuristics import manhattan_distance, pdd_amd_heuristic, pdd_heuristic, pdd_and_manhattan

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
):
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
            speak_freq
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
                      user_metadata=None):
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



    # Get n_atoms and elements from first start point
    first_cnf = start_cnfs[0]
    elements = first_cnf.motif_normal_form.elements
    n_atoms = len(elements)

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
                verbose=True,
                speak_freq=speak_freq
            )
            max_iterations_reached = num_iterations == max_iterations

    metadata= {
        "start_cif": start_cif,
        "end_cif": end_cif,
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


    if path is None:
        print("❌ No path found!")
    else:
        print(f"\n✅ Path found with {len(path)} steps!")


    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(asdict(result), f, indent=2)

    print(f"\n💾 Path saved to {output_json}")

    return result