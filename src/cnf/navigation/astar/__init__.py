"""A* pathfinding for CNF navigation"""

import os
from cnf import rust_cnf

from cnf import CrystalNormalForm
from ..search_filters import FilterSet, MinDistanceFilter
from pymatgen.core import Structure
from ..endpoints import get_endpoint_unit_cells, get_endpoint_cnfs_with_resolution
from .core import astar_pathfind
from .models import PathContext, Path, Attempt, SearchParameters, SearchResult
from .heuristics import manhattan_distance

USE_RUST = os.getenv('USE_RUST') == '1'


def astar_rust(
    start_cnfs: list[CrystalNormalForm],
    goal_cnfs: list[CrystalNormalForm],
    min_distance: float = 0.0,
    max_iterations: int = 10000,
    beam_width: int = None,
    greedy: bool = False,
    dropout: float = 0,
    verbose: bool = False,
    speak_freq=5,
    log_prefix: str = "",
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
        verbose: Print progress during search.
        speak_freq: Print progress every N iterations.
        log_prefix: String to prepend to all log output (e.g., "[xi=1.5] ").

    Returns:
        Tuple of (path, iterations) where path is a list of flat coordinate
        vectors (vonorms + coords) or None, and iterations is the count.
    """
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
        "manhattan",
        1.0,
        log_prefix,
    )


def pathfind(
    start_cnfs: list[CrystalNormalForm],
    goal_cnfs: list[CrystalNormalForm],
    min_distance: float = 0.0,
    max_iterations: int = 10000,
    beam_width: int = 1000,
    greedy: bool = False,
    dropout: float = 0.0,
    use_python: bool = False,
    verbose: bool = True,
) -> SearchResult:
    """Run A* pathfinding and return a SearchResult.

    Args:
        start_cnfs: Starting CNF structures.
        goal_cnfs: Goal CNF structures.
        min_distance: Minimum allowed pairwise atomic distance (Angstroms).
        max_iterations: Maximum search iterations.
        beam_width: Max open-set size for beam search.
        greedy: If True, use greedy best-first (f=h) instead of A* (f=g+h).
        dropout: Probability of dropping a neighbor (0.0-1.0).
        use_python: Use Python implementation instead of Rust.
        verbose: Print progress during search.

    Returns:
        SearchResult containing a single Attempt.
    """
    first_cnf = start_cnfs[0]
    elements = first_cnf.elements
    xi = first_cnf.xi
    delta = first_cnf.delta

    context = PathContext(xi=xi, delta=delta, elements=elements)

    filters = []
    if min_distance > 0:
        filters.append({"type": "min_distance", "value": min_distance})

    params = SearchParameters(
        max_iterations=max_iterations,
        beam_width=beam_width,
        dropout=dropout,
        greedy=greedy,
        heuristic="manhattan",
        filters=filters,
    )

    if verbose:
        print(f"A* pathfinding: {'Python' if use_python else 'Rust'}")
        print(f"  Elements: {elements}")
        print(f"  xi={xi}, delta={delta}")
        print(f"  {len(start_cnfs)} start CNFs, {len(goal_cnfs)} goal CNFs")

    if use_python:
        filter_set = FilterSet([MinDistanceFilter(min_distance)], use_structs=not USE_RUST)

        search_state = astar_pathfind(
            start_cnfs,
            goal_cnfs,
            manhattan_distance,
            filter_set,
            max_iterations=max_iterations,
            beam_width=beam_width,
            greedy=greedy,
            verbose=verbose,
            dropout=dropout,
        )
        path_tuples = search_state.path
        num_iterations = search_state.iterations
    else:
        path_tuples, num_iterations = astar_rust(
            start_cnfs,
            goal_cnfs,
            min_distance=min_distance,
            max_iterations=max_iterations,
            beam_width=beam_width,
            greedy=greedy,
            dropout=dropout,
            verbose=verbose,
        )

    if path_tuples is None:
        attempt = Attempt(path=None, found=False, iterations=num_iterations)
        if verbose:
            print("No path found")
    else:
        path_obj = Path(coords=[tuple(pt) for pt in path_tuples])
        attempt = Attempt(path=path_obj, found=True, iterations=num_iterations)
        if verbose:
            print(f"Path found: {len(path_tuples)} steps, {num_iterations} iterations")

    return SearchResult(context=context, parameters=params, attempts=[attempt])


def pathfind_from_cifs(
    start_cif: str,
    end_cif: str,
    xi: float = 0.2,
    delta: int | None = None,
    atom_step_length: float | None = None,
    min_distance: float = 0.0,
    max_iterations: int = 100_000,
    beam_width: int = 1000,
    greedy: bool = False,
    dropout: float = 0.0,
    min_atoms: int | None = None,
    use_python: bool = False,
    verbose: bool = True,
) -> SearchResult:
    """Run A* pathfinding between two CIF structures.

    Args:
        start_cif: Path to starting structure CIF file.
        end_cif: Path to ending structure CIF file.
        xi: Lattice discretization parameter.
        delta: Motif discretization parameter. If None, computed from atom_step_length
            or defaults to 30.
        atom_step_length: Target physical step size in Angstroms. Used to compute
            delta if delta is not provided. Ensures correct resolution when using
            min_atoms supercells.
        min_distance: Minimum allowed pairwise atomic distance (Angstroms).
        max_iterations: Maximum search iterations.
        beam_width: Max open-set size for beam search.
        greedy: If True, use greedy best-first search.
        dropout: Probability of dropping a neighbor (0.0-1.0).
        min_atoms: Minimum atoms (will create supercells if needed).
        use_python: Use Python implementation instead of Rust.
        verbose: Print progress.

    Returns:
        SearchResult containing a single Attempt.
    """
    start_struct = Structure.from_file(start_cif)
    end_struct = Structure.from_file(end_cif)

    # Default to delta=30 if neither delta nor atom_step_length provided
    if delta is None and atom_step_length is None:
        delta = 30

    # Use get_endpoint_cnfs_with_resolution to handle delta computation correctly
    start_cnfs, goal_cnfs, delta = get_endpoint_cnfs_with_resolution(
        start_struct, end_struct, xi=xi, delta=delta,
        atom_step_length=atom_step_length, min_atoms=min_atoms
    )

    if verbose:
        print(f"Endpoints: {len(start_cnfs)} start CNFs, {len(goal_cnfs)} goal CNFs, delta={delta}")

    result = pathfind(
        start_cnfs,
        goal_cnfs,
        min_distance=min_distance,
        max_iterations=max_iterations,
        beam_width=beam_width,
        greedy=greedy,
        dropout=dropout,
        use_python=use_python,
        verbose=verbose,
    )

    result.metadata["start_cif"] = str(start_cif)
    result.metadata["end_cif"] = str(end_cif)

    return result
