"""Parameter search to find optimal xi/delta/min_distance.

Iterates through resolution levels (coarse to fine), running binary search
at each level to find the maximum min_distance that still allows pathfinding.
"""

import time
from pathlib import Path as PathlibPath
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from cnf import UnitCell
from cnf.navigation.astar import astar_rust
from cnf.navigation.astar.models import (
    PathContext, Path, Attempt, SearchParameters, SearchResult, ParameterSearchResult
)
from cnf.navigation.endpoints import get_endpoint_cnfs
from cnf.navigation.utils import compute_delta_for_step_size



_DEFAULT_XI_VALUES = [1.5, 1.25, 1.0, 0.75]
_DEFAULT_ATOM_STEP_LENGTHS = [0.4, 0.3, 0.2, 0.1]


def _binary_search_min_distance(
    start_cnfs, goal_cnfs,
    min_dist_low: float,
    min_dist_high: float,
    max_iterations: int,
    beam_width: int,
    dropout: float,
    tolerance: float,
    verbosity: int,
    log_prefix: str = "",
) -> tuple[float | None, int | None]:
    """Binary search for the maximum min_distance that allows a path.

    Args:
        verbosity: 0=silent, 1+=progress output.

    Returns:
        (best_min_distance, iterations) or (None, None) if no path found even at low bound.
    """
    best_min_dist = None
    best_iters = None

    low = min_dist_low
    high = min_dist_high

    while high - low > tolerance:
        mid = (low + high) / 2

        if verbosity >= 1:
            print(f"{log_prefix}trying min_distance={mid:.3f} Å...", end=" ", flush=True)

        result, num_iters = astar_rust(
            start_cnfs, goal_cnfs,
            min_distance=mid,
            max_iterations=max_iterations,
            beam_width=beam_width,
            dropout=dropout,
            verbose=(verbosity >= 2),
            log_prefix=log_prefix,
        )

        if result is not None:
            if verbosity >= 1:
                print(f"path found ({num_iters} iters)")
            best_min_dist = mid
            best_iters = num_iters
            low = mid
        else:
            if verbosity >= 1:
                print("no path")
            high = mid

    return best_min_dist, best_iters


def _search_at_resolution(
    start_uc: UnitCell,
    end_uc: UnitCell,
    xi: float,
    atom_step_length: float,
    min_dist_low: float,
    min_dist_high: float,
    max_iterations: int,
    beam_width: int,
    dropout: float,
    tolerance: float,
    verbosity: int,
    log_prefix: str = "",
) -> dict:
    """Search for optimal min_distance at a single resolution.

    Args:
        verbosity: 0=silent, 1+=progress output.

    Returns a dict with resolution info and search result.
    """
    start_struct = start_uc.to_pymatgen_structure()
    end_struct = end_uc.to_pymatgen_structure()

    delta = max(
        compute_delta_for_step_size(start_struct, atom_step_length),
        compute_delta_for_step_size(end_struct, atom_step_length),
    )

    if verbosity >= 1:
        print(f"{log_prefix}Resolution: xi={xi}, atom_step={atom_step_length} Å -> delta={delta}")

    start_cnfs, goal_cnfs = get_endpoint_cnfs(start_uc, end_uc, xi=xi, delta=delta)
    elements = start_cnfs[0].elements

    if verbosity >= 1:
        print(f"{log_prefix}  {len(start_cnfs)} start CNFs, {len(goal_cnfs)} goal CNFs")
        print(f"{log_prefix}  Binary search for min_distance in [{min_dist_low:.2f}, {min_dist_high:.2f}]")

    best_min_dist, best_iters = _binary_search_min_distance(
        start_cnfs, goal_cnfs,
        min_dist_low, min_dist_high,
        max_iterations, beam_width, dropout, tolerance,
        verbosity, log_prefix=f"{log_prefix}    ",
    )

    context = PathContext(xi=xi, delta=delta, elements=elements)

    if best_min_dist is not None:
        params = SearchParameters(
            max_iterations=max_iterations,
            beam_width=beam_width,
            dropout=dropout,
            filters=[{"type": "min_distance", "value": best_min_dist}],
        )
        attempt = Attempt(path=None, found=True, iterations=best_iters or 0)
    else:
        params = SearchParameters(
            max_iterations=max_iterations,
            beam_width=beam_width,
            dropout=dropout,
            filters=[{"type": "min_distance", "value": min_dist_low}],
        )
        attempt = Attempt(path=None, found=False, iterations=0)

    search_result = SearchResult(
        context=context,
        parameters=params,
        attempts=[attempt],
        metadata={
            "xi": xi,
            "atom_step_length": atom_step_length,
            "delta": delta,
            "best_min_distance": best_min_dist,
        }
    )

    return {
        "xi": xi,
        "delta": delta,
        "atom_step_length": atom_step_length,
        "min_distance": best_min_dist,
        "found": best_min_dist is not None,
        "search_result": search_result,
    }


def _worker_search_at_resolution(args):
    """Worker function for parallel resolution search."""
    (start_struct_dict, end_struct_dict, xi, atom_step_length,
     min_dist_low, min_dist_high, max_iterations, beam_width,
     dropout, tolerance, verbosity, log_prefix) = args

    from pymatgen.core import Structure
    start_struct = Structure.from_dict(start_struct_dict)
    end_struct = Structure.from_dict(end_struct_dict)
    start_uc = UnitCell.from_pymatgen_structure(start_struct)
    end_uc = UnitCell.from_pymatgen_structure(end_struct)

    return _search_at_resolution(
        start_uc, end_uc, xi, atom_step_length,
        min_dist_low, min_dist_high, max_iterations, beam_width,
        dropout, tolerance, verbosity, log_prefix,
    )


def search(
    start_uc: UnitCell,
    end_uc: UnitCell,
    xi_values: list[float] | None = None,
    atom_step_lengths: list[float] | None = None,
    min_dist_low: float = 0.5,
    min_dist_high: float = 2.0,
    tolerance: float = 0.05,
    max_iterations: int = 5_000,
    beam_width: int = 1000,
    dropout: float = 0.0,
    n_workers: int = 0,
    verbosity: int = 1,
    output_dir: PathlibPath | str | None = None,
) -> ParameterSearchResult:
    """Search for optimal discretization and filter parameters.

    Iterates through resolution levels (coarse to fine), running binary search
    at each level to find the maximum min_distance that still allows pathfinding.

    Args:
        start_uc: Starting UnitCell.
        end_uc: Ending UnitCell.
        xi_values: Lattice discretization values to try (coarse to fine).
            Default: [1.5, 1.25, 1.0, 0.75]
        atom_step_lengths: Target atom step sizes in Angstroms (coarse to fine).
            Default: [0.4, 0.3, 0.2, 0.1]
        min_dist_low: Lower bound for min_distance binary search.
        min_dist_high: Upper bound for min_distance binary search.
        tolerance: Binary search convergence tolerance (Angstroms).
        max_iterations: Max A* iterations per search.
        beam_width: Max open-set size for beam search.
        dropout: Neighbor dropout probability.
        n_workers: Parallel workers. 0 = auto, 1 = serial.
        verbosity: 0=silent, 1=phase output, 2+=detailed output.
        output_dir: Path to output directory.

    Returns:
        ParameterSearchResult with successful parameters and recommendations.
    """
    if xi_values is None:
        xi_values = _DEFAULT_XI_VALUES
    if atom_step_lengths is None:
        atom_step_lengths = _DEFAULT_ATOM_STEP_LENGTHS

    if len(xi_values) != len(atom_step_lengths):
        raise ValueError(
            f"xi_values ({len(xi_values)}) and atom_step_lengths ({len(atom_step_lengths)}) "
            "must have the same length"
        )

    if output_dir is not None:
        output_dir = PathlibPath(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    resolutions = list(zip(xi_values, atom_step_lengths))

    if verbosity >= 1:
        print(f"\nParameter search: {len(resolutions)} resolution levels")
        print(f"  xi values: {xi_values}")
        print(f"  atom step lengths: {atom_step_lengths}")
        print(f"  min_distance range: [{min_dist_low:.2f}, {min_dist_high:.2f}] Å")

    total_cores = mp.cpu_count()
    if n_workers == 0:
        n_workers = max(1, total_cores // 4)
        if verbosity >= 1:
            print(f"  Auto workers: {n_workers}")

    results = []
    successful_params = []

    if n_workers > 1 and len(resolutions) > 1:
        if verbosity >= 1:
            print(f"\n  Running {len(resolutions)} resolutions with {n_workers} workers...")

        start_struct_dict = start_uc.to_pymatgen_structure().as_dict()
        end_struct_dict = end_uc.to_pymatgen_structure().as_dict()

        args_list = [
            (start_struct_dict, end_struct_dict, xi, atom_step,
             min_dist_low, min_dist_high, max_iterations, beam_width,
             dropout, tolerance, verbosity, f"[xi={xi}] ")
            for xi, atom_step in resolutions
        ]

        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp.get_context('spawn'),
        ) as pool:
            for r in pool.map(_worker_search_at_resolution, args_list):
                results.append(r)
                if r["found"]:
                    successful_params.append((r["xi"], r["delta"], r["min_distance"]))
                    if verbosity >= 1:
                        print(f"  [xi={r['xi']}, delta={r['delta']}] "
                              f"min_distance={r['min_distance']:.3f} Å")
                else:
                    if verbosity >= 1:
                        print(f"  [xi={r['xi']}, delta={r['delta']}] no path found")
    else:
        for i, (xi, atom_step) in enumerate(resolutions):
            if verbosity >= 1:
                print(f"\n[{i+1}/{len(resolutions)}] ", end="")

            r = _search_at_resolution(
                start_uc, end_uc, xi, atom_step,
                min_dist_low, min_dist_high, max_iterations, beam_width,
                dropout, tolerance, verbosity,
            )
            results.append(r)

            if r["found"]:
                successful_params.append((r["xi"], r["delta"], r["min_distance"]))

    total_elapsed = time.perf_counter() - total_start

    recommended_xi = None
    recommended_delta = None
    recommended_min_dist = None

    if successful_params:
        recommended_xi, recommended_delta, recommended_min_dist = successful_params[-1]

    search_results = [r["search_result"] for r in results]

    result = ParameterSearchResult(
        successful_params=successful_params,
        results=search_results,
        recommended_xi=recommended_xi,
        recommended_delta=recommended_delta,
        recommended_min_distance=recommended_min_dist,
        metadata={
            "xi_values": xi_values,
            "atom_step_lengths": atom_step_lengths,
            "min_dist_low": min_dist_low,
            "min_dist_high": min_dist_high,
            "tolerance": tolerance,
            "total_elapsed_seconds": total_elapsed,
        }
    )

    if verbosity >= 1:
        print(f"\n{'='*60}")
        print(f"Parameter search complete:")
        print(f"  Successful resolutions: {len(successful_params)}/{len(resolutions)}")
        if result.success:
            print(f"  Recommended: xi={recommended_xi}, delta={recommended_delta}, "
                  f"min_distance={recommended_min_dist:.3f} Å")
        else:
            print(f"  No successful parameters found")
        print(f"  Total time: {total_elapsed:.1f}s")
        print(f"{'='*60}")

    if output_dir is not None:
        result.to_json(str(output_dir / "parameter_search_result.json"))

    return result
