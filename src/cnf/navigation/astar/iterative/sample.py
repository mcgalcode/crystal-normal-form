"""Path sampling to discover initial energy ceiling.

Runs multiple A* searches with varied dropout to generate diverse paths,
evaluates energies, and returns the best barrier as a starting point
for ceiling sweep (Phase 3).
"""

import random
import time
from pathlib import Path as PathlibPath

from cnf import CrystalNormalForm
from cnf.calculation.grace import GraceCalculator
from cnf.navigation.astar.core import astar_pathfind
from cnf.navigation.astar.heuristics import manhattan_distance
from cnf.navigation.astar.models import (
    PathContext, Path, Attempt, SearchParameters, SearchResult
)
from cnf.navigation.search_filters import FilterSet, MinDistanceFilter

from ._energy import evaluate_path_energies, path_barrier


def sample(
    start_cnfs: list[CrystalNormalForm],
    goal_cnfs: list[CrystalNormalForm],
    energy_calc=None,
    num_samples: int = 20,
    dropout_range: tuple[float, float] = (0.3, 0.7),
    min_distance: float | None = None,
    max_iterations: int = 5_000,
    beam_width: int = 1000,
    bidirectional: bool = True,
    verbose: bool = True,
    output_dir: PathlibPath | str | None = None,
) -> SearchResult:
    """Sample diverse paths to discover initial energy ceiling.

    Runs multiple A* searches with varied dropout to generate path diversity.
    No energy ceiling filter is applied - this is used to discover what
    ceiling to use for Phase 3 (sweep).

    Args:
        start_cnfs: Starting CNF structures.
        goal_cnfs: Goal CNF structures.
        energy_calc: Energy calculator (default: GraceCalculator).
        num_samples: Number of pathfinding attempts.
        dropout_range: (min, max) dropout probability range. Each attempt
            uses a random dropout in this range for path diversity.
        min_distance: Optional minimum interatomic distance filter.
        max_iterations: Max A* iterations per attempt.
        beam_width: Max open-set size for beam search.
        bidirectional: If True, randomly swap start/goal for ~half the attempts
            to explore paths in both directions.
        verbose: Print progress.
        output_dir: Path to output directory. If set, writes sample_result.json
            after completion.

    Returns:
        SearchResult containing all attempts. Use result.best_barrier as
        the initial ceiling for sweep().
    """
    if energy_calc is None:
        energy_calc = GraceCalculator()

    elements = start_cnfs[0].elements
    xi = start_cnfs[0].xi
    delta = start_cnfs[0].delta

    context = PathContext(xi=xi, delta=delta, elements=elements)

    energy_cache = {}

    # Evaluate endpoint energies and seed the cache
    for cnf in start_cnfs + goal_cnfs:
        if cnf.coords not in energy_cache:
            energy_cache[cnf.coords] = energy_calc.calculate_energy(cnf)

    if output_dir is not None:
        output_dir = PathlibPath(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    filters = []
    if min_distance is not None:
        filters.append({"type": "min_distance", "value": min_distance})

    base_params = SearchParameters(
        max_iterations=max_iterations,
        beam_width=beam_width,
        dropout=0.0,  # placeholder, actual dropout varies per attempt
        greedy=False,
        heuristic="manhattan",
        filters=filters,
    )

    total_start = time.perf_counter()
    attempts = []

    if verbose:
        print(f"\nPath sampling: {num_samples} attempts "
              f"(dropout {dropout_range[0]:.1f}-{dropout_range[1]:.1f})")
        if min_distance is not None:
            print(f"  Min distance filter: {min_distance:.2f} Å")

    for attempt_idx in range(num_samples):
        attempt_dropout = random.uniform(dropout_range[0], dropout_range[1])

        if bidirectional and random.random() < 0.5:
            attempt_starts, attempt_goals = goal_cnfs, start_cnfs
            direction = "backward"
        else:
            attempt_starts, attempt_goals = start_cnfs, goal_cnfs
            direction = "forward"

        if verbose:
            print(f"  [{attempt_idx+1}/{num_samples}] "
                  f"dropout={attempt_dropout:.2f} {direction}...", end=" ", flush=True)

        attempt_start = time.perf_counter()

        filter_list = []
        if min_distance is not None:
            filter_list.append(MinDistanceFilter(min_distance))
        filter_set = FilterSet(filter_list) if filter_list else None

        search_state = astar_pathfind(
            attempt_starts,
            attempt_goals,
            heuristic=manhattan_distance,
            filter_set=filter_set,
            max_iterations=max_iterations,
            beam_width=beam_width,
            dropout=attempt_dropout,
            verbose=False,
        )

        attempt_elapsed = time.perf_counter() - attempt_start
        num_iters = search_state.iterations

        if search_state.path is None:
            if verbose:
                print("no path")
            attempts.append(Attempt(
                path=None,
                found=False,
                iterations=num_iters,
                elapsed_seconds=attempt_elapsed,
                metadata={"dropout": attempt_dropout, "direction": direction},
            ))
            continue

        path_tuples = search_state.path

        if direction == "backward":
            path_tuples = list(reversed(path_tuples))

        energies = evaluate_path_energies(
            path_tuples, elements, xi, delta, energy_calc, energy_cache
        )
        barrier = path_barrier(energies)

        if verbose:
            print(f"len={len(path_tuples)}, barrier={barrier:.4f} eV, iters={num_iters}")

        path_obj = Path(
            coords=[tuple(pt) for pt in path_tuples],
            energies=energies,
            barrier=barrier,
        )

        attempts.append(Attempt(
            path=path_obj,
            found=True,
            iterations=num_iters,
            elapsed_seconds=attempt_elapsed,
            metadata={"dropout": attempt_dropout, "direction": direction},
        ))

    total_elapsed = time.perf_counter() - total_start

    result = SearchResult(
        context=context,
        parameters=base_params,
        attempts=attempts,
        metadata={
            "num_samples": num_samples,
            "dropout_range": list(dropout_range),
            "bidirectional": bidirectional,
            "total_elapsed_seconds": total_elapsed,
            "energy_cache_size": len(energy_cache),
        }
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"Sampling complete:")
        print(f"  Paths found: {len(result.paths)}/{num_samples}")
        best = result.best_path
        if best:
            print(f"  Best barrier: {best.barrier:.4f} eV")
            print(f"  Best path length: {len(best)} steps")
        else:
            print(f"  No paths found")
        print(f"  Energy cache size: {len(energy_cache)}")
        print(f"  Total time: {total_elapsed:.1f}s")
        print(f"{'='*60}")

    if output_dir is not None:
        result.to_json(str(output_dir / "sample_result.json"))

    return result
