"""Path sampling to discover initial energy ceiling.

Runs multiple A* searches with varied dropout to generate diverse paths,
evaluates energies, and returns the best barrier as a starting point
for ceiling sweep (Phase 3).
"""

import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path as PathlibPath

from cnf import CrystalNormalForm
from cnf.calculation.base_calculator import CalcProvider
from cnf.calculation.grace import GraceCalcProvider
from cnf.navigation.astar import astar_rust
from cnf.navigation.astar.models import (
    PathContext, Path, Attempt, SearchParameters, SearchResult
)

from ..core import evaluate_path_energies, path_barrier
from ..core import worker as core_worker


def _sample_worker(args):
    """Worker function for a single sampling attempt.

    Returns (attempt_idx, Attempt, cache_items) where cache_items
    is a list of (tuple_key, energy) pairs for merging.
    """
    (attempt_idx, attempt_dropout, direction,
     start_coord_lists, goal_coord_lists,
     elements, xi, delta,
     min_distance, max_iterations, beam_width, verbosity, log_prefix) = args

    # Reconstruct CNFs from coord lists
    start_cnfs = [CrystalNormalForm.from_tuple(tuple(c), elements, xi, delta)
                  for c in start_coord_lists]
    goal_cnfs = [CrystalNormalForm.from_tuple(tuple(c), elements, xi, delta)
                 for c in goal_coord_lists]

    if direction == "backward":
        attempt_starts, attempt_goals = goal_cnfs, start_cnfs
    else:
        attempt_starts, attempt_goals = start_cnfs, goal_cnfs

    attempt_start = time.perf_counter()

    path_tuples, num_iters = astar_rust(
        attempt_starts,
        attempt_goals,
        min_distance=min_distance or 0.0,
        max_iterations=max_iterations,
        beam_width=beam_width,
        dropout=attempt_dropout,
        verbose=(verbosity >= 2),
        log_prefix=log_prefix,
    )

    attempt_elapsed = time.perf_counter() - attempt_start

    if path_tuples is None:
        return (attempt_idx, Attempt(
            path=None,
            found=False,
            iterations=num_iters,
            elapsed_seconds=attempt_elapsed,
            metadata={"dropout": attempt_dropout, "direction": direction},
        ), [])

    if direction == "backward":
        path_tuples = list(reversed(path_tuples))

    # Evaluate energies with a local cache
    local_cache = {}
    energies = evaluate_path_energies(
        path_tuples, elements, xi, delta,
        core_worker.worker_calc, local_cache, verbose=False,
    )
    barrier = path_barrier(energies)

    path_obj = Path(
        coords=[tuple(pt) for pt in path_tuples],
        energies=energies,
        barrier=barrier,
    )

    # Return cache entries for merging
    cache_items = list(local_cache.items())

    return (attempt_idx, Attempt(
        path=path_obj,
        found=True,
        iterations=num_iters,
        elapsed_seconds=attempt_elapsed,
        metadata={"dropout": attempt_dropout, "direction": direction},
    ), cache_items)


def sample(
    start_cnfs: list[CrystalNormalForm],
    goal_cnfs: list[CrystalNormalForm],
    calc_provider: CalcProvider | None = None,
    num_samples: int = 20,
    dropout_range: tuple[float, float] = (0.05, 0.1),
    min_distance: float | None = None,
    max_iterations: int = 5_000,
    beam_width: int = 1000,
    bidirectional: bool = False,
    n_workers: int = 1,
    verbosity: int = 1,
    output_dir: PathlibPath | str | None = None,
) -> SearchResult:
    """Sample diverse paths to discover initial energy ceiling.

    Runs multiple A* searches with varied dropout to generate path diversity.
    No energy ceiling filter is applied - this is used to discover what
    ceiling to use for Phase 3 (sweep).

    Args:
        start_cnfs: Starting CNF structures.
        goal_cnfs: Goal CNF structures.
        calc_provider: Factory for creating energy calculators. Each worker
            process calls this to create its own calculator instance.
            Default: GraceCalcProvider() (foundation model).
        num_samples: Number of pathfinding attempts.
        dropout_range: (min, max) dropout probability range. Each attempt
            uses a random dropout in this range for path diversity.
        min_distance: Optional minimum interatomic distance filter.
        max_iterations: Max A* iterations per attempt.
        beam_width: Max open-set size for beam search.
        bidirectional: If True, randomly swap start/goal for ~half the attempts
            to explore paths in both directions. Default False since min_distance
            from Phase 1 is calibrated for start→goal direction only.
        n_workers: Number of parallel workers. 0 = auto (CPU count),
            1 = serial execution. Default: 1.
        verbosity: 0=silent, 1=phase output, 2+=A* iteration progress.
        output_dir: Path to output directory. If set, writes sample_result.json
            after completion.

    Returns:
        SearchResult containing all attempts. Use result.best_barrier as
        the initial ceiling for sweep().
    """
    if calc_provider is None:
        calc_provider = GraceCalcProvider()

    # Resolve n_workers
    if n_workers == 0:
        n_workers = os.cpu_count() or 1
    use_parallel = n_workers > 1

    elements = start_cnfs[0].elements
    xi = start_cnfs[0].xi
    delta = start_cnfs[0].delta

    context = PathContext(xi=xi, delta=delta, elements=elements)

    # For serial execution or seeding, create a calculator instance
    if not use_parallel:
        energy_calc = calc_provider()

    energy_cache = {}

    # Evaluate endpoint energies and seed the cache (serial only - parallel
    # workers will compute these if needed)
    if not use_parallel:
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

    def _save_intermediate():
        if output_dir is None:
            return
        intermediate = SearchResult(
            context=context,
            parameters=base_params,
            attempts=attempts,
            metadata={
                "num_samples": num_samples,
                "dropout_range": list(dropout_range),
                "bidirectional": bidirectional,
                "n_workers": n_workers,
                "completed_attempts": len(attempts),
                "in_progress": True,
            }
        )
        outpath = str(output_dir / "sample_result.json")
        intermediate.to_json(outpath)
        print(f"  [Sample] Saved: {outpath} (intermediate, {len(attempts)} attempts)")

    if verbosity >= 1:
        mode = f"parallel ({n_workers} workers)" if use_parallel else "serial"
        print(f"\nPath sampling: {num_samples} attempts "
              f"(dropout {dropout_range[0]:.2f}-{dropout_range[1]:.2f}, {mode})")
        print(f"  max_iters: {max_iterations}")
        if min_distance is not None:
            print(f"  Min distance filter: {min_distance:.2f} Å")

    # Prepare work items - pre-generate dropouts and directions for reproducibility
    work_items = []
    for attempt_idx in range(num_samples):
        attempt_dropout = random.uniform(dropout_range[0], dropout_range[1])
        if bidirectional and random.random() < 0.5:
            direction = "backward"
        else:
            direction = "forward"
        work_items.append((attempt_idx, attempt_dropout, direction))

    # Convert CNFs to coord lists for pickling
    start_coord_lists = [list(cnf.coords) for cnf in start_cnfs]
    goal_coord_lists = [list(cnf.coords) for cnf in goal_cnfs]

    if use_parallel:
        # Parallel execution
        attempts = _sample_parallel(
            work_items=work_items,
            start_coord_lists=start_coord_lists,
            goal_coord_lists=goal_coord_lists,
            elements=elements,
            xi=xi,
            delta=delta,
            calc_provider=calc_provider,
            min_distance=min_distance,
            max_iterations=max_iterations,
            beam_width=beam_width,
            n_workers=n_workers,
            verbosity=verbosity,
            energy_cache=energy_cache,
            save_callback=_save_intermediate,
        )
    else:
        # Serial execution
        attempts = _sample_serial(
            work_items=work_items,
            start_cnfs=start_cnfs,
            goal_cnfs=goal_cnfs,
            energy_calc=energy_calc,
            energy_cache=energy_cache,
            elements=elements,
            xi=xi,
            delta=delta,
            min_distance=min_distance,
            max_iterations=max_iterations,
            beam_width=beam_width,
            verbosity=verbosity,
            save_callback=_save_intermediate,
            num_samples=num_samples,
        )

    total_elapsed = time.perf_counter() - total_start

    result = SearchResult(
        context=context,
        parameters=base_params,
        attempts=attempts,
        metadata={
            "num_samples": num_samples,
            "dropout_range": list(dropout_range),
            "bidirectional": bidirectional,
            "n_workers": n_workers,
            "total_elapsed_seconds": total_elapsed,
            "energy_cache_size": len(energy_cache),
        }
    )

    if verbosity >= 1:
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
        outpath = str(output_dir / "sample_result.json")
        result.to_json(outpath)
        print(f"  [Sample] Saved: {outpath}")

    return result


def _sample_serial(
    work_items,
    start_cnfs,
    goal_cnfs,
    energy_calc,
    energy_cache,
    elements,
    xi,
    delta,
    min_distance,
    max_iterations,
    beam_width,
    verbosity,
    save_callback,
    num_samples,
):
    """Run sampling attempts serially."""
    attempts = []

    for attempt_idx, attempt_dropout, direction in work_items:
        if direction == "backward":
            attempt_starts, attempt_goals = goal_cnfs, start_cnfs
        else:
            attempt_starts, attempt_goals = start_cnfs, goal_cnfs

        if verbosity >= 1:
            print(f"  [{attempt_idx+1}/{num_samples}] "
                  f"dropout={attempt_dropout:.2f} {direction}...", end=" ", flush=True)

        attempt_start = time.perf_counter()

        path_tuples, num_iters = astar_rust(
            attempt_starts,
            attempt_goals,
            min_distance=min_distance or 0.0,
            max_iterations=max_iterations,
            beam_width=beam_width,
            dropout=attempt_dropout,
            verbose=(verbosity >= 2),
        )

        attempt_elapsed = time.perf_counter() - attempt_start

        if path_tuples is None:
            if verbosity >= 1:
                print("no path")
            attempts.append(Attempt(
                path=None,
                found=False,
                iterations=num_iters,
                elapsed_seconds=attempt_elapsed,
                metadata={"dropout": attempt_dropout, "direction": direction},
            ))
            if len(attempts) % 10 == 0:
                save_callback()
            continue

        if direction == "backward":
            path_tuples = list(reversed(path_tuples))

        energies = evaluate_path_energies(
            path_tuples, elements, xi, delta, energy_calc, energy_cache,
            verbose=False,
        )
        barrier = path_barrier(energies)

        if verbosity >= 1:
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

        if len(attempts) % 10 == 0:
            save_callback()

    return attempts


def _sample_parallel(
    work_items,
    start_coord_lists,
    goal_coord_lists,
    elements,
    xi,
    delta,
    calc_provider,
    min_distance,
    max_iterations,
    beam_width,
    n_workers,
    verbosity,
    energy_cache,
    save_callback,
):
    """Run sampling attempts in parallel using ProcessPoolExecutor."""
    num_samples = len(work_items)

    # Build args for each worker task - include elements, xi, delta in each task
    worker_args = []
    for attempt_idx, attempt_dropout, direction in work_items:
        log_prefix = f"[{attempt_idx+1}/{num_samples}]" if verbosity >= 2 else ""
        worker_args.append((
            attempt_idx,
            attempt_dropout,
            direction,
            start_coord_lists,
            goal_coord_lists,
            elements,
            xi,
            delta,
            min_distance,
            max_iterations,
            beam_width,
            verbosity,
            log_prefix,
        ))

    # Results will be collected out of order, so we store by index
    results_by_idx = {}
    completed = 0

    # Compute TF threads per worker to avoid contention
    import multiprocessing
    total_cores = multiprocessing.cpu_count()
    tf_threads = max(1, total_cores // n_workers)

    # Use functools.partial to pass phase_name to init_worker
    from functools import partial
    init_fn = partial(core_worker.init_worker, phase_name="Sample")

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_fn,
        initargs=(calc_provider, tf_threads),
    ) as executor:
        futures = {executor.submit(_sample_worker, args): args[0] for args in worker_args}

        for future in as_completed(futures):
            attempt_idx = futures[future]
            try:
                idx, attempt, cache_items = future.result()
                results_by_idx[idx] = attempt

                # Merge cache entries from worker
                for key, energy in cache_items:
                    if key not in energy_cache:
                        energy_cache[key] = energy

                completed += 1
                if verbosity >= 1:
                    status = "no path" if not attempt.found else f"barrier={attempt.path.barrier:.4f} eV"
                    print(f"  [{completed}/{num_samples}] attempt {idx+1}: {status}")

                if completed % 10 == 0:
                    save_callback()

            except Exception as e:
                if verbosity >= 1:
                    print(f"  Worker error for attempt {attempt_idx}: {e}")
                results_by_idx[attempt_idx] = Attempt(
                    path=None,
                    found=False,
                    iterations=0,
                    elapsed_seconds=0.0,
                    metadata={"error": str(e)},
                )

    # Reconstruct attempts in original order
    attempts = [results_by_idx[i] for i in range(num_samples)]
    return attempts
