"""Parallel ceiling sweep with multi-resolution refinement.

Searches for the minimum energy barrier between two crystal structures
by sweeping A* searches across energy ceilings in parallel, then
refining with progressively finer discretization.
"""

import time
from pathlib import Path as PathlibPath

from cnf.calculation.grace import GraceCalculator
from cnf.calculation.relaxation import relax_unit_cell
from cnf.navigation.astar.models import (
    PathContext, Path, Attempt, SearchParameters, SearchResult, CeilingSweepResult
)
from cnf.navigation.endpoints import get_endpoint_cnfs

from ._params import calibrate_max_iters
from ._batch import run_batch
from ._workers import init_search_worker


def sweep(
    start_uc,
    end_uc,
    max_ceiling: float,
    energy_calc=None,
    xi=1.5,
    delta=10,
    num_ceilings=5,
    attempts_per_ceiling=1,
    max_passes=5,
    xi_factor=0.8,
    delta_factor=1.2,
    dropout=0.1,
    min_dropout=0.0,
    beam_width=1000,
    n_workers=0,
    verbose=True,
    output_dir=None,
    relax_endpoints=False,
) -> CeilingSweepResult:
    """Parallel ceiling sweep with multi-resolution refinement.

    Searches for the minimum energy barrier between two crystal structures
    by sweeping A* searches across energy ceilings in parallel, then
    refining with progressively finer discretization.

    Args:
        start_uc: Starting UnitCell.
        end_uc: Ending UnitCell.
        max_ceiling: Upper bound of sweep range (eV). Required.
        energy_calc: Energy calculator (default: GraceCalculator).
        xi: Initial lattice discretization parameter.
        delta: Initial motif discretization parameter.
        num_ceilings: Number of ceiling levels to sweep per pass.
        attempts_per_ceiling: Searches per ceiling level.
        max_passes: Number of resolution refinement passes.
        xi_factor: xi multiplied by this each refinement pass.
        delta_factor: delta multiplied by this each refinement pass.
        dropout: Neighbor dropout probability.
        min_dropout: Minimum dropout for adaptive adjustment.
        beam_width: Max open-set size for beam search.
        n_workers: Parallel worker processes. 0 = auto.
        verbose: Print progress.
        output_dir: Path to output directory.
        relax_endpoints: If True, relax endpoint structures before discretization.

    Returns:
        CeilingSweepResult containing all passes and their attempts.
    """
    if energy_calc is None:
        energy_calc = GraceCalculator()

    if relax_endpoints:
        ase_calc = energy_calc._calc

        if verbose:
            print("Relaxing endpoints in continuous space...")

        start_uc = relax_unit_cell(
            start_uc, ase_calc, verbose=verbose, label="start")
        end_uc = relax_unit_cell(
            end_uc, ase_calc, verbose=verbose, label="end")

        if output_dir is not None:
            out = PathlibPath(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            start_uc.to_cif(str(out / "start_relaxed.cif"))
            end_uc.to_cif(str(out / "end_relaxed.cif"))
            if verbose:
                print(f"  Relaxed CIFs saved to {out}")

    energy_cache = {}
    total_start = time.perf_counter()

    ceiling_top = max_ceiling

    result = CeilingSweepResult(
        results=[],
        metadata={
            "max_ceiling": max_ceiling,
            "xi": xi,
            "delta": delta,
            "num_ceilings": num_ceilings,
            "attempts_per_ceiling": attempts_per_ceiling,
            "max_passes": max_passes,
            "xi_factor": xi_factor,
            "delta_factor": delta_factor,
            "relax_endpoints": relax_endpoints,
        }
    )

    if output_dir is not None:
        output_dir = PathlibPath(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    def _save_result():
        if output_dir is not None:
            result.to_json(str(output_dir / "ceiling_sweep_result.json"))

    import multiprocessing
    total_cores = multiprocessing.cpu_count()
    _TF_THREADS_PER_WORKER = 4

    if n_workers == 0:
        n_workers = max(1, (total_cores - 1) // _TF_THREADS_PER_WORKER)
        if verbose:
            print(f"Auto workers: {n_workers} "
                  f"({total_cores} cores, {_TF_THREADS_PER_WORKER} threads/worker)")

    pool = None
    if n_workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        tf_threads = max(1, total_cores // n_workers)
        pool = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_search_worker,
            initargs=(tf_threads,),
            mp_context=mp.get_context('spawn'),
        )
        if verbose:
            print(f"Worker pool: {n_workers} workers, "
                  f"{tf_threads} TF threads each")

    if verbose:
        print(f"\nStarting ceiling sweep with max_ceiling={max_ceiling:.4f} eV")

    try:
        for pass_num in range(max_passes):
            pass_xi = xi * (xi_factor ** pass_num)
            pass_delta = round(delta * (delta_factor ** pass_num))

            energy_cache.clear()

            start_cnfs, goal_cnfs = get_endpoint_cnfs(
                start_uc, end_uc, xi=pass_xi, delta=pass_delta
            )
            elements = start_cnfs[0].elements

            context = PathContext(xi=pass_xi, delta=pass_delta, elements=elements)

            if pass_num == 0:
                result.metadata["elements"] = list(elements)
                result.metadata["start_cnf_coords"] = [list(c.coords) for c in start_cnfs]
                result.metadata["goal_cnf_coords"] = [list(c.coords) for c in goal_cnfs]

            for cnf in start_cnfs + goal_cnfs:
                if cnf.coords not in energy_cache:
                    energy_cache[cnf.coords] = energy_calc.calculate_energy(cnf)

            endpoint_energies = [
                energy_cache[c.coords] for c in start_cnfs + goal_cnfs
            ]
            base = max(endpoint_energies) + 1e-6

            if verbose:
                print(f"\n{'='*60}")
                print(f"Pass {pass_num} (xi={pass_xi:.2f}, delta={pass_delta})")
                print(f"  {len(start_cnfs)} start CNFs, "
                      f"{len(goal_cnfs)} goal CNFs")
                print(f"  Endpoint energies: "
                      f"{[f'{e:.2f}' for e in endpoint_energies]}")
                print(f"  Base ceiling: {base:.2f} eV")
                print(f"  Ceiling top: {ceiling_top:.2f} eV")
                print(f"{'='*60}")

            if base >= ceiling_top:
                if verbose:
                    print(f"\n  Base ({base:.2f}) >= ceiling_top "
                          f"({ceiling_top:.2f}), skipping pass")
                continue

            current_dropout = dropout
            current_max_iters = calibrate_max_iters(
                start_cnfs, goal_cnfs, beam_width, verbose,
            )

            N = num_ceilings
            if N > 1:
                ceilings = [
                    base + i * (ceiling_top - base) / (N - 1)
                    for i in range(N)
                ]
            else:
                ceilings = [ceiling_top]

            spacing = (ceiling_top - base) / max(N - 1, 1)
            if verbose:
                print(f"\n  {N} ceilings from {base:.2f} to "
                      f"{ceiling_top:.2f} eV "
                      f"(spacing={spacing:.2f} eV)")

            pass_start = time.perf_counter()
            batch_results = run_batch(
                ceilings, start_cnfs, goal_cnfs, elements,
                pass_xi, pass_delta, energy_calc, energy_cache,
                current_dropout, current_max_iters, beam_width,
                n_workers, pool, verbose,
                attempts_per_ceiling=attempts_per_ceiling,
                pass_id=pass_num,
            )
            pass_elapsed = time.perf_counter() - pass_start

            ceiling_to_attempts = {}
            for r in batch_results:
                ceil = r["ceiling"]
                if ceil not in ceiling_to_attempts:
                    ceiling_to_attempts[ceil] = []

                if r["found"]:
                    path_obj = Path(
                        coords=[tuple(pt) for pt in r["path"]],
                        energies=r["energies"],
                        barrier=r["barrier"],
                    )
                    attempt = Attempt(
                        path=path_obj,
                        found=True,
                        iterations=r["iterations"],
                    )
                else:
                    attempt = Attempt(
                        path=None,
                        found=False,
                        iterations=r["iterations"],
                    )
                ceiling_to_attempts[ceil].append(attempt)

            for ceil in sorted(ceiling_to_attempts.keys()):
                attempts = ceiling_to_attempts[ceil]
                search_params = SearchParameters(
                    max_iterations=current_max_iters,
                    beam_width=beam_width,
                    dropout=current_dropout,
                    greedy=False,
                    heuristic="manhattan",
                    filters=[{"type": "energy_ceiling", "value": ceil}],
                )
                search_result = SearchResult(
                    context=context,
                    parameters=search_params,
                    attempts=attempts,
                    metadata={
                        "pass": pass_num,
                        "ceiling": ceil,
                    }
                )
                result.results.append(search_result)

            successes = [r for r in batch_results if r["found"]]
            if successes:
                best_result = min(successes, key=lambda r: r["barrier"])
                ceiling_top = best_result["barrier"]
                if verbose:
                    print(f"\n  *** ceiling_top tightened to "
                          f"{ceiling_top:.2f} eV")

            result.metadata["current_ceiling_top"] = ceiling_top

            if verbose:
                print(f"\n  Pass {pass_num} done: "
                      f"{len(successes)}/{len(batch_results)} paths found, "
                      f"cache={len(energy_cache)} pts, "
                      f"elapsed={pass_elapsed:.1f}s")
                if result.best_barrier is not None:
                    print(f"  Best barrier: {result.best_barrier:.4f} eV")

            _save_result()

    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    total_elapsed = time.perf_counter() - total_start
    result.metadata["total_elapsed_seconds"] = total_elapsed
    result.metadata["final_ceiling_top"] = ceiling_top
    result.metadata["energy_cache_size"] = len(energy_cache)

    if verbose:
        best = result.best_path
        print(f"\n{'='*60}")
        print(f"Final result:")
        if best:
            print(f"  Barrier: {best.barrier:.4f} eV")
            print(f"  Path length: {len(best)} steps")
        else:
            print(f"  No path found")
        print(f"  Total SearchResults: {len(result.results)}")
        print(f"  Total paths found: {len(result.all_paths)}")
        print(f"  Energy cache size: {len(energy_cache)}")
        print(f"  Final ceiling top: {ceiling_top:.4f} eV")
        print(f"  Total time: {total_elapsed:.1f}s")
        print(f"{'='*60}")

    _save_result()

    return result
