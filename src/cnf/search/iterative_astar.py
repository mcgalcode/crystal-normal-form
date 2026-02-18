"""Iterative A* barrier search with adaptive parameters.

Runs multiple A* searches with dropout for path diversity, evaluates energies
post-hoc, then uses the best barrier as an energy ceiling to prune future
searches. Iterates until the ceiling can't be lowered.

Round 0 (fast, no energy filter):
    Run N A* searches with dropout (Rust impl)
    Evaluate energy along all found paths
    Set ceiling = min barrier across paths

Round 1+ (with energy ceiling, adaptive dropout and max_iters):
    Run N A* searches with EnergyFilter(ceiling)
    Ceiling is fixed per round (all paths see the same constraint)
    After each round, adjust dropout and max_iters based on success rate
    Converge when no improvement and parameters are at their limits
"""

import json
import os
import time
from pathlib import Path

from cnf import CrystalNormalForm, UnitCell
from cnf.calculation.base_calculator import BaseCalculator
from cnf.calculation.grace import GraceCalculator
from cnf.navigation.astar import astar_rust
from cnf.navigation.astar.core import astar_pathfind
from cnf.navigation.astar.heuristics import manhattan_distance, UnimodularManhattanHeuristic
from cnf.navigation.endpoints import get_endpoint_cnfs
from cnf.navigation.search_filters import FilterSet, MinDistanceFilter, EnergyFilter

USE_RUST = os.getenv('USE_RUST') == '1'


def _evaluate_path_energies(path_tuples, elements, xi, delta, calc, cache):
    """Evaluate energies along a path, using and populating the shared cache."""
    energies = []
    for pt_tup in path_tuples:
        if pt_tup not in cache:
            cnf = CrystalNormalForm.from_tuple(pt_tup, elements, xi, delta)
            cache[pt_tup] = calc.calculate_energy(cnf)
        energies.append(cache[pt_tup])
    return energies


def _path_barrier(energies):
    """Barrier = max energy along path."""
    return max(energies)


def _make_heuristic(mode, weight):
    if mode == "manhattan":
        return manhattan_distance
    elif mode == "unimodular_light":
        return UnimodularManhattanHeuristic(weight=weight, full=False, partial=False)
    elif mode == "unimodular_partial":
        return UnimodularManhattanHeuristic(weight=weight, full=False, partial=True)
    elif mode == "unimodular_full":
        return UnimodularManhattanHeuristic(weight=weight, full=True, partial=False)
    else:
        raise ValueError(f"Unknown heuristic mode: {mode}")


def _write_round_json(output_dir, round_num, round_data):
    """Write a single round's results to rounds/round_NNN.json."""
    round_path = Path(output_dir) / "rounds" / f"round_{round_num:03d}.json"
    with open(round_path, "w") as f:
        json.dump(round_data, f, indent=2)


def _write_energy_cache(output_dir, cache):
    """Write energy cache as {string_key: energy} JSON."""
    serializable = {str(k): v for k, v in cache.items()}
    cache_path = Path(output_dir) / "energy_cache.json"
    with open(cache_path, "w") as f:
        json.dump(serializable, f, indent=2)


def _write_manifest(output_dir, params, result, timing):
    """Write manifest.json with parameters, results, and timing."""
    manifest_path = Path(output_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"parameters": params, "result": result, "timing": timing}, f, indent=2)


def iterative_astar_barrier(
    start_cnfs,
    goal_cnfs,
    energy_calc=None,
    paths_per_round=10,
    max_rounds=20,
    dropout=0.3,
    min_dropout=0.1,
    max_iterations_per_path=100_000,
    min_distance=0.0,
    beam_width=1000,
    heuristic_mode="manhattan",
    heuristic_weight=0.5,
    verbose=True,
    output_dir=None,
    initial_ceiling=None,
):
    """Iterative A* barrier search with adaptive parameters.

    Args:
        start_cnfs: Starting CNF structures.
        goal_cnfs: Goal CNF structures.
        energy_calc: Energy calculator (default: GraceCalculator).
        paths_per_round: Number of A* runs per round.
        max_rounds: Maximum tightening iterations.
        dropout: Neighbor dropout probability (used for round 0 and as
            starting value for energy rounds).
        min_dropout: Minimum dropout for adaptive adjustment (default 0.1).
        max_iterations_per_path: Max A* iterations (used for round 0 and
            as the absolute cap for energy rounds).
        min_distance: Minimum allowed pairwise atomic distance.
        beam_width: Max open-set size for beam search.
        heuristic_mode: Heuristic for A* ("manhattan", "unimodular_light", etc.).
        heuristic_weight: Weight for unimodular heuristics.
        verbose: Print progress.
        output_dir: Path to output directory (None = no file output).
        initial_ceiling: Starting energy ceiling (eV). If provided, skips
            Round 0 and starts directly with energy-filtered rounds.

    Returns:
        (barrier, best_path_cnfs, best_path_energies) or (None, None, None)
        if no path is found.
    """
    if energy_calc is None:
        energy_calc = GraceCalculator()

    elements = start_cnfs[0].elements
    xi = start_cnfs[0].xi
    delta = start_cnfs[0].delta

    energy_cache = {}  # shared across all rounds

    # Evaluate endpoint energies and seed the cache
    for cnf in start_cnfs + goal_cnfs:
        if cnf.coords not in energy_cache:
            energy_cache[cnf.coords] = energy_calc.calculate_energy(cnf)

    # Set up output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "rounds").mkdir(exist_ok=True)

    ceiling = initial_ceiling
    best_path = None
    best_energies = None
    total_paths_found = 0
    round_times = []
    total_start = time.perf_counter()
    start_round = 1 if initial_ceiling is not None else 0

    if initial_ceiling is not None and verbose:
        print(f"\nSkipping Round 0, using initial ceiling={initial_ceiling:.4f} eV")

    # Adaptive parameters for energy rounds
    current_dropout = dropout
    current_max_iters = min(max_iterations_per_path, 500)
    prev_round_successful_iters = []

    for round_num in range(start_round, max_rounds):
        round_start = time.perf_counter()
        ceiling_at_start = ceiling
        round_ceiling = ceiling  # fixed for this round
        paths_this_round = 0
        round_paths = []
        improved = False
        round_successful_iters = []

        if verbose:
            print(f"\n{'='*60}")
            if round_num == 0:
                print(f"Round {round_num} (Rust A*, no energy filter)")
            else:
                print(f"Round {round_num} (Python A*, ceiling={round_ceiling:.4f} eV, "
                      f"dropout={current_dropout:.2f}, max_iters={current_max_iters})")
            print(f"{'='*60}")

        round_paths_per_round = max(100, paths_per_round) if round_num == 0 else paths_per_round
        for path_idx in range(round_paths_per_round):
            if verbose:
                print(f"  Path {path_idx+1}/{round_paths_per_round}...", end=" ", flush=True)

            path_tuples = None

            if round_num == 0:
                # Round 0: fast Rust A*, no energy filter
                result, num_iters = astar_rust(
                    start_cnfs,
                    goal_cnfs,
                    min_distance=min_distance,
                    max_iterations=max_iterations_per_path,
                    beam_width=beam_width,
                    dropout=dropout,
                    verbose=False,
                    heuristic_mode=heuristic_mode,
                    heuristic_weight=heuristic_weight,
                )
                if result is not None:
                    path_tuples = [tuple(pt) for pt in result]
            else:
                # Round 1+: Python A* with energy filter (fixed ceiling per round)
                heuristic = _make_heuristic(heuristic_mode, heuristic_weight)
                energy_filter = EnergyFilter(
                    round_ceiling, calc=energy_calc, cache=energy_cache
                )
                filters = [energy_filter]
                filter_set = FilterSet(filters, use_structs=not USE_RUST)

                search_state = astar_pathfind(
                    start_cnfs,
                    goal_cnfs,
                    heuristic=heuristic,
                    filter_set=filter_set,
                    max_iterations=current_max_iters,
                    beam_width=beam_width,
                    dropout=current_dropout,
                    verbose=False,
                )
                if search_state.path is not None:
                    path_tuples = search_state.path
                num_iters = search_state.iterations

            if path_tuples is None:
                if verbose:
                    print("no path found")
                continue

            paths_this_round += 1
            total_paths_found += 1
            round_successful_iters.append(num_iters)

            # Evaluate energies along path
            energies = _evaluate_path_energies(
                path_tuples, elements, xi, delta, energy_calc, energy_cache
            )
            barrier = _path_barrier(energies)

            if verbose:
                print(f"len={len(path_tuples)}, barrier={barrier:.4f} eV, iters={num_iters}")

            round_paths.append({
                "path": [list(pt) for pt in path_tuples],
                "energies": energies,
                "barrier": barrier,
                "length": len(path_tuples),
                "iterations": num_iters,
            })

            # Track best across all rounds (but don't update round_ceiling)
            if ceiling is None or barrier < ceiling:
                ceiling = barrier
                best_path = path_tuples
                best_energies = energies
                improved = True

        round_elapsed = time.perf_counter() - round_start
        round_times.append(round_elapsed)

        success_rate = paths_this_round / round_paths_per_round if round_paths_per_round > 0 else 0

        if verbose:
            cache_size = len(energy_cache)
            print(f"  Round {round_num} summary: {paths_this_round}/{round_paths_per_round} paths found "
                  f"({success_rate:.0%}), ceiling={ceiling:.4f} eV, cache={cache_size} pts, "
                  f"elapsed={round_elapsed:.1f}s")

        if output_dir is not None:
            _write_round_json(output_dir, round_num, {
                "round": round_num,
                "ceiling_at_start": ceiling_at_start,
                "ceiling_at_end": ceiling,
                "improved": improved,
                "implementation": "rust" if round_num == 0 else "python",
                "dropout": dropout if round_num == 0 else current_dropout,
                "max_iters": max_iterations_per_path if round_num == 0 else current_max_iters,
                "success_rate": success_rate,
                "paths": round_paths,
            })

        if ceiling is None:
            # No paths found at all yet — keep trying
            if verbose:
                print("  No paths found yet, continuing...")
            continue

        # Adaptive parameter adjustment (after energy rounds)
        if round_num >= 1:
            # Adjust dropout based on success rate
            if success_rate >= 0.67:
                pass  # easy — keep current dropout
            elif success_rate >= 0.33:
                current_dropout = max(current_dropout - 0.1, min_dropout)
            else:
                current_dropout = max(current_dropout * 0.5, min_dropout)

            # Adjust max_iters based on previous round's iteration counts
            if round_successful_iters:
                current_max_iters = min(
                    int(1.5 * max(round_successful_iters)),
                    max_iterations_per_path,
                )
            else:
                # No paths found — give more room
                current_max_iters = min(current_max_iters * 2, max_iterations_per_path)

        # Convergence: only if no improvement AND parameters at limits
        if round_num > 0 and not improved:
            at_limits = (current_dropout <= min_dropout and
                         current_max_iters >= max_iterations_per_path)
            if at_limits:
                if verbose:
                    print(f"\n  Converged! No improvement with parameters at limits.")
                break
            else:
                if verbose:
                    print(f"  No improvement — adapting parameters "
                          f"(dropout={current_dropout:.2f}, max_iters={current_max_iters})")

    total_elapsed = time.perf_counter() - total_start
    converged = round_num > 0 and not improved

    if best_path is None:
        if verbose:
            print("\nNo path found across all rounds.")
        if output_dir is not None:
            _write_manifest(output_dir,
                params={
                    "xi": xi, "delta": delta, "elements": elements,
                    "paths_per_round": paths_per_round, "max_rounds": max_rounds,
                    "dropout": dropout, "min_dropout": min_dropout,
                    "max_iterations_per_path": max_iterations_per_path,
                    "min_distance": min_distance, "beam_width": beam_width,
                    "heuristic_mode": heuristic_mode, "heuristic_weight": heuristic_weight,
                    "start_cnf_coords": [list(c.coords) for c in start_cnfs],
                    "goal_cnf_coords": [list(c.coords) for c in goal_cnfs],
                },
                result={"barrier": None, "best_path": None, "best_path_energies": None,
                        "path_length": 0, "total_paths_found": 0,
                        "total_rounds": round_num + 1, "converged": False,
                        "energy_cache_size": len(energy_cache)},
                timing={"total_seconds": total_elapsed, "round_seconds": round_times},
            )
            _write_energy_cache(output_dir, energy_cache)
        return None, None, None

    # Convert best path tuples to CNFs
    best_path_cnfs = [
        CrystalNormalForm.from_tuple(pt, elements, xi, delta)
        for pt in best_path
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Final result:")
        print(f"  Barrier: {ceiling:.4f} eV")
        print(f"  Path length: {len(best_path)} steps")
        print(f"  Total paths found: {total_paths_found}")
        print(f"  Energy cache size: {len(energy_cache)}")
        print(f"{'='*60}")

    if output_dir is not None:
        _write_manifest(output_dir,
            params={
                "xi": xi, "delta": delta, "elements": elements,
                "paths_per_round": paths_per_round, "max_rounds": max_rounds,
                "dropout": dropout, "min_dropout": min_dropout,
                "max_iterations_per_path": max_iterations_per_path,
                "min_distance": min_distance, "beam_width": beam_width,
                "heuristic_mode": heuristic_mode, "heuristic_weight": heuristic_weight,
                "start_cnf_coords": [list(c.coords) for c in start_cnfs],
                "goal_cnf_coords": [list(c.coords) for c in goal_cnfs],
            },
            result={
                "barrier": ceiling,
                "best_path": [list(pt) for pt in best_path],
                "best_path_energies": best_energies,
                "path_length": len(best_path),
                "total_paths_found": total_paths_found,
                "total_rounds": round_num + 1,
                "converged": converged,
                "energy_cache_size": len(energy_cache),
            },
            timing={"total_seconds": total_elapsed, "round_seconds": round_times},
        )
        _write_energy_cache(output_dir, energy_cache)

    return ceiling, best_path_cnfs, best_energies


# ── Single-search function (usable in-process or in worker) ──

def _search_at_ceiling(ceiling, start_cnfs, goal_cnfs, elements, xi, delta,
                       calc, cache, dropout, max_iters, beam_width,
                       heuristic_mode, heuristic_weight, log_prefix="      "):
    """Run a single A* search at the given energy ceiling.

    Returns dict with keys: ceiling, found, iterations, and if found:
    barrier, path, energies, path_length.
    """
    heuristic = _make_heuristic(heuristic_mode, heuristic_weight)
    energy_filter = EnergyFilter(ceiling, calc=calc, cache=cache)
    filter_set = FilterSet([energy_filter], use_structs=not USE_RUST)

    cache_before = len(cache)
    speak_freq = max(1, max_iters // 10)
    search_state = astar_pathfind(
        start_cnfs, goal_cnfs,
        heuristic=heuristic, filter_set=filter_set,
        max_iterations=max_iters, beam_width=beam_width,
        dropout=dropout, verbose=True, speak_freq=speak_freq,
        log_prefix=log_prefix,
    )
    energy_evals = len(cache) - cache_before

    if search_state.path is None:
        return {"ceiling": ceiling, "found": False,
                "iterations": search_state.iterations,
                "energy_evals": energy_evals}

    path_tuples = search_state.path
    energies = _evaluate_path_energies(
        path_tuples, elements, xi, delta, calc, cache
    )
    energy_evals = len(cache) - cache_before  # include path evaluation
    barrier = _path_barrier(energies)

    return {
        "ceiling": ceiling, "found": True,
        "barrier": barrier, "path": path_tuples, "energies": energies,
        "iterations": search_state.iterations,
        "path_length": len(path_tuples),
        "energy_evals": energy_evals,
    }


# ── Worker pool support for parallel searches ──

_worker_calc = None
_worker_cache = None
_worker_pass_id = None


def _init_search_worker(tf_threads=None):
    """Initialize a worker process with its own energy calculator and cache.

    Args:
        tf_threads: Max TF threads per worker. Prevents CPU contention
            when multiple workers run in parallel.
    """
    if tf_threads is not None:
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(tf_threads)
        tf.config.threading.set_intra_op_parallelism_threads(tf_threads)
    global _worker_calc, _worker_cache, _worker_pass_id
    _worker_calc = GraceCalculator()
    _worker_cache = {}
    _worker_pass_id = None



# ── Batch execution ──

def _search_ceiling_with_attempts(ceiling, start_cnfs, goal_cnfs, elements,
                                  xi, delta, calc, cache, dropout, max_iters,
                                  beam_width, heuristic_mode, heuristic_weight,
                                  attempts, verbose):
    """Run up to `attempts` A* searches at a single ceiling, stop on first success.
    Bumps max_iters by 1.5x after each failed attempt."""
    current_max_iters = max_iters
    for a in range(attempts):
        if verbose:
            label = (f"(ceiling={ceiling:.2f} eV, max_iters={current_max_iters})"
                     if attempts == 1
                     else f"(ceiling={ceiling:.2f} eV, attempt {a+1}/{attempts}, max_iters={current_max_iters})")
            print(f"    Search {label}...", end=" ", flush=True)
        t0 = time.perf_counter()
        r = _search_at_ceiling(
            ceiling, start_cnfs, goal_cnfs, elements, xi, delta,
            calc, cache, dropout, current_max_iters, beam_width,
            heuristic_mode, heuristic_weight,
        )
        elapsed = time.perf_counter() - t0
        if verbose:
            evals = r.get('energy_evals', '?')
            if r["found"]:
                print(f"path found! len={r['path_length']}, "
                      f"barrier={r['barrier']:.2f} eV, "
                      f"iters={r['iterations']}, evals={evals}, "
                      f"{elapsed:.1f}s")
            else:
                print(f"no path ({r['iterations']} iters, {evals} evals, "
                      f"{elapsed:.1f}s)")
        if r["found"]:
            return r
        # Bump max_iters for next attempt
        current_max_iters = int(current_max_iters * 1.5)
    return r  # last attempt's result (not found)

MAX_ITERS_SCALE = 1.0

def _worker_search_with_attempts(args):
    """Worker wrapper: run multiple attempts at a single ceiling."""
    global _worker_calc, _worker_cache, _worker_pass_id
    (ceiling, start_coord_lists, goal_coord_lists, elements, xi, delta,
     dropout, max_iters, beam_width, heuristic_mode, heuristic_weight,
     seed_cache_items, attempts, worker_label, pass_id) = args

    # Clear cache when xi/delta changes between passes — same integer
    # coords map to different physical structures at different resolutions
    if pass_id != _worker_pass_id:
        _worker_cache = {}
        _worker_pass_id = pass_id

    for k, v in seed_cache_items:
        if k not in _worker_cache:
            _worker_cache[k] = v

    start_cnfs = [CrystalNormalForm.from_tuple(tuple(c), elements, xi, delta)
                  for c in start_coord_lists]
    goal_cnfs = [CrystalNormalForm.from_tuple(tuple(c), elements, xi, delta)
                 for c in goal_coord_lists]

    current_max_iters = max_iters
    for a in range(attempts):
        label = f"{worker_label} #{a+1}" if attempts > 1 else worker_label
        print(f"    [{label}] starting (ceiling={ceiling:.2f} eV, max_iters={current_max_iters})...",
              flush=True)

        t0 = time.perf_counter()
        r = _search_at_ceiling(
            ceiling, start_cnfs, goal_cnfs, elements, xi, delta,
            _worker_calc, _worker_cache,
            dropout, current_max_iters, beam_width, heuristic_mode, heuristic_weight,
            log_prefix=f"    [{label}] ",
        )
        elapsed = time.perf_counter() - t0

        evals = r.get('energy_evals', '?')
        if r["found"]:
            print(f"    [{label}] path found! len={r['path_length']}, "
                  f"barrier={r['barrier']:.2f} eV, "
                  f"iters={r['iterations']}, evals={evals}, {elapsed:.1f}s",
                  flush=True)
            return r
        else:
            print(f"    [{label}] no path ({r['iterations']} iters, "
                  f"{evals} evals, {elapsed:.1f}s)",
                  flush=True)
            # Bump max_iters for next attempt
            current_max_iters = int(current_max_iters * MAX_ITERS_SCALE)

    return r  # last attempt's result (not found)


def _run_batch(ceilings, start_cnfs, goal_cnfs, elements, xi, delta,
               calc, cache, dropout, max_iters, beam_width,
               heuristic_mode, heuristic_weight, n_workers, pool, verbose,
               attempts_per_ceiling=1, pass_id=0):
    """Run A* searches across ceilings, with multiple attempts per ceiling.

    Each ceiling gets up to `attempts_per_ceiling` searches (stopping on
    first success). Parallelism is across ceilings (one worker per ceiling),
    not across attempts within a ceiling.

    Across ceilings (ordered low→high), cancels higher ceilings once a path
    is found at a lower one.

    Returns list of result dicts (one per ceiling that was searched).
    """
    if n_workers > 1 and pool is not None:
        from concurrent.futures import as_completed

        seed_cache_items = list(cache.items())
        start_coords = [list(c.coords) for c in start_cnfs]
        goal_coords = [list(c.coords) for c in goal_cnfs]

        # One task per ceiling — each task runs attempts_per_ceiling internally
        futures = {}
        for i, c in enumerate(ceilings):
            args = (c, start_coords, goal_coords, elements, xi, delta,
                    dropout, max_iters, beam_width, heuristic_mode,
                    heuristic_weight, seed_cache_items,
                    attempts_per_ceiling, f"c={c:.2f}", pass_id)
            f = pool.submit(_worker_search_with_attempts, args)
            futures[f] = c

        results = []
        best_found_ceiling = None
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            if r["found"]:
                if best_found_ceiling is None or r["ceiling"] < best_found_ceiling:
                    best_found_ceiling = r["ceiling"]
                # Cancel pending futures at higher ceilings
                for other_f, other_c in list(futures.items()):
                    if other_c > best_found_ceiling and not other_f.done():
                        cancelled = other_f.cancel()
                        if cancelled and verbose:
                            print(f"    [c={other_c:.2f}] cancelled "
                                  f"(path found at {best_found_ceiling:.2f})",
                                  flush=True)

        results.sort(key=lambda r: r["ceiling"])
    else:
        results = []
        for i, c in enumerate(ceilings):
            r = _search_ceiling_with_attempts(
                c, start_cnfs, goal_cnfs, elements, xi, delta,
                calc, cache, dropout, max_iters, beam_width,
                heuristic_mode, heuristic_weight,
                attempts_per_ceiling, verbose,
            )
            results.append(r)
            if r["found"]:
                if verbose and i < len(ceilings) - 1:
                    print(f"    Skipping {len(ceilings) - i - 1} "
                          f"higher ceilings")
                break

    return results


def _serialize_result(r):
    """Make a result dict JSON-serializable."""
    s = {"ceiling": r["ceiling"], "found": r["found"],
         "iterations": r["iterations"]}
    if r["found"]:
        s["barrier"] = r["barrier"]
        s["path_length"] = r["path_length"]
        s["path"] = [list(pt) for pt in r["path"]]
        s["energies"] = r["energies"]
    return s


def _adapt_params(found, total, successful_iters,
                  current_dropout, min_dropout, current_max_iters,
                  max_iters=float('inf')):
    """Adapt dropout and max_iters based on batch success rate."""
    rate = found / total if total > 0 else 0

    if rate >= 0.67:
        new_dropout = current_dropout
    elif rate >= 0.33:
        new_dropout = max(current_dropout - 0.1, min_dropout)
    else:
        new_dropout = max(current_dropout * 0.5, min_dropout)

    if successful_iters:
        new_max_iters = min(int(1.5 * max(successful_iters)), max_iters)
    else:
        new_max_iters = min(current_max_iters * 2, max_iters)

    return new_dropout, new_max_iters


# ── Diagnostic calibration ──

_DIAG_RUNS = 3
_DIAG_DROPOUT = 0.6
_DIAG_SAFETY_FACTOR = 2
_DIAG_MAX_ATTEMPTS = 4  # doubling max_iters each attempt
_DIAG_MIN_DIST_RATIO = 0.75  # fraction of shortest bond length


def _min_bond_length(cnfs):
    """Compute minimum pairwise atomic distance across endpoint structures."""
    min_dist = float('inf')
    for cnf in cnfs:
        struct = cnf.reconstruct()
        for i in range(len(struct)):
            for j in range(i + 1, len(struct)):
                d = struct.get_distance(i, j)
                if d < min_dist:
                    min_dist = d
    return min_dist


def _calibrate_max_iters(start_cnfs, goal_cnfs, beam_width,
                         heuristic_mode, heuristic_weight, verbose):
    """Run diagnostic Rust A* searches to calibrate max_iters for this resolution.

    Uses a min-distance filter based on 75% of the shortest bond length in
    the endpoint structures, so diagnostic paths avoid unphysical regions
    where atoms overlap.

    Runs _DIAG_RUNS fast Rust A* searches (high dropout) per attempt.
    If no paths found, doubles max_iters and retries until at least one
    path is found. Returns calibrated max_iters (average * safety factor).
    """
    # Compute min-distance filter from endpoint bond lengths
    shortest_bond = _min_bond_length(start_cnfs + goal_cnfs)
    diag_min_dist = shortest_bond * _DIAG_MIN_DIST_RATIO

    if verbose:
        print(f"\n  Shortest endpoint bond: {shortest_bond:.3f} A, "
              f"diagnostic min_dist={diag_min_dist:.3f} A", flush=True)

    trial_max_iters = 500
    attempt = 0

    while True:
        successful_iters = []
        attempt += 1

        if verbose:
            print(f"  Diagnostic (attempt {attempt}, "
                  f"max_iters={trial_max_iters}, "
                  f"dropout={_DIAG_DROPOUT}):", flush=True)

        for i in range(_DIAG_RUNS):
            result, num_iters = astar_rust(
                start_cnfs, goal_cnfs,
                min_distance=diag_min_dist,
                max_iterations=trial_max_iters,
                beam_width=beam_width,
                dropout=_DIAG_DROPOUT,
                verbose=False,
                heuristic_mode=heuristic_mode,
                heuristic_weight=heuristic_weight,
            )
            if result is not None:
                successful_iters.append(num_iters)
                if verbose:
                    print(f"    Run {i+1}/{_DIAG_RUNS}: "
                          f"path found, {num_iters} iters", flush=True)
            else:
                if verbose:
                    print(f"    Run {i+1}/{_DIAG_RUNS}: "
                          f"no path ({trial_max_iters} iters)", flush=True)

        if successful_iters:
            avg_iters = sum(successful_iters) / len(successful_iters)
            calibrated = int(avg_iters * _DIAG_SAFETY_FACTOR)
            # Ensure at least as large as the max observed
            calibrated = max(calibrated, max(successful_iters))
            if verbose:
                print(f"  Calibrated max_iters={calibrated} "
                      f"(avg={avg_iters:.0f}, "
                      f"max={max(successful_iters)}, "
                      f"safety={_DIAG_SAFETY_FACTOR}x)", flush=True)
            return calibrated

        # No paths found — double and retry
        trial_max_iters *= 2
        if verbose:
            print(f"  No diagnostic paths found, "
                  f"doubling to {trial_max_iters}...", flush=True)


# ── Main function ──

def _relax_unit_cell(uc, calc, fmax=0.01, max_steps=500, verbose=True, label=""):
    """Relax a UnitCell in continuous space (cell + positions) using ASE FIRE.

    Args:
        uc: UnitCell to relax.
        calc: ASE calculator (e.g. from grace_fm()).
        fmax: Force convergence threshold (eV/A).
        max_steps: Maximum optimizer steps.
        verbose: Print before/after energies.
        label: Label for printing (e.g. "start" or "end").

    Returns:
        Relaxed UnitCell.
    """
    from ase.filters import ExpCellFilter
    from ase.optimize import FIRE
    from pymatgen.io.ase import AseAtomsAdaptor

    struct = uc.to_pymatgen_structure()
    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms.calc = calc

    e_before = atoms.get_potential_energy()
    if verbose:
        print(f"  [{label}] Before: E = {e_before:.4f} eV, "
              f"vol = {atoms.get_volume():.2f} A^3")

    ecf = ExpCellFilter(atoms)
    opt = FIRE(ecf, logfile=None)
    opt.run(fmax=fmax, steps=max_steps)

    e_after = atoms.get_potential_energy()
    if verbose:
        print(f"  [{label}] After:  E = {e_after:.4f} eV, "
              f"vol = {atoms.get_volume():.2f} A^3 "
              f"({opt.nsteps} steps)")

    relaxed_struct = AseAtomsAdaptor.get_structure(atoms)
    return UnitCell.from_pymatgen_structure(relaxed_struct)


def ceiling_barrier_search(
    start_uc,
    end_uc,
    energy_calc=None,
    # Discretization (initial values, refined each pass)
    xi=1.5,
    delta=10,
    # Search parameters
    step_per_atom=0.5,
    num_ceilings=5,
    attempts_per_ceiling=1,
    max_passes=5,
    max_sweep_rounds=20,
    max_ceiling=None,
    xi_factor=0.8,
    delta_factor=1.2,
    # A* parameters
    dropout=0.1,
    min_dropout=0.0,
    beam_width=1000,
    heuristic_mode="manhattan",
    heuristic_weight=0.5,
    # Parallelism
    n_workers=0,
    verbose=True,
    output_dir=None,
    # Endpoint relaxation
    relax_endpoints=False,
):
    """Ceiling barrier search with multi-resolution refinement.

    Searches for the minimum energy barrier between two crystal structures
    by sweeping A* searches across energy ceilings in parallel, then
    refining with progressively finer discretization.

    The sweep range is [base, ceiling_top] where base = max(endpoint_energies).
    ceiling_top is either given via max_ceiling, or discovered by sweeping
    upward from base with step_per_atom spacing until a path is found.

    Each pass refines xi/delta and re-sweeps, tightening ceiling_top to
    the lowest barrier found.

    Args:
        start_uc: Starting UnitCell.
        end_uc: Ending UnitCell.
        energy_calc: Energy calculator (default: GraceCalculator).
        xi: Initial lattice discretization parameter.
        delta: Initial motif discretization parameter.
        step_per_atom: Energy step per atom (eV/atom) for discovering
            ceiling_top when max_ceiling is not set.
        num_ceilings: Number of ceiling levels to sweep per pass.
            When discovering ceiling_top, batch size is n_workers instead.
        attempts_per_ceiling: Searches per ceiling level (default 1). With
            tight energy constraints, 3-5 gives more chances to find paths
            through different dropout randomness.
        max_passes: Number of resolution refinement passes.
        max_sweep_rounds: Safety cap on sweep batches when discovering
            ceiling_top.
        max_ceiling: Upper bound of sweep range (eV). When set, ceilings
            are evenly spaced from base to this value on the first pass.
            Useful for resuming from a previous run's ceiling_top.
        xi_factor: xi multiplied by this each refinement pass.
        delta_factor: delta multiplied by this each refinement pass (rounded).
        dropout: Neighbor dropout probability.
        min_dropout: Minimum dropout for adaptive adjustment.
        beam_width: Max open-set size for beam search.
        heuristic_mode: Heuristic for A*.
        heuristic_weight: Weight for unimodular heuristics.
        n_workers: Parallel worker processes. 0 = auto (cores // 4 workers,
            targeting 4 TF threads each). 1 = sequential. Each worker
            creates its own energy calculator.
        verbose: Print progress.
        output_dir: Path to output directory (None = no file output).
        relax_endpoints: If True, relax both endpoint structures in
            continuous space (cell + positions) using ASE FIRE + ExpCellFilter
            before any CNF discretization. This ensures endpoints are at
            local energy minima, so reported barriers are meaningful.

    Returns:
        (barrier, best_path_cnfs, best_path_energies) or (None, None, None)
    """
    if energy_calc is None:
        energy_calc = GraceCalculator()

    # Relax endpoints in continuous space if requested
    if relax_endpoints:
        from tensorpotential.calculator.foundation_models import grace_fm
        ase_calc = grace_fm(energy_calc.model_string)

        if verbose:
            print("Relaxing endpoints in continuous space...")

        start_uc = _relax_unit_cell(
            start_uc, ase_calc, verbose=verbose, label="start")
        end_uc = _relax_unit_cell(
            end_uc, ase_calc, verbose=verbose, label="end")

        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            start_uc.to_cif(str(out / "start_relaxed.cif"))
            end_uc.to_cif(str(out / "end_relaxed.cif"))
            if verbose:
                print(f"  Relaxed CIFs saved to {out}")

    energy_cache = {}
    total_start = time.perf_counter()
    round_times = []
    round_num = 0

    best_barrier = None
    best_path = None
    best_energies = None
    best_elements = None
    best_xi = None
    best_delta = None
    total_paths_found = 0
    ceiling_top = max_ceiling  # None → discover by sweeping up; set → sweep known range

    # Pre-build params dict for manifest (endpoint CNFs filled per-pass)
    manifest_params = _ceiling_params_dict(
        xi, delta, step_per_atom, num_ceilings,
        attempts_per_ceiling, max_passes, max_sweep_rounds,
        xi_factor, delta_factor, dropout, min_dropout,
        beam_width, heuristic_mode, heuristic_weight, n_workers,
        [], [], relax_endpoints,
    )

    def _update_manifest():
        """Write manifest.json with current best result (called after each round)."""
        if output_dir is None:
            return
        elapsed = time.perf_counter() - total_start
        result = {
            "barrier": best_barrier,
            "best_path": [list(pt) for pt in best_path] if best_path else None,
            "best_path_energies": best_energies,
            "path_length": len(best_path) if best_path else 0,
            "total_paths_found": total_paths_found,
            "total_rounds": round_num,
            "best_xi": best_xi,
            "best_delta": best_delta,
            "ceiling_top": ceiling_top,
            "energy_cache_size": len(energy_cache),
        }
        _write_manifest(output_dir, manifest_params, result,
                        {"total_seconds": elapsed, "round_seconds": round_times})

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "rounds").mkdir(exist_ok=True)

    # Set up worker pool for parallel execution
    import multiprocessing
    total_cores = multiprocessing.cpu_count()
    _TF_THREADS_PER_WORKER = 4

    if n_workers == 0:
        # Auto: target 4 TF threads per worker, leave 1 core for main process
        n_workers = max(1, (total_cores - 1) // _TF_THREADS_PER_WORKER)
        if verbose:
            print(f"Auto workers: {n_workers} "
                  f"({total_cores} cores, {_TF_THREADS_PER_WORKER} threads/worker)")

    pool = None
    if n_workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        tf_threads = max(1, total_cores // n_workers)
        # Use 'spawn' context so each worker gets a fresh process.
        # With 'fork', a TF runtime initialized in the main process
        # (e.g. by relax_endpoints) is inherited and cannot be reconfigured.
        pool = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_search_worker,
            initargs=(tf_threads,),
            mp_context=mp.get_context('spawn'),
        )
        if verbose:
            print(f"Worker pool: {n_workers} workers, "
                  f"{tf_threads} TF threads each")

    try:
        for pass_num in range(max_passes):
            # Compute discretization for this pass
            pass_xi = xi * (xi_factor ** pass_num)
            pass_delta = round(delta * (delta_factor ** pass_num))

            # Clear energy cache — same integer coords map to different
            # physical structures at different xi/delta
            energy_cache.clear()

            # Get endpoint CNFs at this resolution
            start_cnfs, goal_cnfs = get_endpoint_cnfs(
                start_uc, end_uc, xi=pass_xi, delta=pass_delta
            )
            elements = start_cnfs[0].elements
            n_atoms = len(elements)
            energy_step = step_per_atom * n_atoms

            if pass_num == 0:
                manifest_params["elements"] = elements
                manifest_params["start_cnf_coords"] = [list(c.coords) for c in start_cnfs]
                manifest_params["goal_cnf_coords"] = [list(c.coords) for c in goal_cnfs]

            # Evaluate endpoint energies
            for cnf in start_cnfs + goal_cnfs:
                if cnf.coords not in energy_cache:
                    energy_cache[cnf.coords] = energy_calc.calculate_energy(cnf)

            endpoint_energies = [
                energy_cache[c.coords] for c in start_cnfs + goal_cnfs
            ]
            base = max(endpoint_energies) + 1e-6

            if verbose:
                phase = "SWEEP (discovering ceiling)" if ceiling_top is None else "REFINING"
                print(f"\n{'='*60}")
                print(f"Pass {pass_num}: {phase} "
                      f"(xi={pass_xi:.2f}, delta={pass_delta})")
                print(f"  {len(start_cnfs)} start CNFs, "
                      f"{len(goal_cnfs)} goal CNFs")
                print(f"  Endpoint energies: "
                      f"{[f'{e:.2f}' for e in endpoint_energies]}")
                print(f"  Base ceiling: {base:.2f} eV")
                if ceiling_top is not None:
                    print(f"  Ceiling top: {ceiling_top:.2f} eV")
                print(f"{'='*60}")

            # Calibrate max_iters via diagnostic Rust A* runs (no energy filter)
            current_dropout = dropout
            current_max_iters = _calibrate_max_iters(
                start_cnfs, goal_cnfs, beam_width,
                heuristic_mode, heuristic_weight, verbose,
            )

            if ceiling_top is None:
                # ── Sweep phase: discover ceiling_top by stepping up ──
                # Batch size = n_workers (one ceiling per worker); num_ceilings
                # is only used for refinement passes.
                sweep_batch_size = max(1, n_workers)
                found_in_sweep = False
                for batch_idx in range(max_sweep_rounds):
                    ceilings = [
                        base + (batch_idx * sweep_batch_size + i) * energy_step
                        for i in range(sweep_batch_size)
                    ]

                    if verbose:
                        print(f"\n  Sweep batch {batch_idx} "
                              f"(ceilings: {ceilings[0]:.2f} "
                              f"to {ceilings[-1]:.2f} eV)")

                    round_start = time.perf_counter()
                    results = _run_batch(
                        ceilings, start_cnfs, goal_cnfs, elements,
                        pass_xi, pass_delta, energy_calc, energy_cache,
                        current_dropout, current_max_iters, beam_width,
                        heuristic_mode, heuristic_weight,
                        n_workers, pool, verbose,
                        attempts_per_ceiling=attempts_per_ceiling,
                        pass_id=pass_num,
                    )
                    round_elapsed = time.perf_counter() - round_start
                    round_times.append(round_elapsed)

                    successes = [r for r in results if r["found"]]
                    total_paths_found += len(successes)

                    if output_dir is not None:
                        _write_round_json(output_dir, round_num, {
                            "round": round_num, "pass": pass_num,
                            "phase": "sweep", "batch_idx": batch_idx,
                            "xi": pass_xi, "delta": pass_delta,
                            "ceilings": ceilings,
                            "results": [_serialize_result(r) for r in results],
                            "elapsed_seconds": round_elapsed,
                        })
                    round_num += 1

                    if successes:
                        best_result = min(successes,
                                          key=lambda r: r["barrier"])
                        ceiling_top = best_result["barrier"]
                        best_barrier = best_result["barrier"]
                        best_path = best_result["path"]
                        best_energies = best_result["energies"]
                        best_elements = elements
                        best_xi = pass_xi
                        best_delta = pass_delta
                        found_in_sweep = True

                        if verbose:
                            print(f"\n  *** Path found! ceiling_top = "
                                  f"{ceiling_top:.2f} eV")
                        _update_manifest()
                        break

                    _update_manifest()

                    # Adapt parameters between sweep batches
                    successful_iters = [r["iterations"] for r in results
                                        if r["found"]]
                    current_dropout, current_max_iters = _adapt_params(
                        len(successes), len(results), successful_iters,
                        current_dropout, min_dropout,
                        current_max_iters,
                    )

                if not found_in_sweep:
                    if verbose:
                        print(f"\n  No path found after "
                              f"{max_sweep_rounds} sweep batches.")
                    break

            else:
                # ── Refinement phase: evenly spaced from base to ceiling_top ──
                if base >= ceiling_top:
                    if verbose:
                        print(f"\n  Base ({base:.2f}) >= ceiling_top "
                              f"({ceiling_top:.2f}), skipping pass")
                    continue

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
                    print(f"\n  {N} searches from {base:.2f} to "
                          f"{ceiling_top:.2f} eV "
                          f"(spacing={spacing:.2f} eV)")

                round_start = time.perf_counter()
                results = _run_batch(
                    ceilings, start_cnfs, goal_cnfs, elements,
                    pass_xi, pass_delta, energy_calc, energy_cache,
                    current_dropout, current_max_iters, beam_width,
                    heuristic_mode, heuristic_weight,
                    n_workers, pool, verbose,
                    attempts_per_ceiling=attempts_per_ceiling,
                    pass_id=pass_num,
                )
                round_elapsed = time.perf_counter() - round_start
                round_times.append(round_elapsed)

                successes = [r for r in results if r["found"]]
                total_paths_found += len(successes)

                if output_dir is not None:
                    _write_round_json(output_dir, round_num, {
                        "round": round_num, "pass": pass_num,
                        "phase": "refining",
                        "xi": pass_xi, "delta": pass_delta,
                        "ceilings": ceilings, "spacing": spacing,
                        "results": [_serialize_result(r) for r in results],
                        "elapsed_seconds": round_elapsed,
                    })
                round_num += 1

                if successes:
                    best_result = min(successes,
                                      key=lambda r: r["barrier"])
                    ceiling_top = best_result["barrier"]
                    if (best_barrier is None
                            or best_result["barrier"] < best_barrier):
                        best_barrier = best_result["barrier"]
                        best_path = best_result["path"]
                        best_energies = best_result["energies"]
                        best_elements = elements
                        best_xi = pass_xi
                        best_delta = pass_delta

                    if verbose:
                        print(f"\n  *** ceiling_top tightened to "
                              f"{ceiling_top:.2f} eV")
                else:
                    if verbose:
                        print(f"\n  No paths found at this resolution")

                _update_manifest()

            if verbose:
                print(f"\n  Pass {pass_num} done: "
                      f"cache={len(energy_cache)} pts")
                if best_barrier is not None:
                    print(f"  Best barrier: {best_barrier:.4f} eV")

    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    if best_path is None:
        if verbose:
            print("\nNo path found across all passes.")
        _update_manifest()
        if output_dir is not None:
            _write_energy_cache(output_dir, energy_cache)
        return None, None, None

    # Convert best path to CNFs
    best_path_cnfs = [
        CrystalNormalForm.from_tuple(pt, best_elements, best_xi, best_delta)
        for pt in best_path
    ]

    if verbose:
        total_elapsed = time.perf_counter() - total_start
        print(f"\n{'='*60}")
        print(f"Final result:")
        print(f"  Barrier: {best_barrier:.4f} eV")
        print(f"  Path length: {len(best_path)} steps")
        print(f"  Resolution: xi={best_xi:.2f}, delta={best_delta}")
        print(f"  Total paths found: {total_paths_found}")
        print(f"  Energy cache size: {len(energy_cache)}")
        print(f"  Ceiling top: {ceiling_top:.4f} eV")
        print(f"  Total time: {total_elapsed:.1f}s")
        print(f"{'='*60}")

    _update_manifest()
    if output_dir is not None:
        _write_energy_cache(output_dir, energy_cache)

    return best_barrier, best_path_cnfs, best_energies


def _ceiling_params_dict(
    xi, delta, step_per_atom, num_ceilings, attempts_per_ceiling,
    max_passes, max_sweep_rounds,
    xi_factor, delta_factor, dropout, min_dropout,
    beam_width, heuristic_mode, heuristic_weight, n_workers,
    start_cnfs, goal_cnfs, relax_endpoints=False,
):
    """Build params dict for manifest.json."""
    return {
        "algorithm": "ceiling_barrier_search",
        "xi_initial": xi, "delta_initial": delta,
        "step_per_atom": step_per_atom,
        "num_ceilings": num_ceilings,
        "attempts_per_ceiling": attempts_per_ceiling,
        "max_passes": max_passes,
        "max_sweep_rounds": max_sweep_rounds,
        "xi_factor": xi_factor, "delta_factor": delta_factor,
        "dropout": dropout, "min_dropout": min_dropout,
        "beam_width": beam_width,
        "heuristic_mode": heuristic_mode,
        "heuristic_weight": heuristic_weight,
        "n_workers": n_workers,
        "relax_endpoints": relax_endpoints,
        "elements": start_cnfs[0].elements if start_cnfs else [],
        "start_cnf_coords": [list(c.coords) for c in start_cnfs],
        "goal_cnf_coords": [list(c.coords) for c in goal_cnfs],
    }
