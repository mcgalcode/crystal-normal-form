"""Core search execution functions for iterative A*."""

import os
import time

from cnf.navigation.astar.core import astar_pathfind
from cnf.navigation.astar.heuristics import make_heuristic
from cnf.navigation.search_filters import FilterSet, EnergyFilter

from ._energy import evaluate_path_energies, path_barrier

USE_RUST = os.getenv('USE_RUST') == '1'


def search_at_ceiling(ceiling, start_cnfs, goal_cnfs, elements, xi, delta,
                      calc, cache, dropout, max_iters, beam_width,
                      heuristic_mode, heuristic_weight, log_prefix="      "):
    """Run a single A* search at the given energy ceiling.

    Returns dict with keys: ceiling, found, iterations, and if found:
    barrier, path, energies, path_length.
    """
    heuristic = make_heuristic(heuristic_mode, heuristic_weight)
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
    energies = evaluate_path_energies(
        path_tuples, elements, xi, delta, calc, cache
    )
    energy_evals = len(cache) - cache_before  # include path evaluation
    barrier = path_barrier(energies)

    return {
        "ceiling": ceiling, "found": True,
        "barrier": barrier, "path": path_tuples, "energies": energies,
        "iterations": search_state.iterations,
        "path_length": len(path_tuples),
        "energy_evals": energy_evals,
    }


def retry_search(ceiling, start_cnfs, goal_cnfs, elements, xi, delta,
                 calc, cache, dropout, max_iters, beam_width,
                 heuristic_mode, heuristic_weight, attempts,
                 max_iters_scale=1.5, log_prefix="    ", verbose=True):
    """Core retry loop: run up to `attempts` A* searches at a single ceiling.

    Bumps max_iters by max_iters_scale after each failed attempt.
    Returns result dict from first successful attempt, or last attempt's result.
    """
    current_max_iters = max_iters
    r = None
    for a in range(attempts):
        if verbose:
            attempt_str = f" #{a+1}" if attempts > 1 else ""
            print(f"{log_prefix}starting (ceiling={ceiling:.2f} eV{attempt_str}, "
                  f"max_iters={current_max_iters})...", flush=True)

        t0 = time.perf_counter()
        r = search_at_ceiling(
            ceiling, start_cnfs, goal_cnfs, elements, xi, delta,
            calc, cache, dropout, current_max_iters, beam_width,
            heuristic_mode, heuristic_weight, log_prefix=log_prefix,
        )
        elapsed = time.perf_counter() - t0

        if verbose:
            evals = r.get('energy_evals', '?')
            if r["found"]:
                print(f"{log_prefix}path found! len={r['path_length']}, "
                      f"barrier={r['barrier']:.2f} eV, "
                      f"iters={r['iterations']}, evals={evals}, {elapsed:.1f}s",
                      flush=True)
            else:
                print(f"{log_prefix}no path ({r['iterations']} iters, "
                      f"{evals} evals, {elapsed:.1f}s)", flush=True)

        if r["found"]:
            return r

        current_max_iters = int(current_max_iters * max_iters_scale)

    return r  # last attempt's result (not found)


def search_ceiling_with_attempts(ceiling, start_cnfs, goal_cnfs, elements,
                                 xi, delta, calc, cache, dropout, max_iters,
                                 beam_width, heuristic_mode, heuristic_weight,
                                 attempts, verbose):
    """Run up to `attempts` A* searches at a single ceiling, stop on first success."""
    return retry_search(
        ceiling, start_cnfs, goal_cnfs, elements, xi, delta,
        calc, cache, dropout, max_iters, beam_width,
        heuristic_mode, heuristic_weight, attempts,
        max_iters_scale=1.5, log_prefix="    ", verbose=verbose,
    )
