"""Core search execution functions for iterative A*."""

import os
import time

from cnf.navigation.astar.core import astar_pathfind
from cnf.navigation.astar.heuristics import manhattan_distance
from cnf.navigation.search_filters import FilterSet, EnergyFilter

from ._energy import evaluate_path_energies, path_barrier

USE_RUST = os.getenv('USE_RUST') == '1'


def search_at_ceiling(ceiling, start_cnfs, goal_cnfs, elements, xi, delta,
                      calc, cache, dropout, max_iters, beam_width,
                      log_prefix="      ", verbosity=1):
    """Run a single A* search at the given energy ceiling.

    Args:
        verbosity: 0=silent, 1=phase output, 2+=A* iteration progress.
    """
    energy_filter = EnergyFilter(ceiling, calc=calc, cache=cache)
    filter_set = FilterSet([energy_filter], use_structs=not USE_RUST)

    cache_before = len(cache)
    speak_freq = max(1, max_iters // 10)
    search_state = astar_pathfind(
        start_cnfs, goal_cnfs,
        heuristic=manhattan_distance, filter_set=filter_set,
        max_iterations=max_iters, beam_width=beam_width,
        dropout=dropout, verbose=(verbosity >= 2), speak_freq=speak_freq,
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
    energy_evals = len(cache) - cache_before
    barrier = path_barrier(energies)

    return {
        "ceiling": ceiling, "found": True,
        "barrier": barrier, "path": path_tuples, "energies": energies,
        "iterations": search_state.iterations,
        "path_length": len(path_tuples),
        "energy_evals": energy_evals,
    }


def retry_search(ceiling, start_cnfs, goal_cnfs, elements, xi, delta,
                 calc, cache, dropout, max_iters, beam_width, attempts,
                 max_iters_scale=1.5, log_prefix="    ", verbosity=1):
    """Core retry loop: run up to `attempts` A* searches at a single ceiling.

    Args:
        verbosity: 0=silent, 1=phase output, 2+=A* iteration progress.
    """
    current_max_iters = max_iters
    r = None
    for a in range(attempts):
        if verbosity >= 1:
            attempt_str = f" #{a+1}" if attempts > 1 else ""
            print(f"{log_prefix}starting (ceiling={ceiling:.2f} eV{attempt_str}, "
                  f"max_iters={current_max_iters})...", flush=True)

        t0 = time.perf_counter()
        r = search_at_ceiling(
            ceiling, start_cnfs, goal_cnfs, elements, xi, delta,
            calc, cache, dropout, current_max_iters, beam_width,
            log_prefix=log_prefix, verbosity=verbosity,
        )
        elapsed = time.perf_counter() - t0

        if verbosity >= 1:
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

    return r


def search_ceiling_with_attempts(ceiling, start_cnfs, goal_cnfs, elements,
                                 xi, delta, calc, cache, dropout, max_iters,
                                 beam_width, attempts, verbosity):
    """Run up to `attempts` A* searches at a single ceiling, stop on first success.

    Args:
        verbosity: 0=silent, 1=phase output, 2+=A* iteration progress.
    """
    return retry_search(
        ceiling, start_cnfs, goal_cnfs, elements, xi, delta,
        calc, cache, dropout, max_iters, beam_width, attempts,
        max_iters_scale=1.5, log_prefix="    ", verbosity=verbosity,
    )
