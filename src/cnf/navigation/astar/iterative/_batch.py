"""Batch execution of A* searches across multiple ceilings."""

from ._search import search_ceiling_with_attempts
from ._workers import worker_search_with_attempts


def run_batch(ceilings, start_cnfs, goal_cnfs, elements, xi, delta,
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
                    attempts_per_ceiling, f"c={c:.2f} eV", pass_id)
            f = pool.submit(worker_search_with_attempts, args)
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
                            print(f"    [c={other_c:.2f} eV] cancelled "
                                  f"(path found at {best_found_ceiling:.2f} eV)",
                                  flush=True)

        results.sort(key=lambda r: r["ceiling"])
    else:
        results = []
        for i, c in enumerate(ceilings):
            r = search_ceiling_with_attempts(
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
