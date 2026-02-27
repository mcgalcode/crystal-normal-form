"""Worker pool support for parallel A* searches."""

from cnf import CrystalNormalForm

from .search import retry_search
from ..core import worker as core_worker


# Sweep-specific: track pass_id for cache invalidation between passes
_worker_pass_id = None


def init_search_worker(calc_provider, tf_threads=None):
    """Initialize a sweep worker. Wraps core init_worker with phase name."""
    global _worker_pass_id
    _worker_pass_id = None
    core_worker.init_worker(calc_provider, tf_threads, phase_name="Sweep")


def worker_search_with_attempts(args):
    """Worker wrapper: deserialize inputs, manage worker state, then run retry search."""
    global _worker_pass_id
    (ceiling, start_coord_lists, goal_coord_lists, elements, xi, delta,
     dropout, max_iters, beam_width,
     seed_cache_items, attempts, worker_label, pass_id, verbosity) = args

    if pass_id != _worker_pass_id:
        core_worker.worker_cache = {}
        _worker_pass_id = pass_id

    for k, v in seed_cache_items:
        if k not in core_worker.worker_cache:
            core_worker.worker_cache[k] = v

    start_cnfs = [CrystalNormalForm.from_tuple(tuple(c), elements, xi, delta)
                  for c in start_coord_lists]
    goal_cnfs = [CrystalNormalForm.from_tuple(tuple(c), elements, xi, delta)
                 for c in goal_coord_lists]

    return retry_search(
        ceiling, start_cnfs, goal_cnfs, elements, xi, delta,
        core_worker.worker_calc, core_worker.worker_cache, dropout, max_iters, beam_width, attempts,
        max_iters_scale=1.0,
        log_prefix=f"    [{worker_label}] ",
        verbosity=verbosity,
    )
