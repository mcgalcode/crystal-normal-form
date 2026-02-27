"""Worker pool support for parallel A* searches."""

from cnf import CrystalNormalForm

from .search import retry_search


_worker_calc = None
_worker_cache = None
_worker_pass_id = None


def init_search_worker(calc_provider, tf_threads=None):
    """Initialize a worker process with its own energy calculator and cache.

    Args:
        calc_provider: Callable that returns a calculator instance.
            Must be picklable (use GraceCalcProvider, not a lambda).
        tf_threads: Number of TensorFlow threads per worker.
    """
    import os
    if tf_threads is not None:
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(tf_threads)
        tf.config.threading.set_intra_op_parallelism_threads(tf_threads)
    global _worker_calc, _worker_cache, _worker_pass_id
    _worker_calc = calc_provider()
    _worker_cache = {}
    _worker_pass_id = None
    print(f"  [Worker PID {os.getpid()}] Calculator initialized: {_worker_calc.identifier()}")


def worker_search_with_attempts(args):
    """Worker wrapper: deserialize inputs, manage worker state, then run retry search."""
    global _worker_calc, _worker_cache, _worker_pass_id
    (ceiling, start_coord_lists, goal_coord_lists, elements, xi, delta,
     dropout, max_iters, beam_width,
     seed_cache_items, attempts, worker_label, pass_id, verbosity) = args

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

    return retry_search(
        ceiling, start_cnfs, goal_cnfs, elements, xi, delta,
        _worker_calc, _worker_cache, dropout, max_iters, beam_width, attempts,
        max_iters_scale=1.0,
        log_prefix=f"    [{worker_label}] ",
        verbosity=verbosity,
    )
