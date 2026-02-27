"""Shared worker initialization for parallel A* phases."""

import os
import time

# Worker process globals
worker_calc = None
worker_cache = None


def init_worker(calc_provider, tf_threads=None, phase_name="Worker"):
    """Initialize a worker process with its own energy calculator.

    Args:
        calc_provider: Callable that returns a calculator instance.
            Must be picklable (use GraceCalcProvider, not a lambda).
        tf_threads: Number of TensorFlow threads per worker.
        phase_name: Name for logging (e.g., "Sample", "Sweep").
    """
    global worker_calc, worker_cache

    start_time = time.perf_counter()

    if tf_threads is not None:
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(tf_threads)
        tf.config.threading.set_intra_op_parallelism_threads(tf_threads)

    worker_calc = calc_provider()
    worker_cache = {}

    elapsed = time.perf_counter() - start_time
    print(f"  [{phase_name} Worker PID {os.getpid()}] Ready in {elapsed:.1f}s - {worker_calc.identifier()}")
