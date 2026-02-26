"""Parameter adaptation and calibration for A* search."""

from cnf.navigation.astar import astar_rust
from cnf.navigation.search_filters import MinDistanceFilter


_DIAG_RUNS = 3
_DIAG_DROPOUT = 0.6
_DIAG_SAFETY_FACTOR = 2


def adapt_params(found, total, successful_iters,
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


def calibrate_max_iters(start_cnfs, goal_cnfs, beam_width, verbose):
    """Run diagnostic Rust A* searches to calibrate max_iters for this resolution."""
    min_dist_filter = MinDistanceFilter.from_structures(start_cnfs + goal_cnfs)
    diag_min_dist = min_dist_filter.dist

    if verbose:
        print(f"\n  Diagnostic min_dist={diag_min_dist:.3f} A", flush=True)

    trial_max_iters = 500

    while True:
        successful_iters = []

        if verbose:
            print(f"  Diagnostic (max_iters={trial_max_iters}, "
                  f"dropout={_DIAG_DROPOUT}):", flush=True)

        for i in range(_DIAG_RUNS):
            result, num_iters = astar_rust(
                start_cnfs, goal_cnfs,
                min_distance=diag_min_dist,
                max_iterations=trial_max_iters,
                beam_width=beam_width,
                dropout=_DIAG_DROPOUT,
                verbose=False,
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
            calibrated = max(calibrated, max(successful_iters))
            if verbose:
                print(f"  Calibrated max_iters={calibrated} "
                      f"(avg={avg_iters:.0f}, "
                      f"max={max(successful_iters)}, "
                      f"safety={_DIAG_SAFETY_FACTOR}x)", flush=True)
            return calibrated

        trial_max_iters *= 2
        if verbose:
            print(f"  No diagnostic paths found, "
                  f"doubling to {trial_max_iters}...", flush=True)
