"""Estimate max_iterations for A* searches."""

from cnf import CrystalNormalForm


def estimate_max_iterations(
    start_cnfs: list[CrystalNormalForm],
    goal_cnfs: list[CrystalNormalForm],
    beam_width: int = 1000,
    initial_max_iters: int = 5_000,
    max_retries: int = 5,
    multiplier: float = 1.5,
    verbosity: int = 1,
) -> int:
    """Estimate max_iterations by running plain A* searches.

    Runs A* (without energy filtering) with increasing max_iterations
    until a path is found, then returns iterations_used * 1.5 as the estimate.

    Args:
        start_cnfs: Starting CNF structures.
        goal_cnfs: Goal CNF structures.
        beam_width: Beam width for A* search.
        initial_max_iters: Starting max iterations (default 5000).
        max_retries: Max attempts, multiplying max_iters each time.
        multiplier: Factor to multiply max_iters on failure (default 1.5).
        verbosity: 0=silent, 1=progress.

    Returns:
        Estimated max_iterations for subsequent searches.
    """
    from cnf.navigation.astar import astar_rust

    max_iters = initial_max_iters

    for attempt in range(max_retries):
        if verbosity >= 1:
            print(f"  Estimating max_iters: attempt {attempt + 1}/{max_retries}, trying {max_iters}")

        path, iterations = astar_rust(
            start_cnfs=start_cnfs,
            goal_cnfs=goal_cnfs,
            max_iterations=max_iters,
            beam_width=beam_width,
            dropout=0.0,  # Deterministic for estimation
            greedy=False,
            min_distance=0.0,
            verbose=False,
        )

        if path is not None:
            estimated = int(iterations * 1.5)
            if verbosity >= 1:
                print(f"  Path found in {iterations} iters -> using max_iters={estimated}")
            return estimated

        # Failed, increase max_iters
        max_iters = int(max_iters * multiplier)

    # All retries failed, return the final max_iters we tried
    if verbosity >= 1:
        print(f"  Warning: No path after {max_retries} attempts, using max_iters={max_iters}")
    return max_iters
