"""Serial barrier refinement with ratcheting ceiling.

Runs multiple A* searches with energy ceiling filter, evaluates energies,
and ratchets down the ceiling when better paths are found. Iterates until
the ceiling can't be lowered.

Each round:
    Run N A* searches with EnergyFilter(ceiling)
    Ceiling is fixed per round (all paths see the same constraint)
    After each round, adjust dropout and max_iters based on success rate
    Converge when no improvement and parameters are at their limits
"""

import os
import time
from pathlib import Path as PathlibPath

from cnf import CrystalNormalForm
from cnf.calculation.grace import GraceCalculator
from cnf.navigation.astar.core import astar_pathfind
from cnf.navigation.astar.heuristics import make_heuristic
from cnf.navigation.astar.models import (
    PathContext, Path, Attempt, SearchParameters, SearchResult, RefinementResult
)
from cnf.navigation.search_filters import FilterSet, EnergyFilter

from ._energy import evaluate_path_energies, path_barrier
from ._params import adapt_params

USE_RUST = os.getenv('USE_RUST') == '1'


def ratchet(
    start_cnfs: list[CrystalNormalForm],
    goal_cnfs: list[CrystalNormalForm],
    initial_ceiling: float,
    energy_calc=None,
    paths_per_round: int = 10,
    max_rounds: int = 20,
    dropout: float = 0.3,
    min_dropout: float = 0.1,
    max_iterations: int = 100_000,
    beam_width: int = 1000,
    heuristic_mode: str = "manhattan",
    heuristic_weight: float = 0.5,
    verbose: bool = True,
    output_dir: PathlibPath | str | None = None,
) -> RefinementResult:
    """Serial barrier refinement with ratcheting ceiling.

    Requires an initial ceiling (from Phase 2 path sampling or ceiling sweep).
    Runs multiple rounds of A* searches with energy filtering, ratcheting
    down the ceiling when better paths are found.

    Args:
        start_cnfs: Starting CNF structures.
        goal_cnfs: Goal CNF structures.
        initial_ceiling: Starting energy ceiling (eV). Required.
        energy_calc: Energy calculator (default: GraceCalculator).
        paths_per_round: Number of A* runs per round.
        max_rounds: Maximum refinement rounds.
        dropout: Initial neighbor dropout probability.
        min_dropout: Minimum dropout for adaptive adjustment.
        max_iterations: Max A* iterations (absolute cap).
        beam_width: Max open-set size for beam search.
        heuristic_mode: Heuristic for A* ("manhattan", "unimodular_light", etc.).
        heuristic_weight: Weight for unimodular heuristics.
        verbose: Print progress.
        output_dir: Path to output directory. If set, writes refinement_result.json
            after each round (overwriting) for crash resilience.

    Returns:
        RefinementResult containing all rounds and their attempts.
    """
    if energy_calc is None:
        energy_calc = GraceCalculator()

    elements = start_cnfs[0].elements
    xi = start_cnfs[0].xi
    delta = start_cnfs[0].delta

    # Build the shared context
    context = PathContext(xi=xi, delta=delta, elements=elements)

    energy_cache = {}  # shared across all rounds

    # Evaluate endpoint energies and seed the cache
    for cnf in start_cnfs + goal_cnfs:
        if cnf.coords not in energy_cache:
            energy_cache[cnf.coords] = energy_calc.calculate_energy(cnf)

    # Set up output directory
    if output_dir is not None:
        output_dir = PathlibPath(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Build the heuristic string for metadata
    heuristic_str = heuristic_mode
    if heuristic_weight != 1.0:
        heuristic_str = f"{heuristic_mode}:{heuristic_weight}"

    ceiling = initial_ceiling
    total_start = time.perf_counter()

    # Initialize result structure
    result = RefinementResult(
        results=[],
        metadata={
            "initial_ceiling": initial_ceiling,
            "paths_per_round": paths_per_round,
            "max_rounds": max_rounds,
            "start_cnf_coords": [list(c.coords) for c in start_cnfs],
            "goal_cnf_coords": [list(c.coords) for c in goal_cnfs],
        }
    )

    def _save_result():
        """Save current result to disk (overwrites each round)."""
        if output_dir is not None:
            result.to_json(str(output_dir / "refinement_result.json"))

    if verbose:
        print(f"\nStarting refinement with ceiling={initial_ceiling:.4f} eV")

    # Adaptive parameters
    current_dropout = dropout
    current_max_iters = min(max_iterations, 500)

    for round_num in range(max_rounds):
        round_start = time.perf_counter()
        round_ceiling = ceiling  # fixed for this round
        improved = False
        round_successful_iters = []

        if verbose:
            print(f"\n{'='*60}")
            print(f"Round {round_num} (ceiling={round_ceiling:.4f} eV, "
                  f"dropout={current_dropout:.2f}, max_iters={current_max_iters})")
            print(f"{'='*60}")

        # Build parameters for this round
        round_params = SearchParameters(
            max_iterations=current_max_iters,
            beam_width=beam_width,
            dropout=current_dropout,
            greedy=False,
            heuristic=heuristic_str,
            filters=[{"type": "energy_ceiling", "value": round_ceiling}],
        )

        attempts = []

        for path_idx in range(paths_per_round):
            if verbose:
                print(f"  Path {path_idx+1}/{paths_per_round}...", end=" ", flush=True)

            attempt_start = time.perf_counter()

            # Run A* with energy filter
            heuristic = make_heuristic(heuristic_mode, heuristic_weight)
            energy_filter = EnergyFilter(
                round_ceiling, calc=energy_calc, cache=energy_cache
            )
            filter_set = FilterSet([energy_filter], use_structs=not USE_RUST)

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

            attempt_elapsed = time.perf_counter() - attempt_start
            num_iters = search_state.iterations

            if search_state.path is None:
                if verbose:
                    print("no path found")
                attempts.append(Attempt(
                    path=None,
                    found=False,
                    iterations=num_iters,
                    elapsed_seconds=attempt_elapsed,
                ))
                continue

            path_tuples = search_state.path
            round_successful_iters.append(num_iters)

            # Evaluate energies along path
            energies = evaluate_path_energies(
                path_tuples, elements, xi, delta, energy_calc, energy_cache
            )
            barrier = path_barrier(energies)

            if verbose:
                print(f"len={len(path_tuples)}, barrier={barrier:.4f} eV, iters={num_iters}")

            # Create Path object
            path_obj = Path(
                coords=[tuple(pt) for pt in path_tuples],
                energies=energies,
                barrier=barrier,
            )

            attempts.append(Attempt(
                path=path_obj,
                found=True,
                iterations=num_iters,
                elapsed_seconds=attempt_elapsed,
            ))

            # Track best and update ceiling for next round
            if barrier < ceiling:
                ceiling = barrier
                improved = True

        round_elapsed = time.perf_counter() - round_start

        # Build SearchResult for this round
        round_result = SearchResult(
            context=context,
            parameters=round_params,
            attempts=attempts,
            metadata={
                "round": round_num,
                "ceiling_at_start": round_ceiling,
                "ceiling_at_end": ceiling,
                "improved": improved,
                "elapsed_seconds": round_elapsed,
            }
        )
        result.results.append(round_result)

        success_rate = round_result.success_rate

        if verbose:
            cache_size = len(energy_cache)
            print(f"  Round {round_num} summary: {len(round_result.paths)}/{paths_per_round} paths found "
                  f"({success_rate:.0%}), ceiling={ceiling:.4f} eV, cache={cache_size} pts, "
                  f"elapsed={round_elapsed:.1f}s")

        # Save after each round for crash resilience
        _save_result()

        # Adaptive parameter adjustment
        current_dropout, current_max_iters = adapt_params(
            len(round_result.paths), paths_per_round, round_successful_iters,
            current_dropout, min_dropout, current_max_iters,
            max_iters=max_iterations,
        )

        # Convergence: only if no improvement AND parameters at limits
        if not improved:
            at_limits = (current_dropout <= min_dropout and
                         current_max_iters >= max_iterations)
            if at_limits:
                if verbose:
                    print(f"\n  Converged! No improvement with parameters at limits.")
                result.metadata["converged"] = True
                break
            else:
                if verbose:
                    print(f"  No improvement — adapting parameters "
                          f"(dropout={current_dropout:.2f}, max_iters={current_max_iters})")

    total_elapsed = time.perf_counter() - total_start
    result.metadata["total_elapsed_seconds"] = total_elapsed
    result.metadata["final_ceiling"] = ceiling

    if verbose:
        best = result.best_path
        print(f"\n{'='*60}")
        print(f"Final result:")
        if best:
            print(f"  Barrier: {best.barrier:.4f} eV")
            print(f"  Path length: {len(best)} steps")
        else:
            print(f"  No path found")
        print(f"  Total rounds: {len(result.results)}")
        print(f"  Energy cache size: {len(energy_cache)}")
        print(f"  Total time: {total_elapsed:.1f}s")
        print(f"{'='*60}")

    # Final save
    _save_result()

    return result
