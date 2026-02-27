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
from cnf.navigation.astar.heuristics import manhattan_distance
from cnf.navigation.astar.models import (
    PathContext, Path, Attempt, SearchParameters, SearchResult, RefinementResult
)
from cnf.navigation.search_filters import FilterSet, EnergyFilter

from ..core import evaluate_path_energies, path_barrier
from .params import adapt_params

USE_RUST = os.getenv('USE_RUST') == '1'


def ratchet(
    start_cnfs: list[CrystalNormalForm],
    goal_cnfs: list[CrystalNormalForm],
    initial_ceiling: float,
    energy_calc=None,
    paths_per_round: int = 10,
    max_rounds: int = 20,
    dropout: float = 0.1,
    min_dropout: float = 0.1,
    max_iterations: int = 100_000,
    beam_width: int = 1000,
    verbosity: int = 1,
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
        verbosity: 0=silent, 1=phase output, 2+=A* iteration progress.
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

    context = PathContext(xi=xi, delta=delta, elements=elements)

    energy_cache = {}

    for cnf in start_cnfs + goal_cnfs:
        if cnf.coords not in energy_cache:
            energy_cache[cnf.coords] = energy_calc.calculate_energy(cnf)

    if output_dir is not None:
        output_dir = PathlibPath(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    ceiling = initial_ceiling
    total_start = time.perf_counter()

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
        if output_dir is not None:
            outpath = str(output_dir / "refinement_result.json")
            result.to_json(outpath)
            print(f"  [Ratchet] Saved: {outpath}")

    if verbosity >= 1:
        print(f"\nStarting refinement with ceiling={initial_ceiling:.4f} eV")

    current_dropout = dropout
    current_max_iters = min(max_iterations, 500)

    for round_num in range(max_rounds):
        round_start = time.perf_counter()
        round_ceiling = ceiling
        improved = False
        round_successful_iters = []

        if verbosity >= 1:
            print(f"\n{'='*60}")
            print(f"Round {round_num} (ceiling={round_ceiling:.4f} eV, "
                  f"dropout={current_dropout:.2f}, max_iters={current_max_iters})")
            print(f"{'='*60}")

        round_params = SearchParameters(
            max_iterations=current_max_iters,
            beam_width=beam_width,
            dropout=current_dropout,
            greedy=False,
            heuristic="manhattan",
            filters=[{"type": "energy_ceiling", "value": round_ceiling}],
        )

        attempts = []

        for path_idx in range(paths_per_round):
            if verbosity >= 1:
                print(f"  Path {path_idx+1}/{paths_per_round}...", end=" ", flush=True)

            attempt_start = time.perf_counter()

            energy_filter = EnergyFilter(
                round_ceiling, calc=energy_calc, cache=energy_cache
            )
            filter_set = FilterSet([energy_filter], use_structs=not USE_RUST)

            search_state = astar_pathfind(
                start_cnfs,
                goal_cnfs,
                heuristic=manhattan_distance,
                filter_set=filter_set,
                max_iterations=current_max_iters,
                beam_width=beam_width,
                dropout=current_dropout,
                verbose=(verbosity >= 2),
            )

            attempt_elapsed = time.perf_counter() - attempt_start
            num_iters = search_state.iterations

            if search_state.path is None:
                if verbosity >= 1:
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

            energies = evaluate_path_energies(
                path_tuples, elements, xi, delta, energy_calc, energy_cache
            )
            barrier = path_barrier(energies)

            if verbosity >= 1:
                print(f"len={len(path_tuples)}, barrier={barrier:.4f} eV, iters={num_iters}")

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

            if barrier < ceiling:
                ceiling = barrier
                improved = True

        round_elapsed = time.perf_counter() - round_start

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

        if verbosity >= 1:
            cache_size = len(energy_cache)
            print(f"  Round {round_num} summary: {len(round_result.paths)}/{paths_per_round} paths found "
                  f"({success_rate:.0%}), ceiling={ceiling:.4f} eV, cache={cache_size} pts, "
                  f"elapsed={round_elapsed:.1f}s")

        _save_result()

        prev_max_iters = current_max_iters
        current_dropout, current_max_iters = adapt_params(
            len(round_result.paths), paths_per_round, round_successful_iters,
            current_dropout, min_dropout, current_max_iters,
            max_iters=max_iterations,
        )

        if verbosity >= 1 and current_max_iters != prev_max_iters:
            print(f"  max_iters: {prev_max_iters} → {current_max_iters}")

        if not improved:
            at_limits = (current_dropout <= min_dropout and
                         current_max_iters >= max_iterations)
            if at_limits:
                if verbosity >= 1:
                    print(f"\n  Converged! No improvement with parameters at limits.")
                result.metadata["converged"] = True
                break
            else:
                if verbosity >= 1:
                    print(f"  No improvement — adapting parameters "
                          f"(dropout={current_dropout:.2f}, max_iters={current_max_iters})")

    total_elapsed = time.perf_counter() - total_start
    result.metadata["total_elapsed_seconds"] = total_elapsed
    result.metadata["final_ceiling"] = ceiling

    if verbosity >= 1:
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

    _save_result()

    return result
