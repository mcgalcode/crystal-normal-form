"""Serial barrier refinement with ratcheting ceiling.

Runs A* searches with energy ceiling filter, evaluates energies,
and ratchets down the ceiling when better paths are found. If 3 consecutive
searches fail, adapts discretization parameters (xi, delta) and continues.
Terminates after max_adaptations are exhausted and 3 more consecutive failures.

Algorithm:
    1. Start at given ceiling
    2. Run A* search with EnergyFilter(ceiling)
    3. If path found:
        - Lower ceiling by at least 2 meV/atom (or to new barrier if lower)
        - Continue without adapting parameters
    4. If 3 consecutive failures:
        - Adapt xi and delta (finer discretization)
        - Reset failure counter and continue
    5. Terminate when max_adaptations reached AND 3 more consecutive failures
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

from ..core import evaluate_path_energies, path_barrier, estimate_max_iterations

USE_RUST = os.getenv('USE_RUST') == '1'


def ratchet(
    start_cnfs: list[CrystalNormalForm],
    goal_cnfs: list[CrystalNormalForm],
    initial_ceiling: float,
    energy_calc=None,
    start_uc=None,
    end_uc=None,
    min_atoms: int | None = None,
    max_adaptations: int = 5,
    xi_factor: float = 0.8,
    delta_factor: float = 1.2,
    ceiling_step_mev_per_atom: float = 2.0,
    dropout: float = 0.1,
    max_iterations: int = 100_000,
    initial_max_iters: int | None = None,
    beam_width: int = 1000,
    verbosity: int = 1,
    output_dir: PathlibPath | str | None = None,
) -> RefinementResult:
    """Serial barrier refinement with ratcheting ceiling.

    Requires an initial ceiling (from Phase 2 path sampling or ceiling sweep).
    Runs A* searches with energy filtering, ratcheting down the ceiling when
    better paths are found. Adapts discretization parameters after consecutive
    failures.

    Args:
        start_cnfs: Starting CNF structures.
        goal_cnfs: Goal CNF structures.
        initial_ceiling: Starting energy ceiling (eV). Required.
        energy_calc: Energy calculator (default: GraceCalculator).
        start_uc: Starting UnitCell (required for parameter adaptation).
        end_uc: Ending UnitCell (required for parameter adaptation).
        min_atoms: Minimum atoms for supercell (passed to get_endpoint_cnfs).
        max_adaptations: Maximum xi/delta adaptations before final termination.
        xi_factor: Multiply xi by this factor on each adaptation (< 1 = finer).
        delta_factor: Multiply delta by this factor on each adaptation (> 1 = finer).
        ceiling_step_mev_per_atom: Minimum ceiling reduction per atom in meV
            when a path is found (2.0 meV/atom = 0.002 eV/atom).
        dropout: Neighbor dropout probability.
        max_iterations: Max A* iterations (absolute cap).
        initial_max_iters: Starting max iterations. If None, defaults to
            min(max_iterations, 5000).
        beam_width: Max open-set size for beam search.
        verbosity: 0=silent, 1=phase output, 2+=A* iteration progress.
        output_dir: Path to output directory. If set, writes refinement_result.json
            after each attempt for crash resilience.

    Returns:
        RefinementResult containing all attempts.
    """
    from cnf.navigation.endpoints import get_endpoint_cnfs

    if energy_calc is None:
        energy_calc = GraceCalculator()

    elements = start_cnfs[0].elements
    n_atoms = len(elements)
    xi = start_cnfs[0].xi
    delta = start_cnfs[0].delta

    # Convert meV/atom to eV total
    ceiling_step_ev = (ceiling_step_mev_per_atom / 1000.0) * n_atoms

    energy_cache = {}

    for cnf in start_cnfs + goal_cnfs:
        if cnf.coords not in energy_cache:
            energy_cache[cnf.coords] = energy_calc.calculate_energy(cnf)

    # Compute max endpoint energy per atom from DISCRETIZED CNFs
    # This ensures consistent energy reference across all path evaluations
    endpoint_energies = [energy_cache[cnf.coords] for cnf in start_cnfs + goal_cnfs]
    e_max_endpoint_per_atom = max(endpoint_energies) / n_atoms

    # Also compute original structure energies for comparison/logging
    if start_uc is not None and end_uc is not None:
        e_start_orig = energy_calc.calculate_structure_energy(start_uc)
        e_end_orig = energy_calc.calculate_structure_energy(end_uc)
        n_atoms_start = len(start_uc)
        n_atoms_end = len(end_uc)
        e_max_orig_per_atom = max(e_start_orig / n_atoms_start, e_end_orig / n_atoms_end)
        discretization_shift = (e_max_endpoint_per_atom - e_max_orig_per_atom) * 1000
    else:
        e_max_orig_per_atom = None
        discretization_shift = None

    def to_mev_above_endpoint(energy_ev):
        """Convert absolute energy to meV/atom above max discretized endpoint."""
        energy_per_atom = energy_ev / n_atoms
        return (energy_per_atom - e_max_endpoint_per_atom) * 1000.0

    def to_mev_above_orig_endpoint(energy_ev):
        """Convert absolute energy to meV/atom above max original endpoint."""
        if e_max_orig_per_atom is None:
            return None
        energy_per_atom = energy_ev / n_atoms
        return (energy_per_atom - e_max_orig_per_atom) * 1000.0

    # Check that ceiling is above max endpoint energy
    max_endpoint_energy = max(endpoint_energies)
    if initial_ceiling < max_endpoint_energy:
        ceiling_mev = to_mev_above_endpoint(initial_ceiling)
        raise ValueError(
            f"Initial ceiling ({initial_ceiling:.4f} eV = {ceiling_mev:.1f} meV/atom above endpoint) "
            f"is below the max discretized endpoint energy ({max_endpoint_energy:.4f} eV). "
            f"No valid path can exist. Increase the ceiling or check the input structures."
        )

    if output_dir is not None:
        output_dir = PathlibPath(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    ceiling = initial_ceiling
    total_start = time.perf_counter()

    result = RefinementResult(
        results=[],
        metadata={
            "initial_ceiling": initial_ceiling,
            "max_adaptations": max_adaptations,
            "xi_factor": xi_factor,
            "delta_factor": delta_factor,
            "ceiling_step_mev_per_atom": ceiling_step_mev_per_atom,
            "start_cnf_coords": [list(c.coords) for c in start_cnfs],
            "goal_cnf_coords": [list(c.coords) for c in goal_cnfs],
        }
    )

    def _save_result():
        if output_dir is not None:
            outpath = str(output_dir / "refinement_result.json")
            result.to_json(outpath)
            if verbosity >= 1:
                print(f"  [Ratchet] Saved: {outpath}")

    if verbosity >= 1:
        initial_ceiling_mev = to_mev_above_endpoint(initial_ceiling)
        print(f"\nStarting ratchet refinement")
        print(f"  Max endpoint energy (discretized): {e_max_endpoint_per_atom:.4f} eV/atom")
        if e_max_orig_per_atom is not None:
            print(f"  Max endpoint energy (original):    {e_max_orig_per_atom:.4f} eV/atom")
            print(f"  Discretization shift: {discretization_shift:+.1f} meV/atom")
        print(f"  Initial ceiling: {initial_ceiling_mev:.1f} meV/atom above endpoint")
        print(f"  Ceiling step: {ceiling_step_mev_per_atom:.1f} meV/atom")
        print(f"  Max adaptations: {max_adaptations}")
        print(f"  xi_factor={xi_factor}, delta_factor={delta_factor}")

    # Estimate initial max_iters via plain A* search
    if initial_max_iters is not None:
        current_max_iters = initial_max_iters
        if verbosity >= 1:
            print(f"\nUsing provided initial_max_iters={current_max_iters}")
    else:
        if verbosity >= 1:
            print(f"\nEstimating initial max_iters...")
        current_max_iters = estimate_max_iterations(
            start_cnfs=start_cnfs,
            goal_cnfs=goal_cnfs,
            beam_width=beam_width,
            verbosity=verbosity,
        )
        # Cap at max_iterations
        current_max_iters = min(current_max_iters, max_iterations)

    consecutive_failures = 0
    num_adaptations = 0
    attempt_num = 0

    while True:
        context = PathContext(xi=xi, delta=delta, elements=elements)

        ceiling_mev = to_mev_above_endpoint(ceiling)
        ceiling_mev_orig = to_mev_above_orig_endpoint(ceiling)
        if verbosity >= 1:
            print(f"\n{'='*60}")
            if ceiling_mev_orig is not None:
                print(f"Attempt {attempt_num} (ceiling={ceiling_mev:.1f}/{ceiling_mev_orig:.1f} meV/atom disc/orig, "
                      f"xi={xi:.2f}, delta={delta}, max_iters={current_max_iters})")
            else:
                print(f"Attempt {attempt_num} (ceiling={ceiling_mev:.1f} meV/atom, "
                      f"xi={xi:.2f}, delta={delta}, max_iters={current_max_iters})")
            print(f"  Consecutive failures: {consecutive_failures}, "
                  f"adaptations: {num_adaptations}/{max_adaptations}")
            print(f"{'='*60}")

        attempt_start = time.perf_counter()

        energy_filter = EnergyFilter(
            ceiling, calc=energy_calc, cache=energy_cache
        )
        filter_set = FilterSet([energy_filter], use_structs=not USE_RUST)

        search_state = astar_pathfind(
            start_cnfs,
            goal_cnfs,
            heuristic=manhattan_distance,
            filter_set=filter_set,
            max_iterations=current_max_iters,
            beam_width=beam_width,
            dropout=dropout,
            verbose=(verbosity >= 2),
        )

        attempt_elapsed = time.perf_counter() - attempt_start
        num_iters = search_state.iterations

        search_params = SearchParameters(
            max_iterations=current_max_iters,
            beam_width=beam_width,
            dropout=dropout,
            greedy=False,
            heuristic="manhattan",
            filters=[{"type": "energy_ceiling", "value": ceiling}],
        )

        if search_state.path is None:
            # Failed to find path
            if verbosity >= 1:
                print(f"  No path found ({num_iters} iterations)")

            attempt = Attempt(
                path=None,
                found=False,
                iterations=num_iters,
                elapsed_seconds=attempt_elapsed,
            )

            search_result = SearchResult(
                context=context,
                parameters=search_params,
                attempts=[attempt],
                metadata={
                    "attempt": attempt_num,
                    "ceiling": ceiling,
                    "consecutive_failures": consecutive_failures + 1,
                    "num_adaptations": num_adaptations,
                }
            )
            result.results.append(search_result)
            _save_result()

            consecutive_failures += 1
            attempt_num += 1

            # Increase max_iters after each failure
            old_max_iters = current_max_iters
            current_max_iters = min(int(current_max_iters * 1.3), max_iterations)
            if current_max_iters != old_max_iters and verbosity >= 1:
                print(f"  max_iters: {old_max_iters} → {current_max_iters}")

            # Check if we need to adapt parameters
            if consecutive_failures >= 3:
                if num_adaptations >= max_adaptations:
                    # Exhausted all adaptations - terminate
                    if verbosity >= 1:
                        print(f"\n  TERMINATING: {consecutive_failures} consecutive failures "
                              f"after {num_adaptations} adaptations")
                    result.metadata["termination_reason"] = "max_adaptations_exhausted"
                    break
                else:
                    # Adapt parameters
                    num_adaptations += 1
                    old_xi, old_delta = xi, delta
                    xi = xi * xi_factor
                    delta = round(delta * delta_factor)

                    if verbosity >= 1:
                        print(f"\n  ADAPTING PARAMETERS ({num_adaptations}/{max_adaptations})")
                        print(f"    xi: {old_xi:.2f} → {xi:.2f}")
                        print(f"    delta: {old_delta} → {delta}")

                    # Regenerate CNFs with new parameters
                    if start_uc is not None and end_uc is not None:
                        start_cnfs, goal_cnfs = get_endpoint_cnfs(
                            start_uc, end_uc, xi=xi, delta=delta, min_atoms=min_atoms
                        )
                        elements = start_cnfs[0].elements
                        n_atoms = len(elements)
                        ceiling_step_ev = (ceiling_step_mev_per_atom / 1000.0) * n_atoms

                        # Clear and repopulate energy cache for new CNFs
                        energy_cache.clear()
                        for cnf in start_cnfs + goal_cnfs:
                            if cnf.coords not in energy_cache:
                                energy_cache[cnf.coords] = energy_calc.calculate_energy(cnf)

                        if verbosity >= 1:
                            print(f"    Regenerated {len(start_cnfs)} start CNFs, "
                                  f"{len(goal_cnfs)} goal CNFs")
                    else:
                        if verbosity >= 1:
                            print(f"    Warning: Cannot regenerate CNFs (start_uc/end_uc not provided)")

                    # Reset consecutive failures after adaptation
                    consecutive_failures = 0

                    # Re-estimate max_iters for new discretization
                    if verbosity >= 1:
                        print(f"    Re-estimating max_iters for new params...")
                    current_max_iters = estimate_max_iterations(
                        start_cnfs=start_cnfs,
                        goal_cnfs=goal_cnfs,
                        beam_width=beam_width,
                        verbosity=verbosity,
                    )
                    current_max_iters = min(current_max_iters, max_iterations)

            continue

        # Found a path!
        path_tuples = search_state.path
        consecutive_failures = 0

        energies = evaluate_path_energies(
            path_tuples, elements, xi, delta, energy_calc, energy_cache
        )
        barrier = path_barrier(energies)

        barrier_mev = to_mev_above_endpoint(barrier)
        barrier_mev_orig = to_mev_above_orig_endpoint(barrier)
        if verbosity >= 1:
            if barrier_mev_orig is not None:
                print(f"  PATH FOUND: len={len(path_tuples)}, barrier={barrier_mev:.1f}/{barrier_mev_orig:.1f} meV/atom disc/orig, "
                      f"iters={num_iters}")
            else:
                print(f"  PATH FOUND: len={len(path_tuples)}, barrier={barrier_mev:.1f} meV/atom, "
                      f"iters={num_iters}")

        # Check for trivial paths (start ≈ goal after discretization)
        # A path of length 1 means start was already in the goal set - this is degenerate
        if len(path_tuples) == 1:
            if verbosity >= 1:
                print(f"  WARNING: Trivial path (length 1) - start and goal are identical after discretization.")
                print(f"           This means the two structures are the same at xi={xi}, delta={delta}.")
                print(f"           Terminating ratchet - no meaningful barrier exists.")
            result.metadata["termination_reason"] = "trivial_path_start_equals_goal"
            result.metadata["trivial_path_warning"] = True
            break

        # Check if we've effectively found the optimal path (barrier ≈ 0)
        # If barrier is less than the ceiling step, we can't improve further
        if barrier_mev < ceiling_step_mev_per_atom:
            if verbosity >= 1:
                print(f"  Barrier ({barrier_mev:.1f} meV/atom) < ceiling step ({ceiling_step_mev_per_atom:.1f} meV/atom)")
                print(f"  Found optimal path - terminating ratchet.")
            result.metadata["termination_reason"] = "optimal_barrier_found"
            # Still record this path before terminating
            path_obj = Path(
                coords=[tuple(pt) for pt in path_tuples],
                energies=energies,
                barrier=barrier,
            )
            attempt = Attempt(
                path=path_obj,
                found=True,
                iterations=num_iters,
                elapsed_seconds=attempt_elapsed,
            )
            search_result = SearchResult(
                context=context,
                parameters=search_params,
                attempts=[attempt],
                metadata={
                    "attempt": attempt_num,
                    "ceiling": ceiling,
                    "consecutive_failures": 0,
                    "num_adaptations": num_adaptations,
                }
            )
            result.results.append(search_result)
            _save_result()
            break

        # Sanity check: barrier should never exceed ceiling (energy filter enforces this)
        if barrier > ceiling + 1e-6:  # small tolerance for floating point
            ceiling_mev = to_mev_above_endpoint(ceiling)
            print(f"  WARNING: barrier ({barrier_mev:.1f} meV/atom) > ceiling ({ceiling_mev:.1f} meV/atom)")
            print(f"           This should not happen if the energy filter is working correctly.")
            # Find which path nodes exceed the ceiling
            over_ceiling = [(i, e) for i, e in enumerate(energies) if e > ceiling]
            if over_ceiling:
                print(f"           {len(over_ceiling)} nodes exceed ceiling:")
                for idx, e in over_ceiling[:5]:  # show first 5
                    e_mev = to_mev_above_endpoint(e)
                    print(f"             node {idx}: {e_mev:.1f} meV/atom")

        path_obj = Path(
            coords=[tuple(pt) for pt in path_tuples],
            energies=energies,
            barrier=barrier,
        )

        attempt = Attempt(
            path=path_obj,
            found=True,
            iterations=num_iters,
            elapsed_seconds=attempt_elapsed,
        )

        search_result = SearchResult(
            context=context,
            parameters=search_params,
            attempts=[attempt],
            metadata={
                "attempt": attempt_num,
                "ceiling": ceiling,
                "consecutive_failures": 0,
                "num_adaptations": num_adaptations,
            }
        )
        result.results.append(search_result)
        _save_result()

        # Lower the ceiling
        old_ceiling = ceiling
        # New ceiling is the lower of: barrier, or ceiling - step
        new_ceiling_from_step = ceiling - ceiling_step_ev
        ceiling = min(barrier, new_ceiling_from_step)

        old_ceiling_mev = to_mev_above_endpoint(old_ceiling)
        new_ceiling_mev = to_mev_above_endpoint(ceiling)
        old_ceiling_mev_orig = to_mev_above_orig_endpoint(old_ceiling)
        new_ceiling_mev_orig = to_mev_above_orig_endpoint(ceiling)
        if verbosity >= 1:
            if old_ceiling_mev_orig is not None:
                print(f"  Ceiling: {old_ceiling_mev:.1f}/{old_ceiling_mev_orig:.1f} → {new_ceiling_mev:.1f}/{new_ceiling_mev_orig:.1f} meV/atom disc/orig")
            else:
                print(f"  Ceiling: {old_ceiling_mev:.1f} → {new_ceiling_mev:.1f} meV/atom")

        # Update max_iters based on successful search
        new_max_iters = max(current_max_iters, int(num_iters * 1.2))
        new_max_iters = min(new_max_iters, max_iterations)
        if new_max_iters != current_max_iters:
            if verbosity >= 1:
                print(f"  max_iters: {current_max_iters} → {new_max_iters}")
            current_max_iters = new_max_iters

        attempt_num += 1

    total_elapsed = time.perf_counter() - total_start
    result.metadata["total_elapsed_seconds"] = total_elapsed
    result.metadata["final_ceiling"] = ceiling
    result.metadata["num_adaptations"] = num_adaptations
    result.metadata["total_attempts"] = attempt_num

    if verbosity >= 1:
        best = result.best_path
        print(f"\n{'='*60}")
        print(f"Final result:")
        if best:
            best_mev = to_mev_above_endpoint(best.barrier)
            best_mev_orig = to_mev_above_orig_endpoint(best.barrier)
            if best_mev_orig is not None:
                print(f"  Barrier: {best_mev:.1f}/{best_mev_orig:.1f} meV/atom disc/orig")
            else:
                print(f"  Barrier: {best_mev:.1f} meV/atom above endpoint")
            print(f"  Path length: {len(best)} steps")
        else:
            print(f"  No path found")
        print(f"  Total attempts: {attempt_num}")
        print(f"  Parameter adaptations: {num_adaptations}")
        print(f"  Energy cache size: {len(energy_cache)}")
        print(f"  Total time: {total_elapsed:.1f}s")
        print(f"{'='*60}")

    _save_result()

    return result
