"""Jobflow jobs and makers for CNF barrier search workflow.

This module provides jobflow-compatible jobs for the 4-phase barrier search:
1. SearchJob - Parameter search to find optimal xi/delta/min_distance
2. SampleJob - Path sampling to discover initial energy ceiling
3. SweepJob - Parallel ceiling sweep
4. RatchetJob - Serial barrier refinement

These can be run individually or combined using BarrierSearchMaker.
"""

from pathlib import Path

from jobflow import job
from pymatgen.core import Structure

from cnf import UnitCell
from cnf.navigation.astar.models import (
    ParameterSearchResult,
    SearchResult,
    CeilingSweepResult,
    RefinementResult,
)


@job
def search_job(
    start_structure: Structure,
    end_structure: Structure,
    xi_values: list[float] | None = None,
    atom_step_lengths: list[float] | None = None,
    min_dist_low: float = 0.5,
    min_dist_high: float = 2.0,
    tolerance: float = 0.05,
    max_iterations: int = 5_000,
    beam_width: int = 1000,
    dropout: float = 0.3,
    n_workers: int = 0,
    output_dir: str | None = None,
) -> ParameterSearchResult:
    """Phase 1: Search for optimal discretization parameters.

    Args:
        start_structure: Starting crystal structure.
        end_structure: Ending crystal structure.
        xi_values: Lattice discretization values to try.
        atom_step_lengths: Target atom step sizes in Angstroms.
        min_dist_low: Lower bound for min_distance binary search.
        min_dist_high: Upper bound for min_distance binary search.
        tolerance: Binary search convergence tolerance.
        max_iterations: Max A* iterations per search.
        beam_width: Max open-set size for beam search.
        dropout: Neighbor dropout probability.
        n_workers: Number of parallel workers (0=auto, uses CPU count).
        output_dir: Directory for output files.

    Returns:
        ParameterSearchResult with recommended parameters.
    """
    from cnf.navigation.astar.iterative import search

    start_uc = UnitCell.from_pymatgen_structure(start_structure)
    end_uc = UnitCell.from_pymatgen_structure(end_structure)

    return search(
        start_uc=start_uc,
        end_uc=end_uc,
        xi_values=xi_values,
        atom_step_lengths=atom_step_lengths,
        min_dist_low=min_dist_low,
        min_dist_high=min_dist_high,
        tolerance=tolerance,
        max_iterations=max_iterations,
        beam_width=beam_width,
        dropout=dropout,
        n_workers=n_workers,
        verbosity=1,
        output_dir=output_dir,
    )


@job
def sample_job(
    start_structure: Structure,
    end_structure: Structure,
    xi: float,
    delta: int,
    min_distance: float | None = None,
    num_samples: int = 20,
    dropout_range: tuple[float, float] = (0.3, 0.7),
    max_iterations: int = 5_000,
    beam_width: int = 1000,
    n_workers: int = 0,
    grace_model_path: str | None = None,
    output_dir: str | None = None,
) -> SearchResult:
    """Phase 2: Sample diverse paths to discover initial energy ceiling.

    Args:
        start_structure: Starting crystal structure.
        end_structure: Ending crystal structure.
        xi: Lattice discretization parameter.
        delta: Motif discretization parameter.
        min_distance: Optional minimum interatomic distance filter.
        num_samples: Number of pathfinding attempts.
        dropout_range: (min, max) dropout probability range.
        max_iterations: Max A* iterations per attempt.
        beam_width: Max open-set size for beam search.
        n_workers: Number of parallel workers (0=auto, uses CPU count).
        grace_model_path: Path to local GRACE model (uses foundation model if None).
        output_dir: Directory for output files.

    Returns:
        SearchResult containing sampled paths and barriers.
    """
    from cnf.navigation.astar.iterative import sample
    from cnf.navigation.endpoints import get_endpoint_cnfs
    from cnf.calculation.grace import GraceCalcProvider

    start_uc = UnitCell.from_pymatgen_structure(start_structure)
    end_uc = UnitCell.from_pymatgen_structure(end_structure)

    start_cnfs, goal_cnfs = get_endpoint_cnfs(start_uc, end_uc, xi=xi, delta=delta)

    return sample(
        start_cnfs=start_cnfs,
        goal_cnfs=goal_cnfs,
        calc_provider=GraceCalcProvider(model_path=grace_model_path),
        num_samples=num_samples,
        dropout_range=dropout_range,
        min_distance=min_distance,
        max_iterations=max_iterations,
        beam_width=beam_width,
        n_workers=n_workers,
        verbosity=1,
        output_dir=output_dir,
    )


@job
def sweep_job(
    start_structure: Structure,
    end_structure: Structure,
    max_ceiling: float,
    xi: float,
    delta: int,
    num_ceilings: int = 5,
    attempts_per_ceiling: int = 1,
    max_passes: int = 3,
    xi_factor: float = 0.9,
    delta_factor: float = 1.1,
    dropout: float = 0.1,
    beam_width: int = 1000,
    n_workers: int = 0,
    grace_model_path: str | None = None,
    output_dir: str | None = None,
) -> CeilingSweepResult:
    """Phase 3: Parallel ceiling sweep with multi-resolution refinement.

    Args:
        start_structure: Starting crystal structure.
        end_structure: Ending crystal structure.
        max_ceiling: Upper bound of sweep range (eV).
        xi: Initial lattice discretization parameter.
        delta: Initial motif discretization parameter.
        num_ceilings: Number of ceiling levels to sweep per pass.
        attempts_per_ceiling: Searches per ceiling level.
        max_passes: Number of resolution refinement passes.
        xi_factor: xi multiplied by this each refinement pass.
        delta_factor: delta multiplied by this each refinement pass.
        dropout: Neighbor dropout probability.
        beam_width: Max open-set size for beam search.
        n_workers: Number of parallel workers (0=auto, uses CPU count).
        grace_model_path: Path to local GRACE model (uses foundation model if None).
        output_dir: Directory for output files.

    Returns:
        CeilingSweepResult containing all sweep results.
    """
    from cnf.navigation.astar.iterative import sweep
    from cnf.calculation.grace import GraceCalcProvider

    start_uc = UnitCell.from_pymatgen_structure(start_structure)
    end_uc = UnitCell.from_pymatgen_structure(end_structure)

    return sweep(
        start_uc=start_uc,
        end_uc=end_uc,
        max_ceiling=max_ceiling,
        calc_provider=GraceCalcProvider(model_path=grace_model_path),
        xi=xi,
        delta=delta,
        num_ceilings=num_ceilings,
        attempts_per_ceiling=attempts_per_ceiling,
        max_passes=max_passes,
        xi_factor=xi_factor,
        delta_factor=delta_factor,
        dropout=dropout,
        beam_width=beam_width,
        n_workers=n_workers,
        verbosity=1,
        output_dir=output_dir,
    )


@job
def ratchet_job(
    start_structure: Structure,
    end_structure: Structure,
    initial_ceiling: float,
    xi: float,
    delta: int,
    paths_per_round: int = 10,
    max_rounds: int = 20,
    dropout: float = 0.3,
    min_dropout: float = 0.1,
    max_iterations: int = 100_000,
    beam_width: int = 1000,
    grace_model_path: str | None = None,
    output_dir: str | None = None,
) -> RefinementResult:
    """Phase 4: Serial barrier refinement with ratcheting ceiling.

    Args:
        start_structure: Starting crystal structure.
        end_structure: Ending crystal structure.
        initial_ceiling: Starting energy ceiling (eV).
        xi: Lattice discretization parameter.
        delta: Motif discretization parameter.
        paths_per_round: Number of paths to find per round.
        max_rounds: Maximum refinement rounds.
        dropout: Initial neighbor dropout probability.
        min_dropout: Minimum dropout for adaptive adjustment.
        max_iterations: Max A* iterations per search.
        beam_width: Max open-set size for beam search.
        grace_model_path: Path to local GRACE model (uses foundation model if None).
        output_dir: Directory for output files.

    Returns:
        RefinementResult containing all refinement rounds.
    """
    from cnf.navigation.astar.iterative import ratchet
    from cnf.navigation.endpoints import get_endpoint_cnfs
    from cnf.calculation.grace import GraceCalculator

    start_uc = UnitCell.from_pymatgen_structure(start_structure)
    end_uc = UnitCell.from_pymatgen_structure(end_structure)

    start_cnfs, goal_cnfs = get_endpoint_cnfs(start_uc, end_uc, xi=xi, delta=delta)

    return ratchet(
        start_cnfs=start_cnfs,
        goal_cnfs=goal_cnfs,
        initial_ceiling=initial_ceiling,
        energy_calc=GraceCalculator(model_path=grace_model_path),
        paths_per_round=paths_per_round,
        max_rounds=max_rounds,
        dropout=dropout,
        min_dropout=min_dropout,
        max_iterations=max_iterations,
        beam_width=beam_width,
        verbosity=1,
        output_dir=output_dir,
    )


# Convenience aliases with capitalized names for jobflow convention
SearchJob = search_job
SampleJob = sample_job
SweepJob = sweep_job
RatchetJob = ratchet_job


@job
def barrier_search_job(
    start_structure: Structure,
    end_structure: Structure,
    xi: float | None = None,
    delta: int | None = None,
    min_distance: float | None = None,
    num_samples: int = 20,
    num_ceilings: int | None = None,
    max_rounds: int = 20,
    beam_width: int = 1000,
    n_workers: int = 0,
    grace_model_path: str | None = None,
    output_dir: str | None = None,
) -> RefinementResult:
    """Full 4-phase barrier search as a single job.

    Runs all phases sequentially in a single HPC job:
    1. Parameter search (if xi/delta not provided)
    2. Path sampling to find initial ceiling
    3. Ceiling sweep to find good paths
    4. Ratchet refinement to optimize barrier

    Each phase writes results to disk in output_dir, so progress is preserved
    even if the job is interrupted.

    Args:
        start_structure: Starting crystal structure.
        end_structure: Ending crystal structure.
        xi: Lattice discretization (if None, runs parameter search).
        delta: Motif discretization (if None, runs parameter search).
        min_distance: Minimum interatomic distance filter.
        num_samples: Number of paths to sample in Phase 2.
        num_ceilings: Number of ceiling levels in Phase 3 (defaults to n_workers).
        max_rounds: Maximum refinement rounds in Phase 4.
        beam_width: Beam width for all searches.
        n_workers: Number of parallel workers (0=auto, uses CPU count).
        grace_model_path: Path to local GRACE model (uses foundation model if None).
        output_dir: Base directory for outputs.

    Returns:
        RefinementResult from Phase 4 containing the final optimized barrier.
    """
    import os
    from cnf.navigation.astar.iterative import search, sample, sweep, ratchet
    from cnf.navigation.endpoints import get_endpoint_cnfs
    from cnf.calculation.grace import GraceCalculator, GraceCalcProvider

    start_uc = UnitCell.from_pymatgen_structure(start_structure)
    end_uc = UnitCell.from_pymatgen_structure(end_structure)

    output_base = Path(output_dir) if output_dir else None

    # Resolve n_workers
    if n_workers == 0:
        n_workers = os.cpu_count() or 1

    # Default num_ceilings to n_workers
    if num_ceilings is None:
        num_ceilings = n_workers

    # Phase 1: Parameter search (if needed)
    if xi is None or delta is None:
        print("="*60)
        print("Phase 1: Parameter Search")
        print("="*60)

        phase1_dir = str(output_base / "phase1_search") if output_base else None
        search_result = search(
            start_uc=start_uc,
            end_uc=end_uc,
            n_workers=n_workers,
            verbosity=1,
            output_dir=phase1_dir,
        )
        xi = search_result.recommended_xi
        delta = search_result.recommended_delta
        min_distance = search_result.recommended_min_distance

        print(f"Recommended: xi={xi}, delta={delta}, min_distance={min_distance}")

    # Get endpoint CNFs for phases 2-4
    start_cnfs, goal_cnfs = get_endpoint_cnfs(start_uc, end_uc, xi=xi, delta=delta)

    # Create calc_provider for all phases
    calc_provider = GraceCalcProvider(model_path=grace_model_path)

    # Phase 2: Sampling
    print("\n" + "="*60)
    print("Phase 2: Path Sampling")
    print("="*60)

    phase2_dir = str(output_base / "phase2_sample") if output_base else None
    sample_result = sample(
        start_cnfs=start_cnfs,
        goal_cnfs=goal_cnfs,
        calc_provider=calc_provider,
        num_samples=num_samples,
        min_distance=min_distance,
        beam_width=beam_width,
        n_workers=n_workers,
        verbosity=1,
        output_dir=phase2_dir,
    )

    if sample_result.best_barrier is None:
        raise RuntimeError("Phase 2 failed: no paths found during sampling")

    initial_ceiling = sample_result.best_barrier
    print(f"Initial ceiling from sampling: {initial_ceiling:.4f} eV")

    # Phase 3: Sweep
    print("\n" + "="*60)
    print("Phase 3: Ceiling Sweep")
    print("="*60)

    phase3_dir = str(output_base / "phase3_sweep") if output_base else None
    sweep_result = sweep(
        start_uc=start_uc,
        end_uc=end_uc,
        max_ceiling=initial_ceiling,
        calc_provider=calc_provider,
        xi=xi,
        delta=delta,
        num_ceilings=num_ceilings,
        beam_width=beam_width,
        n_workers=n_workers,
        verbosity=1,
        output_dir=phase3_dir,
    )

    if sweep_result.best_barrier is None:
        raise RuntimeError("Phase 3 failed: no paths found during sweep")

    sweep_ceiling = sweep_result.best_barrier
    print(f"Ceiling after sweep: {sweep_ceiling:.4f} eV")

    # Phase 4: Ratchet
    print("\n" + "="*60)
    print("Phase 4: Ratchet Refinement")
    print("="*60)

    phase4_dir = str(output_base / "phase4_ratchet") if output_base else None
    ratchet_result = ratchet(
        start_cnfs=start_cnfs,
        goal_cnfs=goal_cnfs,
        initial_ceiling=sweep_ceiling,
        energy_calc=GraceCalculator(model_path=grace_model_path),
        paths_per_round=10,
        max_rounds=max_rounds,
        beam_width=beam_width,
        verbosity=1,
        output_dir=phase4_dir,
    )

    print("\n" + "="*60)
    print("Barrier Search Complete")
    print("="*60)
    if ratchet_result.best_barrier is not None:
        print(f"Final barrier: {ratchet_result.best_barrier:.4f} eV")
    else:
        print("Warning: No paths found in final refinement")

    return ratchet_result


# Convenience alias
BarrierSearchJob = barrier_search_job
