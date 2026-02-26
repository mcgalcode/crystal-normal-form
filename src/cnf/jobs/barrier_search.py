"""Jobflow jobs and makers for CNF barrier search workflow.

This module provides jobflow-compatible jobs for the 4-phase barrier search:
1. SearchJob - Parameter search to find optimal xi/delta/min_distance
2. SampleJob - Path sampling to discover initial energy ceiling
3. SweepJob - Parallel ceiling sweep
4. RatchetJob - Serial barrier refinement

These can be run individually or combined using BarrierSearchMaker.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jobflow import Flow, Maker, job, Response
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
        n_workers=1,  # HPC handles parallelism
        verbose=True,
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
        grace_model_path: Path to local GRACE model (uses foundation model if None).
        output_dir: Directory for output files.

    Returns:
        SearchResult containing sampled paths and barriers.
    """
    from cnf.navigation.astar.iterative import sample
    from cnf.navigation.endpoints import get_endpoint_cnfs
    from cnf.calculation.grace import GraceCalculator

    start_uc = UnitCell.from_pymatgen_structure(start_structure)
    end_uc = UnitCell.from_pymatgen_structure(end_structure)

    start_cnfs, goal_cnfs = get_endpoint_cnfs(start_uc, end_uc, xi=xi, delta=delta)

    return sample(
        start_cnfs=start_cnfs,
        goal_cnfs=goal_cnfs,
        energy_calc=GraceCalculator(model_path=grace_model_path),
        num_samples=num_samples,
        dropout_range=dropout_range,
        min_distance=min_distance,
        max_iterations=max_iterations,
        beam_width=beam_width,
        verbose=True,
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
        grace_model_path: Path to local GRACE model (uses foundation model if None).
        output_dir: Directory for output files.

    Returns:
        CeilingSweepResult containing all sweep results.
    """
    from cnf.navigation.astar.iterative import sweep
    from cnf.calculation.grace import GraceCalculator

    start_uc = UnitCell.from_pymatgen_structure(start_structure)
    end_uc = UnitCell.from_pymatgen_structure(end_structure)

    return sweep(
        start_uc=start_uc,
        end_uc=end_uc,
        max_ceiling=max_ceiling,
        energy_calc=GraceCalculator(model_path=grace_model_path),
        xi=xi,
        delta=delta,
        num_ceilings=num_ceilings,
        attempts_per_ceiling=attempts_per_ceiling,
        max_passes=max_passes,
        xi_factor=xi_factor,
        delta_factor=delta_factor,
        dropout=dropout,
        beam_width=beam_width,
        n_workers=1,  # HPC handles parallelism
        verbose=True,
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
        verbose=True,
        output_dir=output_dir,
    )


# Convenience aliases with capitalized names for jobflow convention
SearchJob = search_job
SampleJob = sample_job
SweepJob = sweep_job
RatchetJob = ratchet_job


@dataclass
class BarrierSearchMaker(Maker):
    """Maker for the full 4-phase barrier search workflow.

    This creates a jobflow Flow that runs:
    1. Parameter search (optional, if xi/delta not provided)
    2. Path sampling to find initial ceiling
    3. Ceiling sweep to find good paths
    4. Ratchet refinement to optimize barrier

    Attributes:
        name: Name for the flow.
        xi: Lattice discretization (if None, runs parameter search).
        delta: Motif discretization (if None, runs parameter search).
        min_distance: Minimum interatomic distance filter.
        num_samples: Number of paths to sample in Phase 2.
        num_ceilings: Number of ceiling levels in Phase 3.
        max_rounds: Maximum refinement rounds in Phase 4.
        beam_width: Beam width for all searches.
        grace_model_path: Path to local GRACE model (uses foundation model if None).
        output_dir: Base directory for outputs.
    """

    name: str = "barrier_search"
    xi: float | None = None
    delta: int | None = None
    min_distance: float | None = None
    num_samples: int = 20
    num_ceilings: int = 5
    max_rounds: int = 20
    beam_width: int = 1000
    grace_model_path: str | None = None
    output_dir: str | None = None

    @job
    def make(
        self,
        start_structure: Structure,
        end_structure: Structure,
    ) -> Response:
        """Create the barrier search flow.

        Args:
            start_structure: Starting crystal structure.
            end_structure: Ending crystal structure.

        Returns:
            Response with detour to the barrier search flow.
        """
        jobs = []
        output_base = Path(self.output_dir) if self.output_dir else None

        # Phase 1: Parameter search (if needed)
        if self.xi is None or self.delta is None:
            phase1_dir = str(output_base / "phase1_search") if output_base else None
            phase1 = search_job(
                start_structure=start_structure,
                end_structure=end_structure,
                output_dir=phase1_dir,
            )
            phase1.name = f"{self.name} - Phase 1 Search"
            jobs.append(phase1)

            # Extract parameters from Phase 1
            xi = phase1.output.recommended_xi
            delta = phase1.output.recommended_delta
            min_distance = phase1.output.recommended_min_distance
        else:
            xi = self.xi
            delta = self.delta
            min_distance = self.min_distance

        # Phase 2: Sampling
        phase2_dir = str(output_base / "phase2_sample") if output_base else None
        phase2 = sample_job(
            start_structure=start_structure,
            end_structure=end_structure,
            xi=xi,
            delta=delta,
            min_distance=min_distance,
            num_samples=self.num_samples,
            beam_width=self.beam_width,
            grace_model_path=self.grace_model_path,
            output_dir=phase2_dir,
        )
        phase2.name = f"{self.name} - Phase 2 Sample"
        jobs.append(phase2)

        # Phase 3: Sweep (use best barrier from Phase 2 as ceiling)
        phase3_dir = str(output_base / "phase3_sweep") if output_base else None
        phase3 = sweep_job(
            start_structure=start_structure,
            end_structure=end_structure,
            max_ceiling=phase2.output.best_barrier,
            xi=xi,
            delta=delta,
            num_ceilings=self.num_ceilings,
            beam_width=self.beam_width,
            grace_model_path=self.grace_model_path,
            output_dir=phase3_dir,
        )
        phase3.name = f"{self.name} - Phase 3 Sweep"
        jobs.append(phase3)

        # Phase 4: Ratchet (use best barrier from Phase 3 as initial ceiling)
        phase4_dir = str(output_base / "phase4_ratchet") if output_base else None
        phase4 = ratchet_job(
            start_structure=start_structure,
            end_structure=end_structure,
            initial_ceiling=phase3.output.best_barrier,
            xi=xi,
            delta=delta,
            max_rounds=self.max_rounds,
            beam_width=self.beam_width,
            grace_model_path=self.grace_model_path,
            output_dir=phase4_dir,
        )
        phase4.name = f"{self.name} - Phase 4 Ratchet"
        jobs.append(phase4)

        flow = Flow(jobs, output=phase4.output, name=self.name)
        return Response(detour=flow)
