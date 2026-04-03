"""Jobflow job for parallel ratchet barrier search.

Runs multiple independent ratchet processes in parallel, each starting from
a different energy ceiling. This is a standalone search that handles endpoint
relaxation internally.
"""

from jobflow import job
from pymatgen.core import Structure

from cnf import UnitCell
from cnf.navigation.astar.iterative.ratchet.parallel_ratchet import (
    ParallelRatchetResult,
    parallel_ratchet,
)


@job
def parallel_ratchet_job(
    start_structure: Structure,
    end_structure: Structure,
    min_ceiling_mev_per_atom: float = 100.0,
    max_ceiling_mev_per_atom: float = 2000.0,
    n_workers: int = 32,
    xi: float = 1.5,
    delta: int | None = None,
    atom_step_length: float | None = None,
    min_atoms: int | None = None,
    max_adaptations: int = 5,
    xi_factor: float = 0.8,
    delta_factor: float = 1.2,
    ceiling_step_mev_per_atom: float = 2.0,
    dropout: float = 0.1,
    max_iterations: int | None = None,
    initial_max_iters: int | None = None,
    beam_width: int = 1000,
    grace_model_path: str | None = None,
    output_dir: str | None = None,
) -> ParallelRatchetResult:
    """Run parallel ratchet barrier search.

    Runs N independent ratchet processes in parallel, each starting at a
    different energy ceiling. Each worker can adapt its xi/delta parameters
    independently, allowing exploration of the barrier landscape at multiple
    resolution levels simultaneously.

    This is a standalone search that:
    1. Relaxes endpoint structures
    2. Estimates max_iterations via plain A* (if not provided)
    3. Spawns N parallel ratchet workers at linearly-spaced ceiling levels
    4. Each worker ratchets down its ceiling when paths are found
    5. Returns the best barrier found across all workers

    Args:
        start_structure: Starting crystal structure.
        end_structure: Ending crystal structure.
        min_ceiling_mev_per_atom: Lowest starting ceiling (meV/atom above endpoint).
        max_ceiling_mev_per_atom: Highest starting ceiling (meV/atom above endpoint).
        n_workers: Number of parallel ratchet processes.
        xi: Initial lattice discretization parameter.
        delta: Initial motif discretization parameter.
        atom_step_length: Target step size in Angstroms (alternative to delta).
        min_atoms: Minimum atoms for supercell.
        max_adaptations: Max xi/delta adaptations per worker.
        xi_factor: xi multiplied by this on each adaptation (< 1 = finer).
        delta_factor: delta multiplied by this on each adaptation (> 1 = finer).
        ceiling_step_mev_per_atom: Minimum ceiling reduction per step.
        dropout: Neighbor dropout probability.
        max_iterations: Max A* iterations per search (absolute cap). If None,
            defaults to 10000.
        initial_max_iters: Starting max iterations. If None, estimates via
            plain A* search. If provided, skips estimation and uses this value.
        beam_width: Max open-set size for beam search.
        grace_model_path: Path to local GRACE model (uses foundation model if None).
        output_dir: Base directory for outputs.

    Returns:
        ParallelRatchetResult containing all worker results and best barrier.
    """
    from cnf.calculation.grace import GraceCalcProvider
    from cnf.navigation.utils import compute_delta_for_endpoints

    start_uc = UnitCell.from_pymatgen_structure(start_structure)
    end_uc = UnitCell.from_pymatgen_structure(end_structure)

    calc_provider = GraceCalcProvider(model_path=grace_model_path)

    # Compute delta if not provided
    if delta is None:
        if atom_step_length is None:
            raise ValueError("Must provide either delta or atom_step_length")
        delta = compute_delta_for_endpoints(start_uc, end_uc, atom_step_length, min_atoms)

    # If max_iterations not provided, use a high cap and let ratchet estimate internally
    if max_iterations is None:
        max_iterations = 10_000

    return parallel_ratchet(
        start_uc=start_uc,
        end_uc=end_uc,
        calc_provider=calc_provider,
        min_ceiling_mev_per_atom=min_ceiling_mev_per_atom,
        max_ceiling_mev_per_atom=max_ceiling_mev_per_atom,
        n_workers=n_workers,
        xi=xi,
        delta=delta,
        atom_step_length=None,  # Already computed delta
        min_atoms=min_atoms,
        max_adaptations=max_adaptations,
        xi_factor=xi_factor,
        delta_factor=delta_factor,
        ceiling_step_mev_per_atom=ceiling_step_mev_per_atom,
        dropout=dropout,
        max_iterations=max_iterations,
        initial_max_iters=initial_max_iters,
        beam_width=beam_width,
        verbosity=1,
        output_dir=output_dir,
    )


# Convenience alias with capitalized name for jobflow convention
ParallelRatchetJob = parallel_ratchet_job
