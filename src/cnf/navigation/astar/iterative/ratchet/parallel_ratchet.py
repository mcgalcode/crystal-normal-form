"""Parallel ratchet barrier search.

Runs multiple independent ratchet processes in parallel, each starting from
a different energy ceiling. Each ratchet process can adapt its xi/delta
parameters independently, allowing exploration of the barrier landscape
at multiple resolution levels simultaneously.

This is more powerful than ceiling_sweep because:
- Each worker can adapt parameters independently
- Each worker ratchets down its ceiling when paths are found
- Workers explore different regions of parameter space in parallel

Output structure:
    output_dir/
        parallel_ratchet_result.json  # Overall summary
        worker_00/
            refinement_result.json    # Full ratchet trace for worker 0
        worker_01/
            refinement_result.json    # Full ratchet trace for worker 1
        ...
"""

import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path as PathlibPath
from typing import Any, Callable

from monty.json import MSONable

from cnf import UnitCell
from cnf.navigation.astar.models import RefinementResult
from cnf.navigation.astar.iterative.core import worker as core_worker


@dataclass
class ParallelRatchetResult(MSONable):
    """Result from parallel ratchet search."""
    worker_results: list[RefinementResult]
    best_barrier: float | None
    best_barrier_mev_per_atom: float | None
    n_atoms: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def best_result(self) -> RefinementResult | None:
        """Get the RefinementResult with the lowest barrier."""
        results_with_barriers = [
            r for r in self.worker_results
            if r.best_barrier is not None
        ]
        if not results_with_barriers:
            return None
        return min(results_with_barriers, key=lambda r: r.best_barrier)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            '@module': self.__class__.__module__,
            '@class': self.__class__.__name__,
            'worker_results': [r.to_dict() for r in self.worker_results],
            'best_barrier': self.best_barrier,
            'best_barrier_mev_per_atom': self.best_barrier_mev_per_atom,
            'n_atoms': self.n_atoms,
            'metadata': self.metadata,
        }

    def as_dict(self) -> dict:
        """Alias for to_dict() for MSONable compatibility."""
        return self.to_dict()

    @classmethod
    def from_dict(cls, d: dict) -> 'ParallelRatchetResult':
        """Deserialize from a dictionary."""
        return cls(
            worker_results=[RefinementResult.from_dict(r) for r in d['worker_results']],
            best_barrier=d.get('best_barrier'),
            best_barrier_mev_per_atom=d.get('best_barrier_mev_per_atom'),
            n_atoms=d['n_atoms'],
            metadata=d.get('metadata', {}),
        )

    def to_json(self, path: str) -> None:
        """Save result to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'ParallelRatchetResult':
        """Load from a JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


def init_ratchet_worker(calc_provider, tf_threads=None):
    """Initialize a parallel ratchet worker. Wraps core init_worker."""
    core_worker.init_worker(calc_provider, tf_threads, phase_name="Ratchet")


def _run_ratchet_worker(
    worker_id: int,
    worker_ceiling: float,
    ceiling_mev_per_atom: float,
    start_structure_dict: dict,
    end_structure_dict: dict,
    xi: float,
    delta: int,
    min_atoms: int | None,
    max_adaptations: int,
    xi_factor: float,
    delta_factor: float,
    ceiling_step_mev_per_atom: float,
    dropout: float,
    max_iterations: int,
    initial_max_iters: int | None,
    beam_width: int,
    output_dir: str | None,
) -> RefinementResult:
    """Worker function that runs a single ratchet process.

    Each worker produces its own refinement_result.json and worker.log file.
    Uses the calculator initialized by init_ratchet_worker.
    """
    import sys
    from pymatgen.core import Structure
    from cnf import UnitCell
    from cnf.navigation.endpoints import get_endpoint_cnfs
    from cnf.navigation.astar.iterative.ratchet.ratchet import ratchet

    # Get calculator from worker globals (set by init_ratchet_worker)
    calc = core_worker.worker_calc

    # Reconstruct structures from dicts
    start_structure = Structure.from_dict(start_structure_dict)
    end_structure = Structure.from_dict(end_structure_dict)
    start_uc = UnitCell.from_pymatgen_structure(start_structure)
    end_uc = UnitCell.from_pymatgen_structure(end_structure)

    # Get endpoint CNFs
    start_cnfs, goal_cnfs = get_endpoint_cnfs(
        start_uc, end_uc, xi=xi, delta=delta, min_atoms=min_atoms
    )

    # Create worker-specific output directory
    worker_output_dir = None
    output_file = None
    log_file = None
    if output_dir is not None:
        worker_output_dir = PathlibPath(output_dir) / f"worker_{worker_id:02d}"
        worker_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(worker_output_dir / "refinement_result.json")
        log_file = worker_output_dir / "worker.log"

    # Redirect stdout to log file if output_dir is set
    original_stdout = sys.stdout
    log_handle = None
    if log_file is not None:
        log_handle = open(log_file, "w", buffering=1)  # Line buffered
        sys.stdout = log_handle
        print(f"=== Worker {worker_id} started (ceiling={ceiling_mev_per_atom:.0f} meV/atom) ===")
        sys.stdout.flush()

    try:
        # Run ratchet (verbosity=1 so we get per-worker logs)
        result = ratchet(
            start_cnfs=start_cnfs,
            goal_cnfs=goal_cnfs,
            initial_ceiling=worker_ceiling,
            energy_calc=calc,
            start_uc=start_uc,
            end_uc=end_uc,
            min_atoms=min_atoms,
            max_adaptations=max_adaptations,
            xi_factor=xi_factor,
            delta_factor=delta_factor,
            ceiling_step_mev_per_atom=ceiling_step_mev_per_atom,
            dropout=dropout,
            max_iterations=max_iterations,
            initial_max_iters=initial_max_iters,
            beam_width=beam_width,
            verbosity=1,  # Each worker logs its progress
            output_dir=worker_output_dir,
        )
        print(f"=== Worker {worker_id} finished ===")
        sys.stdout.flush()
    finally:
        # Restore stdout
        sys.stdout = original_stdout
        if log_handle is not None:
            log_handle.close()

    # Add worker metadata
    result.metadata["worker_id"] = worker_id
    result.metadata["ceiling_mev_per_atom"] = ceiling_mev_per_atom
    result.metadata["output_file"] = output_file

    return result


def parallel_ratchet(
    start_uc: UnitCell,
    end_uc: UnitCell,
    calc_provider: Callable,
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
    max_iterations: int = 100_000,
    initial_max_iters: int | None = None,
    beam_width: int = 1000,
    verbosity: int = 1,
    output_dir: str | None = None,
) -> ParallelRatchetResult:
    """Run multiple ratchet processes in parallel at different ceiling levels.

    Each worker starts at a different energy ceiling and runs an independent
    ratchet process that can adapt its xi/delta parameters as needed.

    Output files:
        - parallel_ratchet_result.json: Overall summary with best barrier
        - worker_XX/refinement_result.json: Full trace for each worker

    Args:
        start_uc: Starting UnitCell.
        end_uc: Ending UnitCell.
        calc_provider: Callable that returns an energy calculator.
        min_ceiling_mev_per_atom: Lowest starting ceiling (meV/atom above endpoint).
        max_ceiling_mev_per_atom: Highest starting ceiling (meV/atom above endpoint).
        n_workers: Number of parallel ratchet processes.
        xi: Initial lattice discretization parameter.
        delta: Initial motif discretization parameter.
        atom_step_length: Target step size in Angstroms (alternative to delta).
        min_atoms: Minimum atoms for supercell.
        max_adaptations: Max xi/delta adaptations per worker.
        xi_factor: xi multiplied by this on each adaptation.
        delta_factor: delta multiplied by this on each adaptation.
        ceiling_step_mev_per_atom: Minimum ceiling reduction per step.
        dropout: Neighbor dropout probability.
        max_iterations: Max A* iterations per search.
        beam_width: Max open-set size for beam search.
        verbosity: 0=silent, 1=progress, 2+=detailed.
        output_dir: Base directory for outputs.

    Returns:
        ParallelRatchetResult containing all worker results and best barrier.
    """
    from cnf.navigation.utils import compute_delta_for_endpoints
    from cnf.navigation.endpoints import get_endpoint_cnfs
    from cnf.calculation.relaxation import relax_unit_cell

    total_start = time.perf_counter()

    if output_dir is not None:
        output_dir = PathlibPath(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Compute delta from atom_step_length if not provided
    if delta is None:
        if atom_step_length is None:
            raise ValueError("Must provide either delta or atom_step_length")
        delta = compute_delta_for_endpoints(start_uc, end_uc, atom_step_length, min_atoms)

    # Relax endpoints
    if verbosity >= 1:
        print("=" * 60)
        print("Preprocessing: Relaxing Endpoints")
        print("=" * 60)

    calc = calc_provider()
    start_uc = relax_unit_cell(start_uc, calc._calc, verbose=(verbosity >= 1), label="start")
    end_uc = relax_unit_cell(end_uc, calc._calc, verbose=(verbosity >= 1), label="end")

    if output_dir:
        start_uc.to_cif(str(output_dir / "start_relaxed.cif"))
        end_uc.to_cif(str(output_dir / "end_relaxed.cif"))

    # Get endpoint CNFs to compute reference energies
    start_cnfs, goal_cnfs = get_endpoint_cnfs(
        start_uc, end_uc, xi=xi, delta=delta, min_atoms=min_atoms
    )
    n_atoms = len(start_cnfs[0].elements)

    # Compute endpoint energies
    endpoint_energies = []
    for cnf in start_cnfs + goal_cnfs:
        e = calc.calculate_energy(cnf)
        endpoint_energies.append(e)
    max_endpoint_energy = max(endpoint_energies)
    max_endpoint_per_atom = max_endpoint_energy / n_atoms

    if verbosity >= 1:
        print(f"\nEndpoint info:")
        print(f"  Atoms: {n_atoms}")
        print(f"  Max endpoint energy: {max_endpoint_per_atom:.4f} eV/atom")
        print(f"  xi={xi}, delta={delta}")

    # Generate ceiling values (linearly spaced in meV/atom)
    ceilings_mev = []
    ceilings_ev = []
    for i in range(n_workers):
        if n_workers > 1:
            mev = min_ceiling_mev_per_atom + i * (max_ceiling_mev_per_atom - min_ceiling_mev_per_atom) / (n_workers - 1)
        else:
            mev = min_ceiling_mev_per_atom
        ceilings_mev.append(mev)
        # Convert to absolute eV
        ev = max_endpoint_energy + (mev / 1000.0) * n_atoms
        ceilings_ev.append(ev)

    if verbosity >= 1:
        print(f"\nParallel ratchet configuration:")
        print(f"  Workers: {n_workers}")
        print(f"  Ceiling range: {min_ceiling_mev_per_atom:.0f} - {max_ceiling_mev_per_atom:.0f} meV/atom")
        spacing = (max_ceiling_mev_per_atom - min_ceiling_mev_per_atom) / max(n_workers - 1, 1)
        print(f"  Ceiling spacing: {spacing:.1f} meV/atom")
        print(f"  Max adaptations per worker: {max_adaptations}")

    # Serialize structures for workers
    start_structure_dict = start_uc.to_pymatgen_structure().as_dict()
    end_structure_dict = end_uc.to_pymatgen_structure().as_dict()

    # Compute TF threads per worker
    total_cores = mp.cpu_count()
    tf_threads = max(1, total_cores // n_workers)

    if verbosity >= 1:
        print(f"\nLaunching {n_workers} workers ({tf_threads} TF threads each)...")
        print(f"Output directory: {output_dir}")

    # Create worker pool and submit jobs
    results = [None] * n_workers  # Preserve order
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=mp.get_context('spawn'),
        initializer=init_ratchet_worker,
        initargs=(calc_provider, tf_threads),
    ) as pool:
        # Submit all workers
        futures = {}
        for i in range(n_workers):
            future = pool.submit(
                _run_ratchet_worker,
                worker_id=i,
                worker_ceiling=ceilings_ev[i],
                ceiling_mev_per_atom=ceilings_mev[i],
                start_structure_dict=start_structure_dict,
                end_structure_dict=end_structure_dict,
                xi=xi,
                delta=delta,
                min_atoms=min_atoms,
                max_adaptations=max_adaptations,
                xi_factor=xi_factor,
                delta_factor=delta_factor,
                ceiling_step_mev_per_atom=ceiling_step_mev_per_atom,
                dropout=dropout,
                max_iterations=max_iterations,
                initial_max_iters=initial_max_iters,
                beam_width=beam_width,
                output_dir=str(output_dir) if output_dir else None,
            )
            futures[future] = i

        # Collect results as they complete
        for future in as_completed(futures):
            worker_id = futures[future]
            ceiling_mev = ceilings_mev[worker_id]
            try:
                result = future.result()
                results[worker_id] = result

                if verbosity >= 1:
                    if result.best_barrier is not None:
                        barrier_mev = (result.best_barrier / n_atoms - max_endpoint_per_atom) * 1000
                        print(f"  Worker {worker_id:2d} done (ceiling={ceiling_mev:6.0f} meV/atom): "
                              f"barrier={barrier_mev:.1f} meV/atom, "
                              f"paths={len(result.all_paths)}, "
                              f"adaptations={result.metadata.get('num_adaptations', 0)}")
                    else:
                        print(f"  Worker {worker_id:2d} done (ceiling={ceiling_mev:6.0f} meV/atom): "
                              f"no path found, "
                              f"adaptations={result.metadata.get('num_adaptations', 0)}")
            except Exception as e:
                if verbosity >= 1:
                    print(f"  Worker {worker_id:2d} FAILED: {e}")

    # Filter out failed workers (None results)
    valid_results = [r for r in results if r is not None]

    total_elapsed = time.perf_counter() - total_start

    # Find best barrier
    barriers = [r.best_barrier for r in valid_results if r.best_barrier is not None]
    best_barrier = min(barriers) if barriers else None
    best_barrier_mev = None
    if best_barrier is not None:
        best_barrier_mev = (best_barrier / n_atoms - max_endpoint_per_atom) * 1000

    metadata = {
        "min_ceiling_mev_per_atom": min_ceiling_mev_per_atom,
        "max_ceiling_mev_per_atom": max_ceiling_mev_per_atom,
        "n_workers": n_workers,
        "n_workers_succeeded": len(valid_results),
        "xi": xi,
        "delta": delta,
        "max_endpoint_per_atom": max_endpoint_per_atom,
        "total_elapsed_seconds": total_elapsed,
        "total_paths_found": sum(len(r.all_paths) for r in valid_results),
    }
    if atom_step_length is not None:
        metadata["atom_step_length"] = atom_step_length

    parallel_result = ParallelRatchetResult(
        worker_results=valid_results,
        best_barrier=best_barrier,
        best_barrier_mev_per_atom=best_barrier_mev,
        n_atoms=n_atoms,
        metadata=metadata,
    )

    if verbosity >= 1:
        print(f"\n{'=' * 60}")
        print("Parallel Ratchet Complete")
        print(f"{'=' * 60}")
        if best_barrier_mev is not None:
            print(f"  Best barrier: {best_barrier_mev:.1f} meV/atom")
        else:
            print(f"  No paths found")
        print(f"  Workers succeeded: {len(valid_results)}/{n_workers}")
        print(f"  Total paths: {parallel_result.metadata['total_paths_found']}")
        print(f"  Total time: {total_elapsed:.1f}s")

    # Save overall result
    if output_dir:
        result_path = output_dir / "parallel_ratchet_result.json"
        parallel_result.to_json(str(result_path))
        if verbosity >= 1:
            print(f"  Saved: {result_path}")

    return parallel_result
