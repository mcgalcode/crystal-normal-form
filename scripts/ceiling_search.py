"""Ceiling barrier search between two crystal structures."""

import argparse
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("start", help="Starting structure CIF")
    p.add_argument("end", help="Ending structure CIF")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("output"))
    p.add_argument("--model-path", help="Path to fine-tuned GRACE model")
    p.add_argument("--atom-step-length", type=float, default=0.3, help="Target step length in Å")
    p.add_argument("--ceiling-step-per-atom", type=float, default=0.1, help="Ceiling increment in eV/atom")
    p.add_argument("--num-ceilings", type=int, default=8)
    p.add_argument("--attempts-per-ceiling", type=int, default=2)
    p.add_argument("--max-ceiling", type=float, help="Max ceiling in eV (warm start)")
    p.add_argument("--max-passes", type=int, default=3)
    p.add_argument("--max-sweep-rounds", type=int, default=10)
    p.add_argument("--relax-endpoints", action="store_true")
    args = p.parse_args()

    # Deferred imports to keep --help fast
    from pymatgen.core import Structure
    from cnf import UnitCell
    from cnf.calculation.grace import GraceCalculator
    from cnf.navigation import compute_delta_for_step_size
    from cnf.navigation.astar.iterative import ceiling_barrier_search

    calc = GraceCalculator(model_path=args.model_path) if args.model_path else GraceCalculator()

    start = UnitCell.from_pymatgen_structure(Structure.from_file(args.start))
    end = UnitCell.from_pymatgen_structure(Structure.from_file(args.end))
    n_atoms = len(start)

    delta = max(compute_delta_for_step_size(start.to_pymatgen_structure(), args.atom_step_length),
                compute_delta_for_step_size(end.to_pymatgen_structure(), args.atom_step_length))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    barrier, path_cnfs, path_energies = ceiling_barrier_search(
        start, end, calc,
        delta=delta,
        step_per_atom=args.ceiling_step_per_atom,
        num_ceilings=args.num_ceilings,
        attempts_per_ceiling=args.attempts_per_ceiling,
        max_passes=args.max_passes,
        max_sweep_rounds=args.max_sweep_rounds,
        max_ceiling=args.max_ceiling,
        relax_endpoints=args.relax_endpoints,
        output_dir=args.output_dir,
    )

    if barrier is not None:
        min_endpoint = min(path_energies[0], path_energies[-1])
        barrier_height = barrier - min_endpoint
        print(f"\nBarrier: {barrier:.4f} eV (height: {barrier_height:.4f} eV, {barrier_height/n_atoms:.4f} eV/atom), path length: {len(path_cnfs)}")
        sys.exit(0)
    else:
        print("\nNo path found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
