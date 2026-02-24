"""Refine barrier using iterative A* with ratcheting ceiling."""

import argparse
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("start", help="Starting structure CIF")
    p.add_argument("end", help="Ending structure CIF")
    p.add_argument("--initial-ceiling", type=float, required=True, help="Starting ceiling in eV")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("output"))
    p.add_argument("--model-path", help="Path to fine-tuned GRACE model")
    p.add_argument("--atom-step-length", type=float, default=0.3, help="Target step length in Å")
    p.add_argument("--xi", type=float, default=1.5, help="Lattice discretization parameter")
    p.add_argument("--paths-per-round", type=int, default=10)
    p.add_argument("--max-rounds", type=int, default=20)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--min-dropout", type=float, default=0.1)
    p.add_argument("--beam-width", type=int, default=1000)
    args = p.parse_args()

    # Deferred imports to keep --help fast
    from pymatgen.core import Structure
    from cnf import UnitCell
    from cnf.calculation.grace import GraceCalculator
    from cnf.navigation import compute_delta_for_step_size
    from cnf.navigation.endpoints import get_endpoint_cnfs
    from cnf.navigation.astar.iterative import iterative_astar_barrier

    calc = GraceCalculator(model_path=args.model_path) if args.model_path else GraceCalculator()

    start = UnitCell.from_pymatgen_structure(Structure.from_file(args.start))
    end = UnitCell.from_pymatgen_structure(Structure.from_file(args.end))

    delta = max(compute_delta_for_step_size(start.to_pymatgen_structure(), args.atom_step_length),
                compute_delta_for_step_size(end.to_pymatgen_structure(), args.atom_step_length))

    start_cnfs, end_cnfs = get_endpoint_cnfs(start, end, xi=args.xi, delta=delta)
    n_atoms = len(start_cnfs[0].elements)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    barrier, path_cnfs, path_energies = iterative_astar_barrier(
        start_cnfs, end_cnfs, calc,
        initial_ceiling=args.initial_ceiling,
        paths_per_round=args.paths_per_round,
        max_rounds=args.max_rounds,
        dropout=args.dropout,
        min_dropout=args.min_dropout,
        beam_width=args.beam_width,
        output_dir=args.output_dir,
    )

    if barrier is not None:
        print(f"\nBarrier: {barrier:.4f} eV ({barrier/n_atoms:.4f} eV/atom), path length: {len(path_cnfs)}")
        sys.exit(0)
    else:
        print("\nNo path found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
