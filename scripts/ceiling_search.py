"""Ceiling barrier search between two crystal structures."""

import argparse
import sys
from pathlib import Path
from pymatgen.core import Structure

from cnf import UnitCell
from cnf.calculation.grace import GraceCalculator
from cnf.navigation import compute_delta_for_step_size
from cnf.navigation.astar.iterative import ceiling_barrier_search


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

    calc = GraceCalculator(model_path=args.model_path) if args.model_path else GraceCalculator()

    start = UnitCell.from_pymatgen(Structure.from_file(args.start))
    end = UnitCell.from_pymatgen(Structure.from_file(args.end))
    n_atoms = len(start.atoms)

    delta = max(compute_delta_for_step_size(start.to_pymatgen(), args.atom_step_length),
                compute_delta_for_step_size(end.to_pymatgen(), args.atom_step_length))

    e_start = calc.calculate_energy(start.to_cnf(delta)) / n_atoms
    e_end = calc.calculate_energy(end.to_cnf(delta)) / n_atoms
    e_max = max(e_start, e_end)

    ceiling_step = args.ceiling_step_per_atom * n_atoms
    max_ceiling = args.max_ceiling if args.max_ceiling else e_max + ceiling_step * args.num_ceilings

    args.output_dir.mkdir(parents=True, exist_ok=True)

    barrier, path_cnfs, path_energies = ceiling_barrier_search(
        start, end, calc,
        output_dir=args.output_dir,
        initial_delta=delta,
        max_ceiling=max_ceiling,
        ceiling_step=ceiling_step,
        num_ceilings=args.num_ceilings,
        attempts_per_ceiling=args.attempts_per_ceiling,
        max_passes=args.max_passes,
        max_sweep_rounds=args.max_sweep_rounds,
        relax_endpoints=args.relax_endpoints,
    )

    if barrier is not None:
        print(f"\nBarrier: {barrier:.4f} eV ({barrier/n_atoms:.4f} eV/atom), path length: {len(path_cnfs)}")
        sys.exit(0)
    else:
        print("\nNo path found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
