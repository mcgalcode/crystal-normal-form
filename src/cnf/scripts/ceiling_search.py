"""Ceiling barrier search between two crystal structures."""

import argparse
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        description="Ceiling sweep barrier search. Requires --max-ceiling (use path sampling to discover it first)."
    )
    p.add_argument("start", help="Starting structure CIF")
    p.add_argument("end", help="Ending structure CIF")
    p.add_argument("--max-ceiling", type=float, required=True,
                   help="Max ceiling in eV (required, from Phase 2 path sampling)")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("output"))
    p.add_argument("--model-path", help="Path to fine-tuned GRACE model")
    p.add_argument("--atom-step-length", type=float, default=0.3, help="Target step length in Å")
    p.add_argument("--num-ceilings", type=int, default=8)
    p.add_argument("--attempts-per-ceiling", type=int, default=2)
    p.add_argument("--max-passes", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (0.0-1.0)")
    p.add_argument("--relax-endpoints", action="store_true")
    p.add_argument("--min-atoms", type=int, help="Minimum atoms (will create supercells if needed)")
    args = p.parse_args()

    # Deferred imports to keep --help fast
    from pymatgen.core import Structure
    from cnf import UnitCell
    from cnf.calculation.grace import GraceCalcProvider
    from cnf.calculation.relaxation import relax_unit_cell
    from cnf.navigation.endpoints import get_endpoint_unit_cells
    from cnf.navigation.astar.iterative import sweep

    calc_provider = GraceCalcProvider(model_path=args.model_path)

    start = UnitCell.from_pymatgen_structure(Structure.from_file(args.start))
    end = UnitCell.from_pymatgen_structure(Structure.from_file(args.end))

    # Relax endpoints if requested (before any supercell expansion)
    if args.relax_endpoints:
        print("Relaxing endpoints in continuous space...")
        calc = calc_provider()
        start = relax_unit_cell(start, calc._calc, verbose=True, label="start")
        end = relax_unit_cell(end, calc._calc, verbose=True, label="end")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        start.to_cif(str(args.output_dir / "start_relaxed.cif"))
        end.to_cif(str(args.output_dir / "end_relaxed.cif"))
        print(f"  Relaxed CIFs saved to {args.output_dir}")

    # Get n_atoms from supercell if min_atoms specified
    if args.min_atoms:
        start_cells, _ = get_endpoint_unit_cells(start, end, min_atoms=args.min_atoms)
        n_atoms = len(start_cells[0])
    else:
        n_atoms = len(start)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # sweep() computes delta internally from atom_step_length and min_atoms
    result = sweep(
        start, end,
        max_ceiling=args.max_ceiling,
        calc_provider=calc_provider,
        atom_step_length=args.atom_step_length,
        min_atoms=args.min_atoms,
        num_ceilings=args.num_ceilings,
        attempts_per_ceiling=args.attempts_per_ceiling,
        max_passes=args.max_passes,
        dropout=args.dropout,
        output_dir=args.output_dir,
    )

    best = result.best_path
    if best is not None:
        barrier = best.barrier
        energies = best.energies
        min_endpoint = min(energies[0], energies[-1])
        barrier_height = barrier - min_endpoint
        print(f"\nBarrier: {barrier:.4f} eV (height: {barrier_height:.4f} eV, {barrier_height/n_atoms:.4f} eV/atom), path length: {len(best)}")
        sys.exit(0)
    else:
        print("\nNo path found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
