#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import shutil
import tqdm

from cnf import UnitCell
from cnf.calculation.grace import GraceCalculator, DEFAULT_MODEL
from cnf.db.setup_partitions import setup_search_dir
from cnf.navigation.endpoints import get_endpoint_cnfs

def main():
    parser = argparse.ArgumentParser(
        description="Setup a CNF search database from CIF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-calculate supercell indices to match atom counts
  %(prog)s --start-cif bcc.cif --end-cif hcp.cif --partitions-dir search_db --num-partitions 8 --xi 2.0 --delta 15

  # Calculate supercell indices automatically based on specified minimum number of atoms
  %(prog)s --start-cif bcc.cif --end-cif hcp.cif --partitions-dir search_db --num-partitions 8 --min-num-atoms 5 --xi 2.0 --delta 15
        """
    )

    parser.add_argument('--start-cif', help='Path to starting structure CIF file')
    parser.add_argument('--end-cif', help='Path to ending structure CIF file')
    parser.add_argument('--search-dir', help='Directory for database partition files')
    parser.add_argument('--num-partitions', type=int, help='Number of database partitions to create')

    parser.add_argument('--min-num-atoms', type=int, help="Scale supercells until they contain at least this number of atoms")
    parser.add_argument('--calc-model', type=str, default=DEFAULT_MODEL,
                        help="Model path for MLIP energy calculator")
    parser.add_argument('--relax', action='store_true',
                        help='Relax endpoint structures in CNF space using calculator')

    parser.add_argument('--xi', type=float,
                       help=f'Xi parameter for CNF')
    parser.add_argument('--delta', type=int,
                       help=f'Delta parameter for CNF')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Overwrite existing database without prompting')

    args = parser.parse_args()

    xi = args.xi
    delta = args.delta
    should_relax = args.relax
    calc_model = args.calc_model

    # Get start CIF (from args or prompt)
    start_cif = args.start_cif
    if not start_cif:
        start_cif = input("Path to start structure CIF file: ")

    if not Path(start_cif).exists():
        print(f"Error: Start CIF file '{start_cif}' not found")
        sys.exit(1)

    # Get end CIF (from args or prompt)
    end_cif = args.end_cif
    if not end_cif:
        end_cif = input("Path to end structure CIF file: ")

    if not Path(end_cif).exists():
        print(f"Error: End CIF file '{end_cif}' not found")
        sys.exit(1)


    description = f"Search: {Path(start_cif).stem} → {Path(end_cif).stem}"
    # Load CIF files to get atom counts
    print("\nLoading structures...")
    start_unit_cell = UnitCell.from_cif(start_cif)
    end_unit_cell = UnitCell.from_cif(end_cif)

    start_atoms = len(start_unit_cell)
    end_atoms = len(end_unit_cell)

    print(f"Start structure: {start_atoms} atoms per unit cell")
    print(f"End structure:   {end_atoms} atoms per unit cell")

    unrelaxed_start_cnfs, unrelaxed_end_cnfs = get_endpoint_cnfs(start_unit_cell, end_unit_cell, args.min_num_atoms, xi, delta)

    gcalc = GraceCalculator(model_string=calc_model)
    if should_relax:
       
        start_cnfs = []
        for c in tqdm.tqdm(unrelaxed_start_cnfs, desc="Relaxing starting CNFs"):
            print(f"Relaxing CNF {c}")
            relaxed, e, iters = gcalc.relax(c)
            start_cnfs.append(relaxed)

        end_cnfs = []
        for c in tqdm.tqdm(unrelaxed_end_cnfs, desc="Relaxing ending CNFs"):
            print(f"Relaxing CNF {c}")
            relaxed, e, iters = gcalc.relax(c)
            end_cnfs.append(relaxed)    
    else:
        start_cnfs = unrelaxed_start_cnfs
        end_cnfs = unrelaxed_end_cnfs

    # Get database filename (from args or prompt)
    search_dir = args.search_dir
    num_partitions = args.num_partitions
    if not search_dir:
        print("\nDatabase setup")
        print("-" * 70)
        search_dir = input("Enter database filename (e.g., search.db): ")

    # Check if database exists
    if Path(search_dir).exists():
        if args.force:
            print(f"Overwriting existing database: {search_dir}")
            shutil.rmtree(search_dir)
        else:
            response = input(f"Database '{search_dir}' already exists. Overwrite? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Cancelled.")
                sys.exit(0)
            shutil.rmtree(search_dir)

    # Create database and search process
    try:
        setup_search_dir(
            search_dir,
            description,
            num_partitions,
            start_cnfs,
            end_cnfs,
            calculator_model=calc_model,
            calculator=gcalc
        )
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
