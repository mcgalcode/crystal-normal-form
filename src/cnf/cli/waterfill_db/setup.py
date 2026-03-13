"""Setup command for waterfill database."""

import sys
from pathlib import Path


def setup_command(args):
    """Set up a partitioned database for waterfill search from CIF files."""
    import shutil
    import tqdm
    from cnf import UnitCell
    from cnf.calculation.grace import GraceCalculator
    from cnf.db.setup_partitions import setup_search_dir
    from cnf.navigation.endpoints import get_endpoint_cnfs

    start_cif = args.start
    end_cif = args.end

    if not Path(start_cif).exists():
        print(f"Error: Start CIF file '{start_cif}' not found")
        sys.exit(1)

    if not Path(end_cif).exists():
        print(f"Error: End CIF file '{end_cif}' not found")
        sys.exit(1)

    description = f"Search: {Path(start_cif).stem} -> {Path(end_cif).stem}"

    print("\nLoading structures...")
    start_unit_cell = UnitCell.from_cif(start_cif)
    end_unit_cell = UnitCell.from_cif(end_cif)

    print(f"Start structure: {len(start_unit_cell)} atoms per unit cell")
    print(f"End structure:   {len(end_unit_cell)} atoms per unit cell")

    unrelaxed_start_cnfs, unrelaxed_end_cnfs = get_endpoint_cnfs(
        start_unit_cell, end_unit_cell, args.xi, args.delta, args.min_num_atoms
    )

    gcalc = GraceCalculator(model_string=args.calc_model)
    if args.relax:
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

    search_dir = args.search_dir

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

    try:
        setup_search_dir(
            search_dir,
            description,
            args.num_partitions,
            start_cnfs,
            end_cnfs,
            calculator=gcalc
        )
        print(f"\nDatabase created: {search_dir}")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def register_parser(subparsers):
    """Register the setup subcommand."""
    from cnf.calculation.grace import DEFAULT_MODEL

    parser = subparsers.add_parser('setup', help='Set up a partitioned database for waterfill search')
    parser.add_argument('start', help='Path to starting structure CIF file')
    parser.add_argument('end', help='Path to ending structure CIF file')
    parser.add_argument('search_dir', help='Directory for database partition files')
    parser.add_argument('--num-partitions', type=int, required=True, help='Number of database partitions')
    parser.add_argument('--xi', type=float, required=True, help='Xi parameter for CNF')
    parser.add_argument('--delta', type=int, required=True, help='Delta parameter for CNF')
    parser.add_argument('--min-num-atoms', type=int, help='Scale supercells until they contain at least this number of atoms')
    parser.add_argument('--calc-model', type=str, default=DEFAULT_MODEL, help='Model path for MLIP energy calculator')
    parser.add_argument('--relax', action='store_true', help='Relax endpoint structures in CNF space')
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing database without prompting')
    parser.set_defaults(func=setup_command)
