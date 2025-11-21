#!/usr/bin/env python3
"""
Setup a CNF search database from CIF files.

This script creates a partitioned database and search process from start/end CIF files.
It automatically calculates supercell indices to match atom counts, or you can specify them manually.

Usage:
    # Auto-calculate supercell indices (scales smaller structure to match larger)
    python scripts/setup_search_db.py --start-cif bcc.cif --end-cif hcp.cif --partitions-dir search_db --num-partitions 8

    # Manually specify same index for both
    python scripts/setup_search_db.py --start-cif bcc.cif --end-cif hcp.cif --partitions-dir search_db --num-partitions 8 --supercell-index 2

    # Manually specify different indices
    python scripts/setup_search_db.py --start-cif bcc.cif --end-cif hcp.cif --partitions-dir search_db --num-partitions 8 --start-supercell-index 2 --end-supercell-index 1
"""

import argparse
import sys
import os
from pathlib import Path
import shutil
import tqdm

from cnf import UnitCell, CrystalNormalForm
from cnf.db.setup import setup_cnf_db
from cnf.search import instantiate_search
from cnf.utils.pdd import pdd_for_cnfs


# Default CNF parameters
DEFAULT_XI = 1.5
DEFAULT_DELTA = 10


def load_and_validate_supercells(cif_path: str, supercell_index: int,
                                 xi: float, delta: int, label: str) -> list[CrystalNormalForm]:
    """Load CIF, generate supercells, convert to CNF, and validate."""
    print(f"\n{label}:")
    print(f"  Loading: {cif_path}")

    unit_cell = UnitCell.from_cif(cif_path)
    num_atoms = len(unit_cell)

    if supercell_index == 1:
        print(f"  Using primitive cell ({num_atoms} atoms)")
    else:
        print(f"  Generating supercells with index {supercell_index}...")
        print(f"  Primitive cell: {num_atoms} atoms → Supercell: {num_atoms * supercell_index} atoms")

    supercells = unit_cell.supercells(supercell_index)
    cnfs = [uc.to_cnf(xi, delta) for uc in supercells]
    unique_cnfs = list(set(cnfs))

    print(f"  Generated {len(supercells)} supercells → {len(unique_cnfs)} unique CNFs")

    return unique_cnfs


def calculate_supercell_indices(start_atoms: int, end_atoms: int):
    """Calculate supercell indices needed to match atom counts.

    Returns:
        (start_index, end_index): Supercell indices for start and end structures
    """
    if start_atoms == end_atoms:
        return 1, 1

    # Determine which needs scaling
    if start_atoms < end_atoms:
        ratio = end_atoms / start_atoms
        if ratio == int(ratio):
            return int(ratio), 1
        else:
            raise ValueError(
                f"Cannot automatically determine supercell index: "
                f"end structure has {end_atoms} atoms, start has {start_atoms} atoms. "
                f"Ratio {ratio:.2f} is not an integer. "
                f"Please specify supercell indices manually."
            )
    else:  # end_atoms < start_atoms
        ratio = start_atoms / end_atoms
        if ratio == int(ratio):
            return 1, int(ratio)
        else:
            raise ValueError(
                f"Cannot automatically determine supercell index: "
                f"start structure has {start_atoms} atoms, end has {end_atoms} atoms. "
                f"Ratio {ratio:.2f} is not an integer. "
                f"Please specify supercell indices manually."
            )


def setup_search_database(start_cif: str, end_cif: str, partitions_dir: str, num_partitions: int,
                          start_supercell_index: int, end_supercell_index: int,
                          xi: float = DEFAULT_XI, delta: int = DEFAULT_DELTA) -> int:
    """Create database and populate with start/end points.

    Returns:
        search_id: The ID of the created search process
    """

    print("=" * 70)
    print("SETTING UP CNF SEARCH DATABASE")
    print("=" * 70)

    # Load and validate structures
    start_cnfs = load_and_validate_supercells(start_cif, start_supercell_index, xi, delta, "START STRUCTURE")
    end_cnfs = load_and_validate_supercells(end_cif, end_supercell_index, xi, delta, "END STRUCTURE")

    # Validate start-end distances are consistent
    print("\nValidating start-end distances...")
    start_end_dists = []
    for s in start_cnfs:
        for e in end_cnfs:
            dist = pdd_for_cnfs(s, e, k=100)
            start_end_dists.append(round(dist, 10))

    unique_dists = set(start_end_dists)
    if len(unique_dists) == 1:
        print(f"  ✓ All start-end distances equal: {list(unique_dists)[0]:.6f}")
    else:
        print(f"  ⚠️  Warning: Found {len(unique_dists)} different start-end distances")
        print(f"     Range: {min(unique_dists):.6f} to {max(unique_dists):.6f}")

    element_list = start_cnfs[0].elements
    description = f"Search: {Path(start_cif).stem} → {Path(end_cif).stem}"

    # Create database
    print(f"\n")
    os.makedirs(partitions_dir, exist_ok=True)
    for i in tqdm.tqdm(range(num_partitions), total=num_partitions, desc=f"Creating database partitions in: {partitions_dir}"):
        store_file = f"{partitions_dir}/graph_partition_{i}.db" 
        setup_cnf_db(store_file, xi, delta, element_list)

        # Create search process
        search_id = instantiate_search(description, start_cnfs, end_cnfs, store_file)

    print(f"\n✓ Search database partitions initialized!")
    print(f"  Partitions Dir:     {partitions_dir}")
    print(f"  Search ID:    {search_id}")
    print(f"  Start points: {len(start_cnfs)}")
    print(f"  End points:   {len(end_cnfs)}")
    print()

    return search_id


def main():
    parser = argparse.ArgumentParser(
        description="Setup a CNF search database from CIF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-calculate supercell indices to match atom counts
  %(prog)s --start-cif bcc.cif --end-cif hcp.cif --partitions-dir search_db --num-partitions 8

  # Use same supercell index for both structures
  %(prog)s --start-cif bcc.cif --end-cif hcp.cif --partitions-dir search_db --num-partitions 8 --supercell-index 2

  # Specify different supercell indices for each structure
  %(prog)s --start-cif bcc.cif --end-cif hcp.cif --partitions-dir search_db --num-partitions 8 --start-supercell-index 2 --end-supercell-index 1

  # Custom CNF parameters
  %(prog)s --start-cif bcc.cif --end-cif hcp.cif --partitions-dir search_db --num-partitions 8 --xi 2.0 --delta 15
        """
    )

    parser.add_argument('--start-cif', help='Path to starting structure CIF file')
    parser.add_argument('--end-cif', help='Path to ending structure CIF file')
    parser.add_argument('--partitions-dir', help='Directory for database partition files')
    parser.add_argument('--num-partitions', type=int, help='Number of database partitions to create')

    parser.add_argument('--supercell-index', type=int,
                       help='Supercell index for BOTH structures (e.g., 2 means 2x atoms)')
    parser.add_argument('--start-supercell-index', type=int,
                       help='Supercell index for start structure only (overrides --supercell-index)')
    parser.add_argument('--end-supercell-index', type=int,
                       help='Supercell index for end structure only (overrides --supercell-index)')

    parser.add_argument('--xi', type=float, default=DEFAULT_XI,
                       help=f'Xi parameter for CNF (default: {DEFAULT_XI})')
    parser.add_argument('--delta', type=int, default=DEFAULT_DELTA,
                       help=f'Delta parameter for CNF (default: {DEFAULT_DELTA})')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Overwrite existing database without prompting')

    args = parser.parse_args()

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

    # Load CIF files to get atom counts
    print("\nLoading structures...")
    start_unit_cell = UnitCell.from_cif(start_cif)
    end_unit_cell = UnitCell.from_cif(end_cif)

    start_atoms = len(start_unit_cell)
    end_atoms = len(end_unit_cell)

    print(f"Start structure: {start_atoms} atoms per unit cell")
    print(f"End structure:   {end_atoms} atoms per unit cell")

    # Determine supercell indices with precedence:
    # 1. Individual indices (--start/end-supercell-index) override everything
    # 2. Common index (--supercell-index) applies to both
    # 3. Auto-calculate to match atom counts

    if args.start_supercell_index is not None or args.end_supercell_index is not None:
        # Individual indices specified
        start_supercell_index = args.start_supercell_index if args.start_supercell_index is not None else args.supercell_index if args.supercell_index is not None else 1
        end_supercell_index = args.end_supercell_index if args.end_supercell_index is not None else args.supercell_index if args.supercell_index is not None else 1
        print(f"\nUsing specified supercell indices:")
        print(f"  Start: index {start_supercell_index} → {start_atoms * start_supercell_index} atoms")
        print(f"  End:   index {end_supercell_index} → {end_atoms * end_supercell_index} atoms")
    elif args.supercell_index is not None:
        # Common index for both
        start_supercell_index = args.supercell_index
        end_supercell_index = args.supercell_index
        print(f"\nUsing supercell index {args.supercell_index} for both structures:")
        print(f"  Start: {start_atoms * start_supercell_index} atoms")
        print(f"  End:   {end_atoms * end_supercell_index} atoms")
    else:
        # Auto-calculate
        try:
            start_supercell_index, end_supercell_index = calculate_supercell_indices(start_atoms, end_atoms)
            print(f"\nAuto-calculated supercell indices:")
            print(f"  Start: index {start_supercell_index} → {start_atoms * start_supercell_index} atoms")
            print(f"  End:   index {end_supercell_index} → {end_atoms * end_supercell_index} atoms")
        except ValueError as e:
            print(f"\n❌ {e}")
            sys.exit(1)

    # Get database filename (from args or prompt)
    partitions_dir = args.partitions_dir
    num_partitions = args.num_partitions
    if not partitions_dir:
        print("\nDatabase setup")
        print("-" * 70)
        partitions_dir = input("Enter database filename (e.g., search.db): ")

    # Check if database exists
    if Path(partitions_dir).exists():
        if args.force:
            print(f"Overwriting existing database: {partitions_dir}")
            shutil.rmtree(partitions_dir)
        else:
            response = input(f"Database '{partitions_dir}' already exists. Overwrite? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Cancelled.")
                sys.exit(0)
            shutil.rmtree(partitions_dir)

    # Create database and search process
    try:
        search_id = setup_search_database(
            start_cif=start_cif,
            end_cif=end_cif,
            partitions_dir=partitions_dir,
            num_partitions=num_partitions,
            start_supercell_index=start_supercell_index,
            end_supercell_index=end_supercell_index,
            xi=args.xi,
            delta=args.delta
        )
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
