#!/usr/bin/env python3
"""
Setup a CNF search database from CIF files.

This script creates a database and search process from start/end CIF files,
generating supercells and initializing the search.

Usage:
    # Interactive mode (prompts for all inputs)
    python scripts/setup_search_db.py

    # Command-line mode
    python scripts/setup_search_db.py --start-cif bcc.cif --end-cif hcp.cif --db-file search.db --supercell-index 2

    # Partial (prompts for missing inputs)
    python scripts/setup_search_db.py --start-cif bcc.cif --end-cif hcp.cif
"""

import argparse
import sys
import os
from pathlib import Path
import shutil

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
    print(f"  Generating supercells with index {supercell_index}...")

    supercells = unit_cell.supercells(supercell_index)
    cnfs = [uc.to_cnf(xi, delta) for uc in supercells]
    unique_cnfs = list(set(cnfs))

    print(f"  Generated {len(supercells)} supercells → {len(unique_cnfs)} unique CNFs")

    return unique_cnfs


def setup_search_database(start_cif: str, end_cif: str, partitions_dir: str, num_partitions: int,
                          supercell_index: int, xi: float = DEFAULT_XI,
                          delta: int = DEFAULT_DELTA) -> int:
    """Create database and populate with start/end points.

    Returns:
        search_id: The ID of the created search process
    """

    print("=" * 70)
    print("SETTING UP CNF SEARCH DATABASE")
    print("=" * 70)

    # Load and validate structures
    start_cnfs = load_and_validate_supercells(start_cif, supercell_index, xi, delta, "START STRUCTURE")
    end_cnfs = load_and_validate_supercells(end_cif, supercell_index, xi, delta, "END STRUCTURE")

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
    print(f"\nCreating database partitions in: {partitions_dir}")
    os.makedirs(partitions_dir, exist_ok=True)
    for i in range(num_partitions):
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
  # Interactive mode (prompts for inputs)
  %(prog)s

  # Command-line mode
  %(prog)s --start-cif bcc.cif --end-cif hcp.cif --db-file search.db --supercell-index 2

  # Custom CNF parameters
  %(prog)s --start-cif bcc.cif --end-cif hcp.cif --xi 2.0 --delta 15
        """
    )

    parser.add_argument('--start-cif', help='Path to starting structure CIF file')
    parser.add_argument('--end-cif', help='Path to ending structure CIF file')
    parser.add_argument('--partitions-dir', help='Path for the database partition files to create')
    parser.add_argument('--num-partitions', type=int, help='Path for the database partition files to create')

    parser.add_argument('--supercell-index', type=int, help='Supercell index (e.g., 2 for 2x2x2)')
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

    # Get supercell index (from args or prompt)
    supercell_index = args.supercell_index
    if supercell_index is None:
        print("\nSupercell generation")
        print("-" * 70)
        print(f"Start structure: {start_atoms} atoms per unit cell")
        print(f"End structure:   {end_atoms} atoms per unit cell")
        print()
        supercell_index = int(input("Enter supercell index (e.g., 2 for 2x2x2): "))

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
            supercell_index=supercell_index,
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
