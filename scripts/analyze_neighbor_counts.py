#!/usr/bin/env python3
"""
Analyze the number of neighbors as a function of structure properties.

This script uses only the top-level public API to study how neighbor counts
vary across different crystal structures. It helps understand:
- Distribution of neighbor counts
- Correlation with structure properties (atoms, lattice parameters, etc.)
- Which structures have unusually high/low neighbor counts

Usage:
    # Analyze with Python implementation
    python scripts/analyze_neighbor_counts.py

    # Analyze with Rust implementation (faster)
    USE_RUST=1 python scripts/analyze_neighbor_counts.py

    # Analyze more structures (default is 50)
    python scripts/analyze_neighbor_counts.py --max-structures 100
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Use only public API
from cnf import UnitCell
from cnf.navigation.neighbor_finder import NeighborFinder


def load_structures(max_structures: int = 50) -> List[tuple]:
    """Load structures from CIF files using public API only."""
    cif_dir = Path("tests/data/mp_cifs")
    cif_files = sorted(glob.glob(str(cif_dir / "*.cif")))[:max_structures]

    structures = []
    print(f"Loading {len(cif_files)} structures...")

    for cif_file in cif_files:
        try:
            unit_cell = UnitCell.from_cif(cif_file)
            struct_name = Path(cif_file).stem
            structures.append((struct_name, unit_cell))
        except Exception as e:
            print(f"Warning: Failed to load {cif_file}: {e}")
            continue

    print(f"Successfully loaded {len(structures)} structures\n")
    return structures


def extract_structure_properties(unit_cell: UnitCell) -> Dict[str, Any]:
    """Extract properties from a structure using public API."""
    # Get pymatgen structure for analysis
    pmg_struct = unit_cell.to_pymatgen_structure()

    # Get lattice parameters
    lattice = pmg_struct.lattice

    return {
        'n_atoms': len(pmg_struct),
        'volume': lattice.volume,
        'density': pmg_struct.density,
        'a': lattice.a,
        'b': lattice.b,
        'c': lattice.c,
        'alpha': lattice.alpha,
        'beta': lattice.beta,
        'gamma': lattice.gamma,
        'n_species': len(set(pmg_struct.species)),
    }


def analyze_neighbors(
    structures: List[tuple],
    xi: float = 0.01,
    delta: int = 10
) -> List[Dict[str, Any]]:
    """
    Analyze neighbor counts for each structure.

    Args:
        structures: List of (name, UnitCell) tuples
        xi: Lattice step size
        delta: Motif discretization parameter

    Returns:
        List of dictionaries with analysis results
    """
    results = []

    print(f"Analyzing neighbor counts...")
    print(f"Parameters: xi={xi}, delta={delta}")
    print()

    for i, (name, unit_cell) in enumerate(structures):
        print(f"  [{i+1}/{len(structures)}] {name}...", end=' ')

        try:
            # Get structure properties
            props = extract_structure_properties(unit_cell)

            # Convert to CNF
            cnf = unit_cell.to_cnf(xi=xi, delta=delta)

            # Find neighbors
            finder = NeighborFinder(cnf)
            neighbors = finder.find_neighbors()
            n_neighbors = len(neighbors)

            print(f"{n_neighbors} neighbors ({props['n_atoms']} atoms)")

            results.append({
                'name': name,
                'n_neighbors': n_neighbors,
                **props
            })

        except Exception as e:
            print(f"FAILED: {e}")
            continue

    print(f"\nCompleted: {len(results)}/{len(structures)} structures")
    return results


def compute_statistics(results: List[Dict[str, Any]]):
    """Compute and print statistical summary."""
    neighbor_counts = [r['n_neighbors'] for r in results]

    print("\n" + "=" * 80)
    print("NEIGHBOR COUNT STATISTICS")
    print("=" * 80)
    print(f"Number of structures analyzed: {len(results)}")
    print()
    print(f"Neighbor count distribution:")
    print(f"  Mean:       {np.mean(neighbor_counts):8.1f}")
    print(f"  Median:     {np.median(neighbor_counts):8.1f}")
    print(f"  Std dev:    {np.std(neighbor_counts):8.1f}")
    print(f"  Min:        {np.min(neighbor_counts):8.0f}")
    print(f"  Max:        {np.max(neighbor_counts):8.0f}")
    print()

    # Percentiles
    print(f"Percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(neighbor_counts, p)
        print(f"  {p:2d}th:      {val:8.1f}")
    print()


def analyze_correlations(results: List[Dict[str, Any]]):
    """Analyze correlations between neighbor count and structure properties."""
    if len(results) < 2:
        return

    neighbor_counts = np.array([r['n_neighbors'] for r in results])

    print("=" * 80)
    print("CORRELATIONS WITH STRUCTURE PROPERTIES")
    print("=" * 80)

    properties = ['n_atoms', 'volume', 'density', 'n_species', 'a', 'b', 'c']

    correlations = []
    for prop in properties:
        values = np.array([r[prop] for r in results])
        if np.std(values) > 0:  # Only if there's variation
            corr = np.corrcoef(neighbor_counts, values)[0, 1]
            correlations.append((prop, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"Property correlations (sorted by strength):")
    for prop, corr in correlations:
        sign = "+" if corr > 0 else "-"
        strength = "***" if abs(corr) > 0.7 else "**" if abs(corr) > 0.5 else "*" if abs(corr) > 0.3 else ""
        print(f"  {prop:12s}: {sign}{abs(corr):5.3f} {strength}")
    print()


def find_outliers(results: List[Dict[str, Any]]):
    """Find structures with unusually high or low neighbor counts."""
    neighbor_counts = np.array([r['n_neighbors'] for r in results])

    # Sort by neighbor count
    sorted_results = sorted(results, key=lambda x: x['n_neighbors'])

    print("=" * 80)
    print("INTERESTING CASES")
    print("=" * 80)

    print(f"\nStructures with FEWEST neighbors:")
    for r in sorted_results[:5]:
        print(f"  {r['name']:20s}: {r['n_neighbors']:3d} neighbors, {r['n_atoms']:3d} atoms")

    print(f"\nStructures with MOST neighbors:")
    for r in sorted_results[-5:][::-1]:
        print(f"  {r['name']:20s}: {r['n_neighbors']:3d} neighbors, {r['n_atoms']:3d} atoms")
    print()


def plot_results(results: List[Dict[str, Any]], output_dir: Path):
    """Generate visualization plots."""
    output_dir.mkdir(exist_ok=True)

    neighbor_counts = np.array([r['n_neighbors'] for r in results])
    n_atoms = np.array([r['n_atoms'] for r in results])

    # Plot 1: Distribution of neighbor counts
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(neighbor_counts, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Neighbors', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Neighbor Counts', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add statistics annotation
    mean_val = np.mean(neighbor_counts)
    median_val = np.median(neighbor_counts)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'neighbor_count_distribution.png', dpi=150)
    print(f"Saved: {output_dir / 'neighbor_count_distribution.png'}")
    plt.close()

    # Plot 2: Neighbor count vs structure size
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(n_atoms, neighbor_counts, alpha=0.6, s=50)
    ax.set_xlabel('Number of Atoms', fontsize=12)
    ax.set_ylabel('Number of Neighbors', fontsize=12)
    ax.set_title('Neighbor Count vs Structure Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add correlation
    if len(results) > 1:
        corr = np.corrcoef(neighbor_counts, n_atoms)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'neighbors_vs_atoms.png', dpi=150)
    print(f"Saved: {output_dir / 'neighbors_vs_atoms.png'}")
    plt.close()


def save_results(results: List[Dict[str, Any]], output_file: Path):
    """Save detailed results to CSV."""
    import csv

    if not results:
        return

    # Get all keys
    keys = list(results[0].keys())

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved detailed results: {output_file}")


def main():
    """Run the neighbor count analysis."""
    parser = argparse.ArgumentParser(description='Analyze neighbor counts across structures')
    parser.add_argument('--max-structures', type=int, default=50,
                        help='Maximum number of structures to analyze (default: 50)')
    parser.add_argument('--xi', type=float, default=0.01,
                        help='Lattice step size (default: 0.01)')
    parser.add_argument('--delta', type=int, default=10,
                        help='Motif discretization parameter (default: 10)')
    parser.add_argument('--output-dir', type=Path, default=Path('neighbor_analysis'),
                        help='Output directory for plots and data (default: neighbor_analysis)')

    args = parser.parse_args()

    # Check if using Rust
    use_rust = os.getenv("USE_RUST") is not None
    implementation = "Rust" if use_rust else "Python"

    print("=" * 80)
    print(f"NEIGHBOR COUNT ANALYSIS - {implementation} implementation")
    print("=" * 80)
    print()

    # Load structures
    structures = load_structures(max_structures=args.max_structures)

    if not structures:
        print("Error: No structures loaded")
        sys.exit(1)

    # Analyze neighbors
    results = analyze_neighbors(structures, xi=args.xi, delta=args.delta)

    if not results:
        print("Error: No results obtained")
        sys.exit(1)

    # Compute statistics
    compute_statistics(results)

    # Analyze correlations
    analyze_correlations(results)

    # Find outliers
    find_outliers(results)

    # Generate plots
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    try:
        plot_results(results, args.output_dir)
    except Exception as e:
        print(f"Warning: Failed to generate plots: {e}")

    # Save detailed results
    save_results(results, args.output_dir / 'neighbor_counts.csv')

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
