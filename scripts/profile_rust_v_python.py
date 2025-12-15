#!/usr/bin/env python3
"""
Profile the stabilizer combination optimization across multiple structures.

This script uses only the top-level public API to ensure it remains valid
even if implementation details change. It tests both CNF construction and
neighbor finding performance.

Usage:
    # Test with Python implementation
    python scripts/profile_stabilizer_optimization.py

    # Test with Rust implementation
    USE_RUST=1 python scripts/profile_stabilizer_optimization.py
"""

import os
import sys
import time
import glob
from pathlib import Path
import numpy as np
from typing import List, Dict, Any

# Use only public API
from cnf import UnitCell
from cnf.navigation.neighbor_finder import NeighborFinder


def load_structures(max_structures: int = 100) -> List[tuple]:
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


def benchmark_cnf_construction(
    structures: List[tuple],
    xi: float = 0.01,
    delta: int = 10,
    n_iterations: int = 3
) -> Dict[str, Any]:
    """
    Benchmark CNF construction using public API only.

    Args:
        structures: List of (name, UnitCell) tuples
        xi: Lattice step size
        delta: Motif discretization parameter
        n_iterations: Number of iterations per structure

    Returns:
        Dictionary with benchmark results
    """
    results = []

    print(f"Benchmarking CNF construction ({n_iterations} iterations per structure)...")
    print(f"Parameters: xi={xi}, delta={delta}")

    for i, (name, unit_cell) in enumerate(structures):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(structures)}")

        try:
            # Warmup
            cnf = unit_cell.to_cnf(xi=xi, delta=delta)
            n_atoms = len(cnf.motif_normal_form.elements)

            # Benchmark
            times = []
            for _ in range(n_iterations):
                start = time.perf_counter()
                cnf = unit_cell.to_cnf(xi=xi, delta=delta)
                elapsed = time.perf_counter() - start
                times.append(elapsed * 1000)  # Convert to ms

            results.append({
                'name': name,
                'n_atoms': n_atoms,
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            })
        except Exception as e:
            print(f"  Warning: Failed to benchmark {name}: {e}")
            continue

    print(f"  Completed: {len(results)}/{len(structures)}\n")
    return {
        'results': results,
        'n_iterations': n_iterations
    }


def benchmark_neighbor_finding(
    structures: List[tuple],
    xi: float = 0.01,
    delta: int = 10,
    max_structures: int = 20,
    n_iterations: int = 2,
    use_pure_rust: bool = False
) -> Dict[str, Any]:
    """
    Benchmark neighbor TUPLE finding using public API only.

    Note: Neighbor finding is expensive, so we limit to fewer structures.

    Args:
        structures: List of (name, UnitCell) tuples
        xi: Lattice step size
        delta: Motif discretization parameter
        max_structures: Maximum number of structures to test
        n_iterations: Number of iterations per structure
        use_pure_rust: If True, use rust_cnf.find_neighbor_tuples_rust directly

    Returns:
        Dictionary with benchmark results
    """
    results = []

    mode = "Pure Rust" if use_pure_rust else "Python/Rust hybrid"
    print(f"Benchmarking neighbor TUPLE finding - {mode} ({n_iterations} iterations, max {max_structures} structures)...")
    print(f"Parameters: xi={xi}, delta={delta}")

    test_structures = structures[:max_structures]

    for i, (name, unit_cell) in enumerate(test_structures):
        print(f"  Progress: {i+1}/{len(test_structures)}")

        try:
            # Convert to CNF
            cnf = unit_cell.to_cnf(xi=xi, delta=delta)
            n_atoms = len(cnf.motif_normal_form.elements)

            if use_pure_rust:
                # Use pure Rust implementation
                import rust_cnf

                vonorms_i32 = np.array(cnf.lattice_normal_form.vonorms.tuple, dtype=np.int32)
                coords_i32 = np.array(cnf.motif_normal_form.coord_list, dtype=np.int32)
                elements = [str(el) for el in cnf.motif_normal_form.elements]
                n_atoms_param = len(elements)

                # Warmup
                neighbors = rust_cnf.find_neighbor_tuples_rust(
                    vonorms_i32, coords_i32, elements, n_atoms_param, xi, delta
                )
                n_neighbors = len(neighbors)

                # Benchmark
                times = []
                for _ in range(n_iterations):
                    start = time.perf_counter()
                    neighbors = rust_cnf.find_neighbor_tuples_rust(
                        vonorms_i32, coords_i32, elements, n_atoms_param, xi, delta
                    )
                    elapsed = time.perf_counter() - start
                    times.append(elapsed * 1000)  # Convert to ms
            else:
                # Use NeighborFinder (Python or hybrid)
                finder = NeighborFinder(cnf)
                neighbors = finder.find_neighbor_tuples()
                n_neighbors = len(neighbors)

                # Benchmark
                times = []
                for _ in range(n_iterations):
                    start = time.perf_counter()
                    neighbors = NeighborFinder(cnf).find_neighbor_tuples()
                    elapsed = time.perf_counter() - start
                    times.append(elapsed * 1000)  # Convert to ms

            results.append({
                'name': name,
                'n_atoms': n_atoms,
                'n_neighbors': n_neighbors,
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            })
        except Exception as e:
            print(f"  Warning: Failed to benchmark {name}: {e}")
            continue

    print(f"  Completed: {len(results)}/{len(test_structures)}\n")
    return {
        'results': results,
        'n_iterations': n_iterations
    }


def print_statistics(results: List[Dict[str, Any]], metric: str = 'mean_time', name: str = ""):
    """Print statistical summary of benchmark results."""
    if not results:
        print("No results to display")
        return

    times = [r[metric] for r in results]
    n_atoms_list = [r['n_atoms'] for r in results]

    print(f"{name} Statistics:")
    print(f"  Number of structures: {len(results)}")
    print(f"  Mean time:   {np.mean(times):8.3f} ms")
    print(f"  Median time: {np.median(times):8.3f} ms")
    print(f"  Std dev:     {np.std(times):8.3f} ms")
    print(f"  Min time:    {np.min(times):8.3f} ms")
    print(f"  Max time:    {np.max(times):8.3f} ms")
    print(f"  ")
    print(f"  Atoms range: {np.min(n_atoms_list)} - {np.max(n_atoms_list)} atoms")
    print(f"  Mean atoms:  {np.mean(n_atoms_list):.1f} atoms")

    # Show correlation with structure size
    if len(results) > 1:
        correlation = np.corrcoef(times, n_atoms_list)[0, 1]
        print(f"  Correlation (time vs atoms): {correlation:.3f}")


def analyze_variation(results: List[Dict[str, Any]]):
    """Analyze how much CNF construction time varies with structure."""
    if not results:
        return

    times = [r['mean_time'] for r in results]
    n_atoms = [r['n_atoms'] for r in results]

    print("\nVariation Analysis:")
    print(f"  Time range: {np.min(times):.3f} ms - {np.max(times):.3f} ms")
    print(f"  Coefficient of variation: {np.std(times)/np.mean(times)*100:.1f}%")

    # Find slowest and fastest
    slowest_idx = np.argmax(times)
    fastest_idx = np.argmin(times)

    print(f"\n  Fastest: {results[fastest_idx]['name']}")
    print(f"    Time: {results[fastest_idx]['mean_time']:.3f} ms")
    print(f"    Atoms: {results[fastest_idx]['n_atoms']}")

    print(f"\n  Slowest: {results[slowest_idx]['name']}")
    print(f"    Time: {results[slowest_idx]['mean_time']:.3f} ms")
    print(f"    Atoms: {results[slowest_idx]['n_atoms']}")


def main():
    """Run the profiling benchmarks."""

    # Check if using pure Rust mode
    use_pure_rust = os.getenv("USE_RUST") is not None
    implementation = "Pure Rust" if use_pure_rust else "Python"

    print("=" * 80)
    print(f"CNF Performance Profiling - {implementation}")
    print("=" * 80)
    print()

    # Load structures
    structures = load_structures(max_structures=100)

    if not structures:
        print("Error: No structures loaded")
        sys.exit(1)

    # Benchmark CNF construction
    print("-" * 80)
    cnf_results = benchmark_cnf_construction(structures, n_iterations=3)
    print_statistics(cnf_results['results'], name="CNF Construction")
    analyze_variation(cnf_results['results'])

    # Benchmark neighbor finding (on subset)
    print("\n" + "-" * 80)
    neighbor_results = benchmark_neighbor_finding(
        structures,
        max_structures=20,
        n_iterations=2,
        use_pure_rust=use_pure_rust
    )
    print_statistics(neighbor_results['results'], name="Neighbor Finding")

    print("\n" + "=" * 80)
    print(f"Profiling complete using {implementation} implementation")
    print("=" * 80)


if __name__ == "__main__":
    main()
