#!/usr/bin/env python3
"""
Profile the breakdown of neighbor TUPLE finding to understand where time is spent.
"""

import time
import numpy as np
from pathlib import Path
from cnf import UnitCell
from cnf.navigation.neighbor_finder import NeighborFinder

# Test on a few structures
test_structures = [
    'tests/data/mp_cifs/mp-66.cif',  # Simple
    'TiO2_anatase.cif',  # Medium - the one we line profiled
    'tests/data/mp_cifs/mp-1019569.cif',  # Complex (168 atoms)
]

xi = 0.01
delta = 10

for cif_file in test_structures:
    try:
        print(f"\n{'='*80}")
        print(f"Structure: {Path(cif_file).stem}")
        print(f"{'='*80}")

        # Load and convert
        unit_cell = UnitCell.from_cif(cif_file)
        pmg = unit_cell.to_pymatgen_structure()
        print(f"Atoms: {len(pmg)}")

        cnf = unit_cell.to_cnf(xi=xi, delta=delta)

        # Time neighbor finding with breakdown
        finder = NeighborFinder(cnf)

        # Patch the methods to add timing (using tuple methods)
        from cnf.navigation.lattice_neighbor_finder import LatticeNeighborFinder
        from cnf.navigation.motif_neighbor_finder import MotifNeighborFinder

        original_lattice_find = LatticeNeighborFinder.find_neighbor_tuples
        original_motif_find = MotifNeighborFinder.find_neighbor_tuples

        lattice_times = []
        motif_times = []

        def timed_lattice_find(self):
            start = time.perf_counter()
            result = original_lattice_find(self)
            elapsed = time.perf_counter() - start
            lattice_times.append(elapsed)
            return result

        def timed_motif_find(self):
            start = time.perf_counter()
            result = original_motif_find(self)
            elapsed = time.perf_counter() - start
            motif_times.append(elapsed)
            return result

        LatticeNeighborFinder.find_neighbor_tuples = timed_lattice_find
        MotifNeighborFinder.find_neighbor_tuples = timed_motif_find

        # Run neighbor finding (tuples only)
        start_total = time.perf_counter()
        neighbors = finder.find_neighbor_tuples()
        total_time = time.perf_counter() - start_total

        # Restore original methods
        LatticeNeighborFinder.find_neighbor_tuples = original_lattice_find
        MotifNeighborFinder.find_neighbor_tuples = original_motif_find

        print(f"Neighbors found: {len(neighbors)}")
        print(f"\nTime breakdown:")
        print(f"  Total time:          {total_time*1000:8.2f} ms")
        print(f"  Lattice neighbors:   {sum(lattice_times)*1000:8.2f} ms  ({sum(lattice_times)/total_time*100:5.1f}%)")
        print(f"  Motif neighbors:     {sum(motif_times)*1000:8.2f} ms  ({sum(motif_times)/total_time*100:5.1f}%)")
        print(f"  Other overhead:      {(total_time - sum(lattice_times) - sum(motif_times))*1000:8.2f} ms  ({(total_time - sum(lattice_times) - sum(motif_times))/total_time*100:5.1f}%)")

        print(f"\nPer-neighbor timing:")
        print(f"  Lattice find calls:  {len(lattice_times)}")
        print(f"  Motif find calls:    {len(motif_times)}")
        if lattice_times:
            print(f"  Avg lattice time:    {np.mean(lattice_times)*1000:8.2f} ms")
        if motif_times:
            print(f"  Avg motif time:      {np.mean(motif_times)*1000:8.2f} ms")

    except Exception as e:
        print(f"Error processing {cif_file}: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*80}")
print("Analysis complete")
print(f"{'='*80}")
