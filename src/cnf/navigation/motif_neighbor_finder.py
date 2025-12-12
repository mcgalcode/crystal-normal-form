from cnf import CrystalNormalForm
from cnf.cnf_constructor import should_use_rust
from cnf.motif import MotifNormalForm
from cnf.motif.mnf_constructor import MNFConstructor
from cnf.lattice.lattice_normal_form import LatticeNormalForm
from .neighbor_set import NeighborSet
from pymatgen.core.periodic_table import Element
import copy

class MotifStepResult():

    def __init__(self, result: CrystalNormalForm, affected_idxs: list[int], adj: int):
        self.result = result
        self.affected_idxs = affected_idxs
        self.adj = adj

class MotifNeighborFinder():

    def __init__(self, point: CrystalNormalForm):
        self.point = point
    


    def find_neighbor_tuples(self):
        """
        Find neighbor tuples without constructing CNF objects.

        Returns list of (vonorms_tuple, coords_tuple, affected_idxs, adj) for each neighbor.
        This is faster than find_motif_neighbors() when you don't need full objects.
        """
        neighbor_mnf_tuples = []
        self_disc = self.point.motif_normal_form.to_discretized_motif()
        self_stabs = self.point.lattice_normal_form.vonorms.stabilizer_matrices_fast()
        # Try ALL the stabilizers!
        for stabilizer in self_stabs:
            current_mnf_tuple = self_disc.apply_unimodular(stabilizer).to_mnf_list(sort=True)
            current_mnf_tuple = (0,0,0) + current_mnf_tuple
            for idx in range(len(current_mnf_tuple)):
                for adj in [-1, +1]:
                    n = list(copy.copy(current_mnf_tuple))
                    n[idx] = n[idx] + adj
                    neighbor_mnf_tuples.append((tuple(n), [idx], adj))
            
            for idx in range(0, len(current_mnf_tuple), 3):
                for adj in [-1, +1]:
                    n = list(copy.copy(current_mnf_tuple))
                    n[idx] = n[idx] + adj
                    n[idx + 1] = n[idx + 1] + adj
                    n[idx + 2] = n[idx + 2] + adj
                    neighbor_mnf_tuples.append((tuple(n), [idx, idx + 1, idx + 2], adj))

        original_els = self.point.motif_normal_form.elements

        # Pre-compute values that don't change across neighbors
        vonorms = self.point.lattice_normal_form.vonorms
        atomic_numbers = [Element(el).Z for el in original_els]
        stabilizers = [s.matrix for s in self_stabs]

        # Extract just the coordinate tuples for batch processing
        coords_list = [list(nb_mnf) for nb_mnf, _, _ in neighbor_mnf_tuples]

        # Batch build MNFs
        USE_RUST = should_use_rust()

        mnf_constructor = MNFConstructor(self.point.delta, stabilizers)
        mnf_coords_list = mnf_constructor.build_many_from_raw_coords(coords_list, atomic_numbers, use_rust=USE_RUST)

        # Return tuples with metadata
        results = []
        vonorms_tuple = vonorms.tuple
        for (nb_mnf, affected_idxs, adj), mnf_coords in zip(neighbor_mnf_tuples, mnf_coords_list):
            results.append((vonorms_tuple, mnf_coords, affected_idxs, adj))

        return results

    def tuples_to_motif_neighbors(self, neighbor_tuples):
        """
        Convert neighbor tuples to MotifStepResult objects.

        Args:
            neighbor_tuples: List of (vonorms_tuple, coords_tuple, affected_idxs, adj)

        Returns:
            NeighborSet containing MotifStepResult objects
        """
        nbs = NeighborSet()
        original_els = self.point.motif_normal_form.elements
        xi = self.point.xi
        delta = self.point.delta

        for vonorms_tuple, mnf_coords, affected_idxs, adj in neighbor_tuples:
            # Create objects
            vonorms = self.point.lattice_normal_form.vonorms  # Reuse same vonorms object
            lnf = LatticeNormalForm(vonorms, xi)
            mnf = MotifNormalForm(mnf_coords, original_els, delta)
            cnf = CrystalNormalForm(lnf, mnf)
            result = MotifStepResult(cnf, affected_idxs, adj)
            nbs.add_neighbor(result)

        return nbs

    def find_motif_neighbors(self):
        """Find motif neighbors."""
        neighbor_tuples = self.find_neighbor_tuples()
        return self.tuples_to_motif_neighbors(neighbor_tuples)