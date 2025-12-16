from cnf.lattice.voronoi import VonormList
from cnf.motif.mnf_constructor import MNFConstructor, extract_coord_matrix_from_mnf_tuple, sort_motif_coord_arr
from cnf.motif.atomic_motif import DiscretizedMotif
from cnf.motif.utils import move_coords_into_cell
from ..utils.config import should_use_rust
from ..crystal_normal_form import CrystalNormalForm
from pymatgen.core.periodic_table import Element
import copy

class MotifNeighborFinder():

    @classmethod
    def from_cnf(cls, pt: CrystalNormalForm):
        return cls(pt.xi, pt.delta, pt.elements)

    def __init__(self, xi: float, delta: int, elements: list[str]):
        self.xi = xi
        self.delta = delta
        self.elements = elements
        self.atomic_numbers = [Element(el).Z for el in elements]

    def find_neighbor_tuples(self, cnf_tuple: tuple):
        if should_use_rust():
            return self._find_neighbor_tuples_rust(cnf_tuple)
        else:
            return self._find_neighbor_tuples_py(cnf_tuple)

    def _find_neighbor_tuples_rust(self, cnf_tuple: tuple):
        """
        Rust implementation of motif neighbor finding.

        Returns list of (vonorms_tuple, coords_tuple) for each neighbor.
        """
        import rust_cnf
        import numpy as np

        # Get input data
        motif_coords = cnf_tuple[7:]  # WITHOUT origin

        # Get stabilizers
        pt_vonorms = VonormList(cnf_tuple[:7])
        self_stabs = pt_vonorms.stabilizer_matrices_fast()
        stabilizers_flat = np.array([s.matrix for s in self_stabs]).astype(np.int32).flatten()

        # Call Rust function - returns coords as tuples already
        canonical_coords_list = rust_cnf.find_and_canonicalize_motif_neighbors(
            motif_coords,
            self.elements,
            stabilizers_flat,
            self.delta
        )

        # Pair with vonorms tuple
        vonorms_tuple = pt_vonorms.tuple
        return [(*vonorms_tuple, *coords_tuple) for coords_tuple in canonical_coords_list]

    def _find_neighbor_tuples_py(self, cnf_tuple: tuple):
        """
        Find motif neighbor tuples without constructing CNF objects.

        Returns list of (vonorms_tuple, coords_tuple, affected_idxs, adj) for each neighbor.
        This is faster than find_motif_neighbors() when you don't need full objects.

        Note: Always uses Python implementation. For pure Rust, use NeighborFinder directly.
        """
        neighbor_mnf_tuples = []
        pt_vonorms = VonormList(cnf_tuple[:7], conorm_tol=0)
        curr_coords, _ = extract_coord_matrix_from_mnf_tuple(cnf_tuple[7:], include_origin=False)
        disc = DiscretizedMotif.from_mnf_list(cnf_tuple[7:], self.elements, delta=self.delta)
        self_stabs = pt_vonorms.stabilizer_matrices_fast()
        # Try ALL the stabilizers!
        for stabilizer in self_stabs:
            current_mnf_tuple = disc.apply_unimodular(stabilizer, skip_det_check=True).to_mnf_list(sort=True)
            current_mnf_tuple = (0, 0, 0) + current_mnf_tuple
            # inv = stabilizer.inverse().matrix
            # # curr_cords has shape (3, N) and inv is (3,3) so transformed is (3, N)
            # transformed = inv @ curr_coords
            # positions = move_coords_into_cell(transformed, self.delta)
            # positions = sort_motif_coord_arr(positions, self.atomic_numbers)
            # current_mnf_tuple = [0, 0, 0] + positions.T.flatten()
            for idx in range(len(current_mnf_tuple)):
                for adj in [-1, +1]:
                    n = list(copy.copy(current_mnf_tuple))
                    n[idx] = n[idx] + adj
                    neighbor_mnf_tuples.append(n)
            
            for idx in range(0, len(current_mnf_tuple), 3):
                for adj in [-1, +1]:
                    n = list(copy.copy(current_mnf_tuple))
                    n[idx] = n[idx] + adj
                    n[idx + 1] = n[idx + 1] + adj
                    n[idx + 2] = n[idx + 2] + adj
                    neighbor_mnf_tuples.append(n)

        stabilizers = [s.matrix for s in self_stabs]

        # Batch build MNFs (always use Python in this path)
        mnf_constructor = MNFConstructor(self.delta, stabilizers)
        mnf_coords_list = mnf_constructor.build_many_from_raw_coords(neighbor_mnf_tuples, self.atomic_numbers, use_rust=False)

        results = [pt_vonorms.tuple + mnf_coords for mnf_coords in mnf_coords_list]

        return results