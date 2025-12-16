import numpy as np
from ..crystal_normal_form import CrystalNormalForm
from ..cnf_constructor import CNFConstructor
from ..lattice.lattice_normal_form import LatticeNormalForm
from ..lattice.voronoi import VonormList
from ..motif.motif_normal_form import MotifNormalForm
from ..motif.mnf_constructor import extract_coord_matrix_from_mnf_tuple
from .lattice_step import LatticeStep
from ..utils.config import should_use_rust

class LatticeNeighborFinder():

    @classmethod
    def from_cnf(cls, pt: CrystalNormalForm):
        return cls(pt.xi, pt.delta, pt.elements)    

    def __init__(self, xi: float, delta: int, elements: list[str], verbose_logging=False):
        self.verbose_logging = verbose_logging
        self.xi = xi
        self.delta = delta
        self.elements = elements

    def _log(self, msg):
        if self.verbose_logging:
            print(msg)

    def find_neighbor_tuples(self, cnf_tuple: tuple) -> list[tuple]:
        """
        Find lattice neighbor tuples without constructing CNF objects.

        Returns list of (vonorms_tuple, coords_tuple) for each neighbor.
        This is faster than find_cnf_neighbors() when you don't need full objects.

        Note: Always uses Python implementation. For pure Rust, use NeighborFinder directly.
        """
        if should_use_rust():
            return self._find_neighbor_tuples_rust(cnf_tuple)
        else:
            return self._find_neighbor_tuples_python(cnf_tuple)
   
    def validate_step_data(self, step_data):
        validated_steps = []
        for step_vec, vonorms_tuple, motif_coords, matrix in step_data:
            vonorms = VonormList(vonorms_tuple)

            # Validate
            if not vonorms.has_valid_conorms_exact():
                continue
            if not (vonorms.is_obtuse() and vonorms.is_superbasis_exact()):
                continue

            validated_steps.append((step_vec, vonorms_tuple, motif_coords, matrix))
        return validated_steps

    def _find_neighbor_tuples_rust(self, cnf_tuple: tuple) -> list[tuple]:
        """Rust implementation - returns tuples without constructing CNF objects."""
        import rust_cnf

        # Prepare all inputs for combined Rust function
        vonorms = cnf_tuple[:7]
        input_vonorms = np.array(vonorms, dtype=np.float64)

        # Get motif data using helper (Rust needs origin atom)
        motif_coord_matrix, n_atoms = extract_coord_matrix_from_mnf_tuple(cnf_tuple[7:], include_origin=True)
        motif_coords_flat = np.ascontiguousarray(motif_coord_matrix.flatten(), dtype=np.float64)

        # Single Rust call that does: step generation, validation, and canonicalization
        # Returns list of (vonorms_tuple, coords_tuple) - already as tuples with ints
        # Note: Stabilizers are computed internally in Rust
        results = rust_cnf.find_and_canonicalize_lattice_neighbors(
            input_vonorms, motif_coords_flat,
            n_atoms, self.delta, self.elements, self.xi, self.delta
        )
        return [(*v, *c) for v, c in results]

    def _find_neighbor_tuples_python(self, cnf_tuple: tuple) -> list[tuple]:
        """Python implementation - returns tuples without constructing CNF objects."""
        # Get raw step data without creating LatticeStep objects
        step_data = self._compute_step_data_raw_python(cnf_tuple)

        # Filter valid steps and construct CNFs in batch
        valid_step_data = self.validate_step_data(step_data)

        # Batch canonicalize using tuple method (avoids CNF object construction)
        cnf_constructor = CNFConstructor(self.xi, self.delta, verbose_logging=False)

        results = []
        for step_vec, vonorms_tuple, motif_coords, matrix in valid_step_data:
            # Convert to tuple format for canonicalize_tuple
            coords_tuple = tuple(motif_coords.flatten())
            cnf_tuple = tuple([*vonorms_tuple, *coords_tuple])

            # Canonicalize using tuple method (avoids DiscretizedMotif construction)
            canonical_tuple = cnf_constructor.canonicalize_tuple(cnf_tuple, self.elements)

            results.append(canonical_tuple)

        return results

    def _compute_step_data_raw_python(self, cnf_tuple: tuple):
        """Python implementation of step data computation."""
        vonorms = VonormList(cnf_tuple[:7])
        step_data = []
        current_stabilizer = vonorms.stabilizer_matrices_fast()

        # Get motif data using helper (Python doesn't need origin)
        motif_coord_matrix, _, = extract_coord_matrix_from_mnf_tuple(cnf_tuple[7:], include_origin=False)

        for _, data in vonorms.maximally_ascending_equivalence_class_members().items():
            permuted_vonorms = data['permuted_vonorms']
            transform_mats = data['transition_mats']
            # We launch the neighbor finding process from a single arbitrarily chosen
            # member from each group of permutations where the _primary vonorm indices_ are
            # the same.
            #
            # To do this, we search motifs produced by matrices belonging to the cartesian
            # product:
            #   original point stablizer x 
            #   (a single transformation matrix from the original primary vonorm set to the new primary vonorm set)
            #   the stabilizer of the other primary vonorm set
            s1_mats = np.array([s.matrix for s in current_stabilizer])
            t_mats = np.array([t.matrix for t in transform_mats[:1]])
            s2_mats = np.array([s.matrix for s in permuted_vonorms.stabilizer_matrices_fast()])

            # Batch compute all matrix products using einsum
            all_products = np.einsum('nij,mjk,okl->nmoil', s1_mats, t_mats, s2_mats)
            all_products_flat = all_products.reshape(-1, 3, 3)

            # Deduplicate
            all_products_rounded = np.round(all_products_flat).astype(int)
            all_products_2d = all_products_rounded.reshape(-1, 9)
            unique_flat = np.unique(all_products_2d, axis=0)
            unimodular_mats = unique_flat.reshape(-1, 3, 3)

            # These step vecs are the canonical vonorm adjustments (+1,-1) pairs
            # described in Dr. Mrdjenovich's thesis.
            step_vecs = LatticeStep.all_step_vecs()
            permuted_vonorms_arr = np.array(permuted_vonorms.vonorms)

            for mat in unimodular_mats:
                # Transform motif coordinates directly
                mat_inv = np.linalg.inv(mat).astype(int)
                transformed_coords = mat_inv @ motif_coord_matrix
                transformed_coords = np.mod(transformed_coords, self.delta).T  # (N, 3)

                for step_vec in step_vecs:
                    # Compute new vonorms by applying the step vec adjustment (a pair of +1, -1) at
                    # particular indices
                    new_vonorms_arr = permuted_vonorms_arr + np.array(step_vec)
                    new_vonorms_tuple = tuple(new_vonorms_arr.astype(int))

                    # Store the data associated with this step
                    step_data.append((step_vec, new_vonorms_tuple, transformed_coords, mat))

        # Deduplicate by vonorms and transformed coords
        unique_steps = {}
        for step_vec, vonorms_tuple, coords, mat in step_data:
            # Use vonorms and flattened coords as key
            coords_tuple = tuple(coords.flatten())
            key = (vonorms_tuple, coords_tuple)
            if key not in unique_steps:
                unique_steps[key] = (step_vec, vonorms_tuple, coords, mat)

        return list(unique_steps.values())