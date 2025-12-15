import numpy as np
import itertools
from ..crystal_normal_form import CrystalNormalForm
from ..cnf_constructor import CNFConstructor
from ..lattice.lnf_constructor import LatticeNormalFormConstructor
from ..lattice.lattice_normal_form import LatticeNormalForm
from ..lattice.voronoi import VonormList
from ..motif.motif_normal_form import MotifNormalForm
from .lattice_step import LatticeStep, LatticeStepResult
from .neighbor_set import NeighborSet
from ..utils.pdd import pdd_for_cnfs
from ..linalg.unimodular import combine_unimodular_matrices
from ..utils.prof import maybe_profile
from ..motif.atomic_motif import DiscretizedMotif

class LatticeNeighborFinder():

    def __init__(self, cnf_point: CrystalNormalForm, verbose_logging=False):
        self.verbose_logging = verbose_logging
        self.point = cnf_point

    def _log(self, msg):
        if self.verbose_logging:
            print(msg)

    def _get_atoms_list(self):
        """Extract atoms list from motif normal form."""
        return list(self.point.motif_normal_form.elements)

    def _get_stabilizers(self):
        """Extract stabilizer matrices from lattice normal form."""
        return self.point.lattice_normal_form.vonorms.stabilizer_matrices_fast()

    def _extract_coord_matrix_from_mnf(self, include_origin: bool = False):
        """
        Reconstruct coordinate matrix from MNF coord_list.

        Args:
            include_origin: If True, prepend origin atom at (0,0,0)

        Returns:
            coord_matrix: (3, N) or (3, N-1) array depending on include_origin
            n_atoms: Total number of atoms (including origin)
            motif_delta: Delta value from MNF
        """
        mnf = self.point.motif_normal_form
        motif_delta = mnf.delta

        # Reconstruct coord_matrix from coord_list (which excludes origin atom)
        n_stored_atoms = len(mnf.coord_list) // 3
        coords_array = np.array(mnf.coord_list, dtype=np.int32).reshape(n_stored_atoms, 3)

        if include_origin:
            # Prepend origin atom for Rust (needs full structure for transformations)
            coords_with_origin = np.vstack([np.array([[0, 0, 0]]), coords_array])
            motif_coord_matrix = coords_with_origin.T  # (3, N) - with origin atom
            n_atoms = len(coords_with_origin)
        else:
            # No origin for Python (transformation of (0,0,0) is always (0,0,0))
            motif_coord_matrix = coords_array.T  # (3, N-1) - no origin atom
            n_atoms = n_stored_atoms + 1  # +1 for implicit origin

        return motif_coord_matrix, n_atoms, motif_delta

    def find_neighbor_tuples(self) -> list[tuple]:
        """
        Find lattice neighbor tuples without constructing CNF objects.

        Returns list of (vonorms_tuple, coords_tuple) for each neighbor.
        This is faster than find_cnf_neighbors() when you don't need full objects.

        Note: Always uses Python implementation. For pure Rust, use NeighborFinder directly.
        """
        return self._find_neighbor_tuples_python()

    def tuples_to_cnf_neighbors(self, neighbor_tuples: list[tuple]) -> list[CrystalNormalForm]:
        """
        Convert neighbor tuples to CNF objects.

        Args:
            neighbor_tuples: List of (vonorms_tuple, coords_tuple)

        Returns:
            List of CrystalNormalForm objects
        """
        atoms = self._get_atoms_list()
        xi = float(self.point.xi)
        delta = int(self.point.delta)

        results = []
        for vonorms_tuple, coords_tuple in neighbor_tuples:
            vonorms = VonormList(vonorms_tuple)
            lnf = LatticeNormalForm(vonorms, xi)

            mnf = MotifNormalForm(
                coords_tuple,
                atoms,
                delta
            )

            cnf = CrystalNormalForm(lnf, mnf)

            # Only include if different from input CNF
            if cnf != self.point:
                results.append(cnf)

        return results

    def find_cnf_neighbors(self) -> list[CrystalNormalForm]:
        """Fast neighbor finding that avoids intermediate object creation."""
        neighbor_tuples = self.find_neighbor_tuples()
        return self.tuples_to_cnf_neighbors(neighbor_tuples)
    
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

    def _find_neighbor_tuples_rust(self) -> list[tuple]:
        """Rust implementation - returns tuples without constructing CNF objects."""
        import rust_cnf

        # Prepare all inputs for combined Rust function
        vonorms = self.point.lattice_normal_form.vonorms
        input_vonorms = np.array(vonorms.vonorms, dtype=np.float64)

        # Get motif data using helper (Rust needs origin atom)
        atoms = [str(atom) for atom in self.point.motif_normal_form.elements]
        motif_coord_matrix, n_atoms, motif_delta = self._extract_coord_matrix_from_mnf(include_origin=True)
        motif_coords_flat = np.ascontiguousarray(motif_coord_matrix.flatten(), dtype=np.float64)

        xi = float(self.point.xi)
        delta = int(self.point.delta)

        # Single Rust call that does: step generation, validation, and canonicalization
        # Returns list of (vonorms_tuple, coords_tuple) - already as tuples with ints
        # Note: Stabilizers are computed internally in Rust
        return rust_cnf.find_and_canonicalize_lattice_neighbors(
            input_vonorms, motif_coords_flat,
            n_atoms, motif_delta, atoms, xi, delta
        )

    def _find_neighbor_tuples_python(self) -> list[tuple]:
        """Python implementation - returns tuples without constructing CNF objects."""
        # Get raw step data without creating LatticeStep objects
        step_data = self._compute_step_data_raw_python()

        # Filter valid steps and construct CNFs in batch
        valid_step_data = self.validate_step_data(step_data)

        # Batch canonicalize using tuple method (avoids CNF object construction)
        cnf_constructor = CNFConstructor(self.point.xi, self.point.delta, verbose_logging=False)
        atoms = self._get_atoms_list()

        results = []
        for step_vec, vonorms_tuple, motif_coords, matrix in valid_step_data:
            # Convert to tuple format for canonicalize_tuple
            coords_tuple = tuple(motif_coords.flatten())
            cnf_tuple = (vonorms_tuple, coords_tuple)

            # Canonicalize using tuple method (avoids DiscretizedMotif construction)
            canonical_tuple = cnf_constructor.canonicalize_tuple(cnf_tuple, atoms)

            results.append(canonical_tuple)

        return results

    def _compute_step_data_raw_python(self):
        """Python implementation of step data computation."""
        vonorms = self.point.lattice_normal_form.vonorms
        step_data = []
        current_stabilizer = self._get_stabilizers()

        # Get motif data using helper (Python doesn't need origin)
        motif_coord_matrix, _, motif_delta = self._extract_coord_matrix_from_mnf(include_origin=False)

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
                transformed_coords = np.mod(transformed_coords, motif_delta).T  # (N, 3)

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

    def _compute_step_data_raw_rust(self):
        """Rust implementation of step data computation."""
        import rust_cnf

        vonorms = self.point.lattice_normal_form.vonorms

        # Get motif data using helper (Rust needs origin atom)
        motif_coord_matrix, n_atoms, motif_delta = self._extract_coord_matrix_from_mnf(include_origin=True)

        # Prepare input vonorms - Rust will compute S4 groups and stabilizers internally
        input_vonorms = np.array(vonorms.vonorms, dtype=np.float64)

        # Call Rust function - Rust now handles stabilizers and S4 grouping internally!
        motif_coords_flat = np.ascontiguousarray(motif_coord_matrix.flatten(), dtype=np.float64)
        result = rust_cnf.compute_step_data_raw_rust(
            input_vonorms,
            motif_coords_flat,
            n_atoms,
            motif_delta
        )

        # Convert result back to Python format
        # Result is list of (step_vec, vonorms_tuple, coords, matrix_flat)
        # Note: coords from Rust include origin, need to remove it to match Python format
        python_result = []
        for step_vec, vonorms_list, coords_flat, mat_flat in result:
            vonorms_tuple = tuple([int(v) for v in vonorms_list])
            # Rust returns coords WITH origin, Python expects WITHOUT origin
            coords_with_origin = np.array(coords_flat).reshape(n_atoms, 3)
            coords = coords_with_origin[1:]  # Remove origin atom
            matrix = np.array(mat_flat, dtype=np.int32).reshape(3, 3)
            python_result.append((step_vec, vonorms_tuple, coords, matrix))

        return python_result