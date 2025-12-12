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

        if self._is_cnf_neighbor_finder():
            self.discretized_motif = cnf_point.motif_normal_form.to_discretized_motif()
            self.fractional_motif = cnf_point.motif_normal_form.to_motif()

    def _is_cnf_neighbor_finder(self):
        return isinstance(self.point, CrystalNormalForm)
    
    def _lnf(self):
        if self._is_cnf_neighbor_finder():
            return self.point.lattice_normal_form
        else:
            return self.point

    def _log(self, msg):
        if self.verbose_logging:
            print(msg)
    
    @maybe_profile
    def find_cnf_neighbor_results(self, step: LatticeStep) -> list[LatticeStepResult]:
        results = []

        if not step.vonorms.has_valid_conorms_exact():
            self._log("Neighbor had invalid conorms")
            return results

        is_obtuse = step.vonorms.is_obtuse()
        is_sb = step.vonorms.is_superbasis_exact()
        if not (is_obtuse and is_sb):
            if self.verbose_logging:
                if not step.vonorms.is_obtuse():
                    self._log(f"Neighbor was not obtuse.")

                if not step.vonorms.is_superbasis():
                    self._log(f"Neighbor was not a superbasis.")
            return results

        cnf_constructor = CNFConstructor(
            self.point.xi,
            self.point.delta,
            verbose_logging=False,
        )

        cnf_result = cnf_constructor.from_vonorms_and_motif(step.vonorms, step.transformed_motif)

        if cnf_result.cnf != self.point:
            results.append(LatticeStepResult(
                step,
                cnf_result.cnf.lattice_normal_form.vonorms,
                cnf_result,
                cnf_result.cnf,
                step.matrix
            ))

        return results

    def find_cnf_neighbors(self) -> list[CrystalNormalForm]:
        """Fast neighbor finding that avoids intermediate object creation."""
        import os

        # When using Rust, use the complete Rust pipeline
        use_rust = os.getenv("USE_RUST") is not None
        if use_rust:
            return self._find_cnf_neighbors_fast_rust()

        # Python fallback
        return self._find_cnf_neighbors_fast_python()
    
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

    def _find_cnf_neighbors_fast_rust(self) -> list:
        """Rust implementation of CNF neighbor finding with complete pipeline in Rust."""
        import rust_cnf

        # Prepare all inputs for combined Rust function
        vonorms = self._lnf().vonorms
        current_stabilizer = self.point.lattice_normal_form.vonorms.stabilizer_matrices_fast()
        current_stabilizers_flat = np.array([s.matrix for s in current_stabilizer]).astype(np.int32).flatten()

        input_vonorms = np.array(vonorms.vonorms, dtype=np.float64)
        motif_coord_matrix = self.discretized_motif.coord_matrix
        motif_coords_flat = np.ascontiguousarray(motif_coord_matrix.flatten(), dtype=np.float64)
        n_atoms = len(self.discretized_motif.atoms)
        motif_delta = self.discretized_motif._mod

        atoms = [str(atom) for atom in self.discretized_motif.atoms]
        xi = float(self.point.xi)
        delta = int(self.point.delta)

        # Single Rust call that does: step generation, validation, and canonicalization
        canonical_results = rust_cnf.find_and_canonicalize_lattice_neighbors(
            current_stabilizers_flat, input_vonorms, motif_coords_flat,
            n_atoms, motif_delta, atoms, xi, delta
        )

        # Convert canonical results to Python CNF objects
        results = []
        for canonical_vonorms_list, canonical_coords_list in canonical_results:
            # Data from Rust is already canonical - just wrap in Python objects
            vonorms = VonormList(tuple([int(v) for v in canonical_vonorms_list[:7]]))
            lnf = LatticeNormalForm(vonorms, xi)

            # Create MNF from canonical coords
            mnf = MotifNormalForm(
                tuple([int(c) for c in canonical_coords_list]),
                atoms,  # element_list
                delta
            )

            # Create CNF
            cnf = CrystalNormalForm(lnf, mnf)

            # Only include if different from input CNF
            if cnf != self.point:
                results.append(cnf)

        return results

    def _find_cnf_neighbors_fast_python(self) -> list:
        """Python implementation of CNF neighbor finding."""
        # Get raw step data without creating LatticeStep objects
        step_data = self._compute_step_data_raw_python()

        # Filter valid steps and construct CNFs in batch
        valid_cnfs = []
        valid_step_data = []

        for step_vec, vonorms_tuple, motif_coords, matrix in step_data:
            # Create VonormList
            vonorms = VonormList(vonorms_tuple)

            # Validate
            if not vonorms.has_valid_conorms_exact():
                continue
            if not (vonorms.is_obtuse() and vonorms.is_superbasis_exact()):
                continue

            valid_step_data.append((vonorms, motif_coords, matrix))

        # Batch construct CNFs
        cnf_constructor = CNFConstructor(self.point.xi, self.point.delta, verbose_logging=False)

        for vonorms, motif_coords, matrix in valid_step_data:
            # Create DiscretizedMotif only when needed for CNF construction
            from ..motif.atomic_motif import DiscretizedMotif
            motif = DiscretizedMotif.from_elements_and_positions(
                self.discretized_motif.atoms,
                motif_coords.reshape(-1, 3),
                delta=self.point.delta
            )

            cnf_result = cnf_constructor.from_vonorms_and_motif(vonorms, motif)

            if cnf_result.cnf != self.point:
                valid_cnfs.append(cnf_result.cnf)

        return valid_cnfs

    def _compute_step_data_raw_python(self):
        """Python implementation of step data computation."""
        vonorms = self._lnf().vonorms
        step_data = []
        current_stabilizer = self.point.lattice_normal_form.vonorms.stabilizer_matrices()

        # Get motif data once
        motif_coord_matrix = self.discretized_motif.coord_matrix
        motif_atoms = self.discretized_motif.atoms
        motif_delta = self.discretized_motif._mod

        for _, data in vonorms.s4_equivalence_class_representatives().items():
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
            s2_mats = np.array([s.matrix for s in permuted_vonorms.stabilizer_matrices()])

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

        vonorms = self._lnf().vonorms
        current_stabilizer = self.point.lattice_normal_form.vonorms.stabilizer_matrices_fast()

        # Get motif data
        motif_coord_matrix = self.discretized_motif.coord_matrix
        n_atoms = len(self.discretized_motif.atoms)
        motif_delta = self.discretized_motif._mod

        # Prepare current stabilizers as flat array
        current_stabilizers_flat = np.array([s.matrix for s in current_stabilizer]).astype(np.int32).flatten()

        # Prepare input vonorms - Rust will compute S4 groups internally using precomputed data
        input_vonorms = np.array(vonorms.vonorms, dtype=np.float64)

        # Call Rust function - Rust now handles the S4 grouping internally!
        motif_coords_flat = np.ascontiguousarray(motif_coord_matrix.flatten(), dtype=np.float64)
        result = rust_cnf.compute_step_data_raw_rust(
            current_stabilizers_flat,
            input_vonorms,
            motif_coords_flat,
            n_atoms,
            motif_delta
        )

        # Convert result back to Python format
        # Result is list of (step_vec, vonorms_tuple, coords, matrix_flat)
        python_result = []
        for step_vec, vonorms_list, coords_flat, mat_flat in result:
            vonorms_tuple = tuple([int(v) for v in vonorms_list])
            coords = np.array(coords_flat).reshape(n_atoms, 3)
            matrix = np.array(mat_flat, dtype=np.int32).reshape(3, 3)
            python_result.append((step_vec, vonorms_tuple, coords, matrix))

        return python_result