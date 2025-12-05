import numpy as np
import itertools
from ..crystal_normal_form import CrystalNormalForm
from ..cnf_constructor import CNFConstructor
from ..lattice.lnf_constructor import LatticeNormalFormConstructor
from ..lattice.voronoi import VonormList
from .lattice_step import LatticeStep, LatticeStepResult
from .neighbor_set import NeighborSet
from ..utils.pdd import pdd_for_cnfs
from ..linalg.unimodular import combine_unimodular_matrices
from ..utils.prof import maybe_profile

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

    def possible_steps(self):
        vonorms = self._lnf().vonorms
        # Use tuples for deduplication, defer LatticeStep creation
        step_tuples = []
        current_stabilizer = self.point.lattice_normal_form.vonorms.stabilizer_matrices()

        for s4_idxs, data in vonorms.s4_equivalence_class_representatives().items():
            permuted_vonorms = data['permuted_vonorms']
            transform_mats = data['transition_mats']

            # Batch compute all matrix products using einsum
            # Convert to numpy arrays
            s1_mats = np.array([s.matrix for s in current_stabilizer])  # (N1, 3, 3)
            t_mats = np.array([t.matrix for t in transform_mats])  # (N2, 3, 3)
            s2_mats = np.array([s.matrix for s in permuted_vonorms.stabilizer_matrices()])  # (N3, 3, 3)

            # Compute all s1 @ t @ s2 combinations using einsum
            # Result shape: (N1, N2, N3, 3, 3)
            all_products = np.einsum('nij,mjk,okl->nmoil', s1_mats, t_mats, s2_mats)

            # Reshape to (N1*N2*N3, 3, 3)
            all_products_flat = all_products.reshape(-1, 3, 3)

            # Deduplicate using numpy.unique (much faster than dict+tuple)
            # Flatten to (N, 9) for unique operation
            all_products_rounded = np.round(all_products_flat).astype(int)
            all_products_2d = all_products_rounded.reshape(-1, 9)

            # Find unique rows
            unique_flat = np.unique(all_products_2d, axis=0)

            # Reshape back to (M, 3, 3) where M is number of unique matrices
            unimodular_mats = unique_flat.reshape(-1, 3, 3)

            # Get step vectors once
            step_vecs = LatticeStep.all_step_vecs()
            permuted_vonorms_arr = np.array(permuted_vonorms.vonorms)

            for mat in unimodular_mats:
                from ..linalg import MatrixTuple
                mat_tuple = MatrixTuple(mat)
                # Compute transformed_motif once per matrix
                transformed_motif = self.discretized_motif.apply_unimodular(mat_tuple)
                transformed_motif_mnf = transformed_motif.to_mnf_list()

                # Pre-compute matrix tuple
                mat_tuple_tuple = mat_tuple.tuple

                for step_vec in step_vecs:
                    step_vec_tuple = tuple(step_vec)
                    # Compute new vonorms tuple using numpy (faster)
                    new_vonorms_arr = permuted_vonorms_arr + np.array(step_vec)
                    new_vonorms_tuple = tuple(new_vonorms_arr.astype(int))

                    # Store as tuple for fast deduplication
                    step_tuples.append((step_vec_tuple, new_vonorms_tuple, transformed_motif_mnf, mat_tuple_tuple, mat_tuple, transformed_motif))

        # Deduplicate using set on tuples (first 4 elements define uniqueness)
        unique_step_data = {}
        for step_vec_tuple, new_vonorms_tuple, transformed_motif_mnf, mat_tuple_tuple, mat_tuple, transformed_motif in step_tuples:
            key = (step_vec_tuple, new_vonorms_tuple, transformed_motif_mnf, mat_tuple_tuple)
            if key not in unique_step_data:
                unique_step_data[key] = (mat_tuple, transformed_motif)

        # Now create LatticeStep objects only for unique steps
        steps = []
        for (step_vec_tuple, new_vonorms_tuple, transformed_motif_mnf, mat_tuple_tuple), (mat_tuple, transformed_motif) in unique_step_data.items():
            new_vonorms = VonormList(new_vonorms_tuple)
            steps.append(LatticeStep(list(step_vec_tuple), new_vonorms, transformed_motif, mat_tuple))

        return steps


    def get_vonorm_neighbor(self, step: LatticeStep):
        permuted_vonorms = step.vonorms
        self._log(f"Permuted vonorms: {permuted_vonorms}")
        old_vonorms = np.array(permuted_vonorms.vonorms)
        
        new_vonorms = VonormList(tuple([int(v) for v in old_vonorms + np.array(step.vals)]))
        self._log(f"Computed neighbor vonorms before canonicalization: {new_vonorms}")

        if not new_vonorms.has_valid_conorms():
            self._log("Neighbor had invalid conorms")
            return None

        if new_vonorms.is_superbasis() and new_vonorms.is_obtuse():
            return new_vonorms
        else:
            if self.verbose_logging:
                if not new_vonorms.is_obtuse():
                    self._log(f"Neighbor was not obtuse.")

                if not new_vonorms.is_superbasis():
                    self._log(f"Neighbor was not a superbasis.")
            return None
    
    def find_lnf_neighbor(self, step: LatticeStep):
        new_vonorms = step.vonorms
        if new_vonorms is None:
            return None
        
        lnf_constructor = LatticeNormalFormConstructor(self._lnf().lattice_step_size)
        construction_result = lnf_constructor.build_lnf_from_vonorms(new_vonorms, skip_reduction=True)
        neighbor_lnf = construction_result.lnf
            
        return LatticeStepResult(step, new_vonorms, construction_result, neighbor_lnf, step.matrix)
        

    def find_lnf_neighbors(self) -> NeighborSet:
        neighbors = NeighborSet()
        for step in self.possible_steps():
            result = self.find_lnf_neighbor(step)
            if result is not None:
                neighbors.add_neighbor(result)
        return neighbors
    
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

    def find_cnf_neighbors_fast(self) -> list:
        """Fast neighbor finding that avoids intermediate object creation."""
        import os

        # When using Rust, use the complete Rust pipeline
        use_rust = os.getenv("USE_RUST") is not None
        if use_rust:
            return self._find_cnf_neighbors_fast_rust()

        # Python fallback
        return self._find_cnf_neighbors_fast_python()

    def _find_cnf_neighbors_fast_rust(self) -> list:
        """Rust implementation of CNF neighbor finding."""
        import rust_cnf

        # Get validated step data from Rust
        step_data = self._compute_step_data_raw_rust()

        # Build CNFs from step data in Rust
        input_vonorms = np.array(self.point.lattice_normal_form.vonorms.vonorms_np, dtype=np.float64)
        input_coords = np.ascontiguousarray(self.discretized_motif.coord_matrix.flatten(), dtype=np.float64)

        # Convert atoms to list of Python strings (in case they're numpy strings)
        atoms = [str(atom) for atom in self.discretized_motif.atoms]

        # Ensure xi and delta are Python scalars, not numpy types
        xi = float(self.point.xi)
        delta = int(self.point.delta)

        neighbor_tuples = rust_cnf.build_cnfs_from_step_data_rust(
            step_data,
            input_vonorms,
            input_coords,
            atoms,
            xi,
            delta
        )

        # Convert Rust results back to Python CNF objects
        from ..cnf_constructor import CNFConstructor
        cnf_constructor = CNFConstructor(self.point.xi, self.point.delta, verbose_logging=False)

        results = []
        for vonorms_list, coords_list, _ in neighbor_tuples:
            # Data from Rust is not canonical - CNF constructor will canonicalize
            vonorms = VonormList(tuple([int(v) for v in vonorms_list]))

            from ..motif.atomic_motif import DiscretizedMotif

            # coords_list is flattened coords for all atoms
            motif = DiscretizedMotif.from_elements_and_positions(
                atoms,
                np.array(coords_list).reshape(-1, 3),
                delta=self.point.delta
            )

            cnf_result = cnf_constructor.from_vonorms_and_motif(vonorms, motif)

            # Only include if different from input CNF
            if cnf_result.cnf != self.point:
                results.append(cnf_result.cnf)

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
        from ..cnf_constructor import CNFConstructor
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

        for s4_idxs, data in vonorms.maximally_ascending_equivalence_class_members().items():
            permuted_vonorms = data['maximal_permuted_list']
            transform_mats = data['transition_mats']

            # Batch compute all matrix products using einsum
            s1_mats = np.array([s.matrix for s in current_stabilizer])
            t_mats = np.array([t.matrix for t in transform_mats[:1]])
            s2_mats = np.array([s.matrix for s in permuted_vonorms.stabilizer_matrices()])

            all_products = np.einsum('nij,mjk,okl->nmoil', s1_mats, t_mats, s2_mats)
            all_products_flat = all_products.reshape(-1, 3, 3)

            # Deduplicate
            all_products_rounded = np.round(all_products_flat).astype(int)
            all_products_2d = all_products_rounded.reshape(-1, 9)
            unique_flat = np.unique(all_products_2d, axis=0)
            unimodular_mats = unique_flat.reshape(-1, 3, 3)

            step_vecs = LatticeStep.all_step_vecs()
            permuted_vonorms_arr = np.array(permuted_vonorms.vonorms)

            for mat in unimodular_mats:
                # Transform motif coordinates directly
                mat_inv = np.linalg.inv(mat).astype(int)
                transformed_coords = mat_inv @ motif_coord_matrix
                transformed_coords = np.mod(transformed_coords, motif_delta).T  # (N, 3)

                for step_vec in step_vecs:
                    # Compute new vonorms
                    new_vonorms_arr = permuted_vonorms_arr + np.array(step_vec)
                    new_vonorms_tuple = tuple(new_vonorms_arr.astype(int))

                    # Store raw data
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

    def find_cnf_neighbors(self) -> NeighborSet:
        neighbors = NeighborSet()
        steps = self.possible_steps()
        self._log(f"Considering {len(steps)} possible steps...")
        for step in steps:
            self._log("")
            self._log(f"Step: {step.vals}, {step.matrix}")
            results = self.find_cnf_neighbor_results(step)
            for r in results:
                neighbors.add_neighbor(r)
        return neighbors