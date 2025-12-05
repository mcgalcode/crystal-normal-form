import numpy as np
from itertools import product
from pymatgen.core import Structure # Or other library
from .lattice import Superbasis
from .motif.atomic_motif import FractionalMotif, DiscretizedMotif
from .lattice.lnf_constructor import LatticeNormalFormConstructor, LatticeNormalFormConstructionResult, VonormSorter
from .lattice.voronoi import VonormList
from .lattice.selling import VonormListSellingReducer
from .lattice.lattice_normal_form import LatticeNormalForm
from .motif.motif_normal_form import MotifNormalForm
from .lattice.permutations import MatrixTuple
from .lattice.rounding import DiscretizedVonormComputer
from .motif.mnf_constructor import MNFConstructor, MNFConstructionResult
from .crystal_normal_form import CrystalNormalForm
from .linalg.unimodular import combine_unimodular_matrices, combine_unimodular_mats_np
from .utils.prof import maybe_profile

class CNFConstructionResult():

    def __init__(self,
                 cnf: CrystalNormalForm,
                 lnf_result: LatticeNormalFormConstructionResult,
                 mnf_construction_result: MNFConstructionResult):
        self.cnf = cnf
        self.lnf_result = lnf_result
        self.mnf_result = mnf_construction_result
    
    def print_details(self):
        self.lnf_result.print_details()
        print()
        self.mnf_result.print_details()

class CNFConstructor():

    def __init__(self,
                 xi: float,
                 delta: int,
                 verbose_logging: bool = False):
        self.xi = xi
        self.delta = delta
        self.verbose_logging = verbose_logging


    def from_cnf(self, cnf: CrystalNormalForm):
        assert cnf.xi == self.xi
        assert cnf.delta == self.delta
        disc_vns = cnf.lattice_normal_form.vonorms
        motif = cnf.motif_normal_form.to_discretized_motif()
        return self.from_vonorms_and_motif(disc_vns, motif)

    def from_motif_and_superbasis(self, motif: FractionalMotif, superbasis: Superbasis):
        vonorms = superbasis.compute_vonorms()
        return self.from_vonorms_and_motif_undiscretized_rust(vonorms, motif)
    
    def from_vonorms_and_motif_undiscretized(self, vonorms: VonormList, motif: FractionalMotif):
        vonorms = vonorms.set_tol(1e-3)
        undisc_cnf = self.from_vonorms_and_motif(vonorms, motif)
        vonorms = undisc_cnf.cnf.lattice_normal_form.vonorms

        motif = undisc_cnf.mnf_result.canonical_motif
        motif = motif.discretize(self.delta)

        dvc = DiscretizedVonormComputer(self.xi, self.verbose_logging)
        vonorms = dvc.find_closest_valid_vonorms(vonorms)

        return self.from_vonorms_and_motif(vonorms, motif)
    
    def from_vonorms_and_motif_undiscretized_rust(self, vonorms: VonormList, motif: FractionalMotif):
        # First pass: use Rust float version (tolerance-based)
        undisc_cnf = self._from_vonorms_and_motif_rust(vonorms, motif, float_tol=1e-4)
        vonorms = undisc_cnf.cnf.lattice_normal_form.vonorms

        motif = undisc_cnf.mnf_result.canonical_motif
        motif = motif.discretize(self.delta)

        # Second pass: use Rust discretized version (exact)
        dvc = DiscretizedVonormComputer(self.xi, self.verbose_logging)
        vonorms = dvc.find_closest_valid_vonorms(vonorms)

        return self._from_vonorms_and_motif_rust(vonorms, motif)
    
    @maybe_profile
    def from_vonorms_and_motif(self, vonorms: VonormList, motif: DiscretizedMotif | FractionalMotif):
        # return self._from_vonorms_and_motif_rust(vonorms, motif)
        lnf_constructor = LatticeNormalFormConstructor(self.xi, self.verbose_logging)
        lnf_result = lnf_constructor.build_lnf_from_vonorms(vonorms)
        if self.verbose_logging:
            print(f"Successfully constructed LNF! {lnf_result.lnf}")

        stabilizer_1 = vonorms.stabilizer_matrices()
        selling = [lnf_result.selling_transform_mat()]
        sorting_transforms = lnf_result.sorting_transforms()[:1]
        stabilizer_2 = lnf_result.stabilizer()

        if self.verbose_logging:
            print(f"Stabilizer_1 count: {len(stabilizer_1)}")
            print(f"Selling matrix:\n{selling[0].matrix}")
            print(f"Sorting matrix:\n{sorting_transforms[0].matrix}")
            print(f"Stabilizer_2 count: {len(stabilizer_2)}")

        all_stabilizers = [MatrixTuple(combine_unimodular_mats_np([s.matrix for s in stack])) for stack in product(stabilizer_1, selling, sorting_transforms, stabilizer_2)]
        all_stabilizers = list(set(all_stabilizers))
        np_stabs = [s.matrix for s in all_stabilizers]
        # Option 1 - use the transforms TO the sorted list as the search set
        # stabilizer_mats = lnf_result.sorting_transforms()

        # Option 2 - transform via ANY ONE of the transforms to sorted list then use stabilizer as search set
        # motif = motif.apply_unimodular(lnf_result.sorting_transforms()[0])
        # stabilizer_mats = lnf_result.stabilizer(tol=1e-3)

        if self.verbose_logging:
            print(f"Found {len(all_stabilizers)} stabilizers...")
            if len(all_stabilizers) <= 10:
                print("Stabilizer matrices (Python):")
                for i, mat in enumerate(np_stabs):
                    print(f"  Stabilizer {i}:\n{mat}")

        mnf_constructor = MNFConstructor(self.delta, np_stabs, self.verbose_logging)
        mnf_construction_res = mnf_constructor.build_vectorized(motif)

        if self.verbose_logging:
            print(f"Found MNF! {mnf_construction_res.mnf}")
            print(f"Achieved by matrix: {mnf_construction_res.sorted_mnf_candidates[0].unimodular}")
            print(f"And shift {mnf_construction_res.sorted_mnf_candidates[0].shift}")
            print(f"Based on motif:")
            mnf_construction_res.canonical_motif.print_details()

        cnf = CrystalNormalForm(lnf_result.lnf, mnf_construction_res.mnf)
        return CNFConstructionResult(cnf, lnf_result, mnf_construction_res)

    @maybe_profile
    def from_vonorms_and_motif_fast(self, vonorms: VonormList, motif: DiscretizedMotif | FractionalMotif):
        """
        Fast path for CNF construction when vonorms are already discretized.

        Uses optimized LNF construction that avoids object creation overhead
        and uses exact arithmetic instead of tolerance-based comparisons.

        Args:
            vonorms: VonormList with already-discretized values
            motif: DiscretizedMotif or FractionalMotif

        Returns:
            CNFConstructionResult
        """
        lnf_constructor = LatticeNormalFormConstructor(self.xi, self.verbose_logging)
        canonical_vonorms, lnf, selling_transform, sorting_matrices = lnf_constructor.build_lnf_from_discretized_vonorms_fast(vonorms)

        if self.verbose_logging:
            print(f"Successfully constructed LNF (fast path)! {lnf}")

        # Compute stabilizers using fast path for discretized vonorms
        stabilizer_1 = vonorms.stabilizer_matrices_fast()
        selling_mat = selling_transform.matrix if selling_transform else np.eye(3)
        sorting_mat = sorting_matrices[0].matrix if sorting_matrices else np.eye(3)
        stabilizer_2 = canonical_vonorms.stabilizer_matrices_fast()

        # Vectorized matrix multiplication using einsum
        # Pre-compute middle product
        middle = selling_mat @ sorting_mat

        # Stack all s1 and s2 matrices into 3D arrays
        s1_stack = np.array([s.matrix for s in stabilizer_1])  # shape (N1, 3, 3)
        s2_stack = np.array([s.matrix for s in stabilizer_2])  # shape (N2, 3, 3)

        # Compute all combinations: s1[i] @ middle @ s2[j] for all i, j
        # Result shape: (N1, N2, 3, 3)
        result = np.einsum('nij,jk,mkl->nmil', s1_stack, middle, s2_stack)

        # Flatten and convert to MatrixTuple
        result_flat = result.reshape(-1, 3, 3)
        all_stabilizers = [MatrixTuple(mat) for mat in result_flat]

        # Deduplicate
        all_stabilizers = list(set(all_stabilizers))
        np_stabs = [s.matrix for s in all_stabilizers]

        if self.verbose_logging:
            print(f"Found {len(all_stabilizers)} stabilizers...")

        mnf_constructor = MNFConstructor(self.delta, np_stabs, self.verbose_logging)
        mnf_construction_res = mnf_constructor.build_vectorized(motif)

        if self.verbose_logging:
            print(f"Found MNF! {mnf_construction_res.mnf}")

        cnf = CrystalNormalForm(lnf, mnf_construction_res.mnf)

        # Create minimal result objects for compatibility
        # (these are only used if caller needs them, but CNF is what matters)
        lnf_result = None  # We don't need the full result object
        return CNFConstructionResult(cnf, lnf_result, mnf_construction_res)

    def _from_vonorms_and_motif_rust(self, vonorms: VonormList, motif: DiscretizedMotif | FractionalMotif, float_tol=None):
        """
        Rust-accelerated CNF construction.

        Uses Rust for LNF construction, stabilizer finding, and combination.
        """
        import rust_cnf
        use_float = float_tol is not None
        vonorms_arr = np.array(vonorms.vonorms, dtype=np.float64)

        # Use Rust for LNF construction (now returns transformation matrices)
        if use_float:
            canonical_arr, _, selling_flat, sorting_mats_flat = rust_cnf.build_lnf_raw_float_rust(vonorms_arr, float_tol)
        else:
            canonical_arr, _, selling_flat, sorting_mats_flat = rust_cnf.build_lnf_raw_rust(vonorms_arr)

        canonical_vonorms = VonormList(canonical_arr)
        lnf = LatticeNormalForm(canonical_vonorms, self.xi)

        if self.verbose_logging:
            print(f"Successfully constructed LNF (Rust)! {lnf}")

        # Compute middle matrix from selling and sorting transforms
        if selling_flat is not None:
            selling_mat = np.array(selling_flat, dtype=np.int32).reshape(3, 3)
        else:
            selling_mat = np.eye(3, dtype=np.int32)

        # Use first sorting matrix (always available)
        sorting_mat = np.array(sorting_mats_flat[0], dtype=np.int32).reshape(3, 3)
        middle = (selling_mat @ sorting_mat).astype(np.int32)
        middle_flat = middle.flatten()

        # Use combined Rust function to find and combine stabilizers in one go
        # Returns a flat numpy array that we reshape to (N, 3, 3)
        if use_float:
            combined_stabs_flat = rust_cnf.find_and_combine_stabilizers_rust_float(
                vonorms_arr, canonical_arr, middle_flat, float_tol
            )
        else:
            combined_stabs_flat = rust_cnf.find_and_combine_stabilizers_rust(
                vonorms_arr, canonical_arr, middle_flat
            )

        # Reshape to (N, 3, 3) and convert to list of matrices
        n_matrices = len(combined_stabs_flat) // 9
        combined_stabs_3d = combined_stabs_flat.reshape(n_matrices, 3, 3)
        np_stabs = list(combined_stabs_3d)

        if self.verbose_logging:
            print(f"Found {len(np_stabs)} stabilizers (Rust)...")

        mnf_constructor = MNFConstructor(self.delta, np_stabs, self.verbose_logging)
        mnf_construction_res = mnf_constructor.build_vectorized(motif, use_rust=True)

        if self.verbose_logging:
            print(f"Found MNF (Rust)! {mnf_construction_res.mnf}")

        cnf = CrystalNormalForm(lnf, mnf_construction_res.mnf)

        return CNFConstructionResult(cnf, None, mnf_construction_res)

    def from_motif_and_basis_vecs(self, motif: FractionalMotif, basis_vecs: np.array):
        superbasis = Superbasis.from_generating_vecs(basis_vecs)
        return self.from_motif_and_superbasis(motif, superbasis)

    def from_pymatgen_structure(self, struct: Structure):
        motif = FractionalMotif.from_pymatgen_structure(struct)
        superbasis = Superbasis.from_pymatgen_structure(struct)
        return self.from_motif_and_superbasis(motif, superbasis)