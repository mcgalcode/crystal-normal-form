import numpy as np
import os
from pymatgen.core import Structure # Or other library
from .lattice import Superbasis
from .motif.atomic_motif import FractionalMotif, DiscretizedMotif
from .lattice.lnf_constructor import LatticeNormalFormConstructor, LatticeNormalFormConstructionResult
from .lattice.voronoi import VonormList
from .lattice.lattice_normal_form import LatticeNormalForm
from .motif.motif_normal_form import MotifNormalForm
from .lattice.permutations import MatrixTuple
from .lattice.rounding import DiscretizedVonormComputer
from .motif.mnf_constructor import MNFConstructor, MNFConstructionResult
from .crystal_normal_form import CrystalNormalForm
from .utils.prof import maybe_profile


def should_use_rust():
    return os.getenv("USE_RUST") is not None


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
        self.USE_RUST = should_use_rust()

        self.xi = xi
        self.delta = delta
        self.verbose_logging = verbose_logging


    def _build_lnf_and_get_transforms(self, vonorms: VonormList, use_float: bool = False, use_rust: bool = False, float_tol: float = None):
        """Build LNF and extract transformation matrices."""

        if use_rust:
            # Use Rust for LNF construction
            import rust_cnf
            vonorms_arr = np.ascontiguousarray(vonorms.vonorms_np, dtype=np.float64)

            if use_float:
                canonical_arr, _, selling_flat, sorting_mats_flat = rust_cnf.build_lnf_raw_float_rust(vonorms_arr, float_tol)
            else:
                canonical_arr, _, selling_flat, sorting_mats_flat = rust_cnf.build_lnf_raw_rust(vonorms_arr)

            canonical_vonorms = VonormList(canonical_arr)
            lnf = LatticeNormalForm(canonical_vonorms, self.xi)
            lnf_result = None

            # Extract transformation matrices
            if selling_flat is not None:
                selling_mat = np.array(selling_flat, dtype=np.int32).reshape(3, 3)
            else:
                selling_mat = np.eye(3, dtype=np.int32)
            sorting_mat = np.array(sorting_mats_flat[0], dtype=np.int32).reshape(3, 3)
            middle = (selling_mat @ sorting_mat).astype(np.int32)
        else:
            # Use Python LNF construction
            lnf_constructor = LatticeNormalFormConstructor(self.xi, self.verbose_logging)

            if use_float:
                lnf_result = lnf_constructor.build_lnf_from_vonorms(vonorms)
                lnf = lnf_result.lnf
                canonical_vonorms = lnf_result.lnf.vonorms
                selling_transform = lnf_result.selling_transform_mat()
                sorting_transforms = lnf_result.sorting_transforms()
            else:
                canonical_vonorms, lnf, selling_transform, sorting_matrices = lnf_constructor.build_lnf_from_discretized_vonorms_fast(vonorms)
                sorting_transforms = sorting_matrices
                lnf_result = None

            # Compute middle transformation matrix
            selling_mat = selling_transform.matrix if selling_transform else np.eye(3)
            sorting_mat = sorting_transforms[0].matrix
            middle = selling_mat @ sorting_mat

        return lnf, lnf_result, canonical_vonorms, middle

    def _combine_stabilizers(self, vonorms: VonormList, canonical_vonorms: VonormList, middle: np.ndarray, use_float: bool = False, use_rust: bool = False):
        """Find and combine stabilizers for lattice transformations."""

        if use_rust:
            # Use Rust's combined function
            import rust_cnf
            vonorms_arr = np.ascontiguousarray(vonorms.vonorms_np, dtype=np.float64)
            canonical_arr = np.ascontiguousarray(canonical_vonorms.vonorms_np, dtype=np.float64)
            middle_flat = middle.astype(np.int32).flatten()

            if use_float:
                combined_stabs_flat = rust_cnf.find_and_combine_stabilizers_rust_float(
                    vonorms_arr, canonical_arr, middle_flat, 1e-8  # TODO: make tolerance configurable
                )
            else:
                combined_stabs_flat = rust_cnf.find_and_combine_stabilizers_rust(
                    vonorms_arr, canonical_arr, middle_flat
                )

            # Reshape to (N, 3, 3)
            n_matrices = len(combined_stabs_flat) // 9
            combined_stabs_3d = combined_stabs_flat.reshape(n_matrices, 3, 3)
            np_stabs = list(combined_stabs_3d)
        else:
            # Python implementation with einsum
            if use_float:
                stabilizer_1 = vonorms.stabilizer_matrices()
                stabilizer_2 = canonical_vonorms.stabilizer_matrices()
            else:
                stabilizer_1 = vonorms.stabilizer_matrices_fast()
                stabilizer_2 = canonical_vonorms.stabilizer_matrices_fast()

            # Combine all stabilizers
            s1_stack = np.array([s.matrix for s in stabilizer_1])
            s2_stack = np.array([s.matrix for s in stabilizer_2])
            result = np.einsum('nij,jk,mkl->nmil', s1_stack, middle, s2_stack)
            result_flat = result.reshape(-1, 3, 3)
            all_stabilizers = [MatrixTuple(mat) for mat in result_flat]
            np_stabs = [s.matrix for s in all_stabilizers]

        if self.verbose_logging:
            print(f"Found {len(np_stabs)} combined stabilizers...")

        return np_stabs

    def from_cnf(self, cnf: CrystalNormalForm):
        assert cnf.xi == self.xi
        assert cnf.delta == self.delta
        disc_vns = cnf.lattice_normal_form.vonorms
        motif = cnf.motif_normal_form.to_discretized_motif()
        return self.from_vonorms_and_motif(disc_vns, motif)

    def from_motif_and_superbasis(self, motif: FractionalMotif, superbasis: Superbasis):
        vonorms = superbasis.compute_vonorms()
        return self.from_vonorms_and_motif_undiscretized(vonorms, motif)
    
    def from_vonorms_and_motif_undiscretized(self, vonorms: VonormList, motif: FractionalMotif):
        return self._from_vonorms_and_motif_undiscretized_impl(vonorms, motif, use_rust=self.USE_RUST)
    
    def from_vonorms_and_motif(self, vonorms: VonormList, motif: FractionalMotif):
        return self._from_vonorms_and_motif_impl(vonorms, motif, use_rust=self.USE_RUST)
    
    def _from_vonorms_and_motif_undiscretized_impl(self, vonorms: VonormList, motif: FractionalMotif, use_rust: bool):
        # First pass: float version with tolerance to find approximate canonical form
        undisc_cnf = self._from_vonorms_and_motif_impl(vonorms, motif, float_tol=1e-4, use_rust=use_rust)
        vonorms = undisc_cnf.cnf.lattice_normal_form.vonorms

        motif = undisc_cnf.mnf_result.canonical_motif
        motif = motif.discretize(self.delta)

        # Second pass: discretized version with exact arithmetic
        dvc = DiscretizedVonormComputer(self.xi, self.verbose_logging)
        vonorms = dvc.find_closest_valid_vonorms(vonorms)

        return self._from_vonorms_and_motif_impl(vonorms, motif, use_rust=use_rust)

    @maybe_profile
    def _from_vonorms_and_motif_impl(self, vonorms: VonormList, motif: DiscretizedMotif | FractionalMotif, float_tol=None, use_rust=False):
        """Unified CNF construction implementation for both Python and Rust paths."""
        use_float = float_tol is not None
        if use_float and not use_rust:
            vonorms = vonorms.set_tol(float_tol)

        # Build LNF and get transformation matrices
        lnf, lnf_result, canonical_vonorms, middle = self._build_lnf_and_get_transforms(
            vonorms, use_float=use_float, use_rust=use_rust, float_tol=float_tol
        )

        # Find and combine stabilizers
        np_stabs = self._combine_stabilizers(vonorms, canonical_vonorms, middle, use_float, use_rust=use_rust)

        # Build MNF
        mnf_constructor = MNFConstructor(self.delta, np_stabs, self.verbose_logging)
        mnf_construction_res = mnf_constructor.build_vectorized(motif, use_rust=use_rust)

        if self.verbose_logging:
            mode = "Rust" if use_rust else "Python"
            print(f"Found MNF ({mode})! {mnf_construction_res.mnf}")

        cnf = CrystalNormalForm(lnf, mnf_construction_res.mnf)
        return CNFConstructionResult(cnf, lnf_result, mnf_construction_res)

    def from_motif_and_basis_vecs(self, motif: FractionalMotif, basis_vecs: np.array):
        superbasis = Superbasis.from_generating_vecs(basis_vecs)
        return self.from_motif_and_superbasis(motif, superbasis)

    def from_pymatgen_structure(self, struct: Structure):
        motif = FractionalMotif.from_pymatgen_structure(struct)
        superbasis = Superbasis.from_pymatgen_structure(struct)
        return self.from_motif_and_superbasis(motif, superbasis)