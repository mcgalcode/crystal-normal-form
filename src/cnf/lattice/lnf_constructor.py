import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from .voronoi.vonorm_list import VonormList
from .voronoi.conorm_list import ConormList
from .selling import VonormListSellingReducer
from .permutations import PermutationMatrix
from .rounding import DiscretizedVonormComputer
from .superbasis import Superbasis
from .lattice_normal_form import LatticeNormalForm
from ..linalg import MatrixTuple
from .voronoi.vonorm_list import VONORM_TO_DOT_PRODUCTS
from .permutations import ZERO_CONORM_SETS_TO_PERMUTATIONS_TO_UNIMOD_MATS, CONORM_PERMUTATION_TO_VONORM_PERMUTATION, CONORM_PERMUTATION_TO_VONORM_PERMUTATION_ARRAY
from ..utils.prof import maybe_profile

@maybe_profile
def build_lnf_raw(vonorms_tuple, xi, discretized=True):
    """
    Fast LNF construction for discretized vonorms.

    Assumes vonorms are already:
    - Discretized (exact integer multiples of xi)
    - In valid form (correct conorms)

    Works directly with tuples/arrays to avoid object creation overhead.
    Uses exact equality instead of tolerance-based comparisons.

    Args:
        vonorms_tuple: tuple or array of 7 vonorm values (already discretized)
        xi: lattice step size
        discretized: if True, returns integer vonorms; if False, returns float vonorms

    Returns:
        tuple: (canonical_vonorms_tuple, selling_transform_matrix, sorting_perm_matrices)
    """

    # Use appropriate dtype based on whether vonorms are discretized
    dtype = int if discretized else float
    vonorms = np.array(vonorms_tuple, dtype=dtype)

    # Step 1: Fast Selling reduction for discretized vonorms
    # Work directly with arrays and use exact arithmetic
    vonorms_reduced = vonorms.copy()
    selling_transform = None

    # Compute conorms to check if already obtuse
    raw_conorms = 0.5 * VONORM_TO_DOT_PRODUCTS @ vonorms_reduced[:6]

    # If all conorms <= 0, already obtuse (no reduction needed)
    if not np.all(raw_conorms <= 0):
        # Need to do Selling reduction - fall back to existing reducer
        # This is rare for discretized vonorms from neighbor finding
        reducer = VonormListSellingReducer(tol=0, verbose_logging=False)
        vonorm_list_obj = VonormList(vonorms)
        reduction_result = reducer.reduce(vonorm_list_obj)
        vonorms_reduced = np.array(reduction_result.reduced_object.vonorms, dtype=dtype)
        selling_transform = reduction_result.transform_matrix

    # Step 2: Compute conorms directly using exact arithmetic
    raw_conorms = (0.5 * VONORM_TO_DOT_PRODUCTS @ vonorms_reduced[:6])

    # Step 3: Find zero conorms using EXACT equality (no tolerance)
    zero_idxs = tuple([idx for idx, cn in enumerate(raw_conorms) if cn == 0])

    # Step 4: Get permissible permutations for this zero set
    if zero_idxs not in ZERO_CONORM_SETS_TO_PERMUTATIONS_TO_UNIMOD_MATS:
        raise ValueError(f"Invalid zero conorm set: {zero_idxs}")

    perm_to_mats = ZERO_CONORM_SETS_TO_PERMUTATIONS_TO_UNIMOD_MATS[zero_idxs]

    # Step 5: Apply all permissible permutations and find lexicographically smallest
    # Work directly with arrays, no object creation
    permuted_vonorms_list = []
    perm_matrices_list = []

    for conorm_perm, mat_list in perm_to_mats.items():
        # Convert conorm permutation to vonorm permutation using pre-computed array
        vonorm_perm_arr = CONORM_PERMUTATION_TO_VONORM_PERMUTATION_ARRAY[conorm_perm]

        # Apply permutation directly on array (faster with numpy array indices)
        permuted = vonorms_reduced[vonorm_perm_arr]
        permuted_tuple = tuple(permuted)  # Create tuple once, reuse it
        permuted_vonorms_list.append(permuted_tuple)
        perm_matrices_list.append((permuted_tuple, mat_list))

    # Step 6: Sort to find canonical (lexicographically smallest)
    permuted_vonorms_list.sort()
    canonical_vonorms = permuted_vonorms_list[0]

    # Find all equivalent transformations (those that give the canonical form)
    equivalent_matrices = [mat_list for perm_tuple, mat_list in perm_matrices_list
                          if perm_tuple == canonical_vonorms]

    # Flatten the list of matrix lists
    all_equivalent_mats = []
    for mat_list in equivalent_matrices:
        all_equivalent_mats.extend(mat_list)

    return canonical_vonorms, selling_transform, all_equivalent_mats


class VonormSorter():

    def __init__(self, verbose_logging=False):
        self._verbose_logging = verbose_logging
        self.sorting_dec_places = 5

    def _log(self, msg):
        if self._verbose_logging:
            print(msg)

    def get_canonicalized_vonorms(self, vonorms: VonormList, coform_tolerance=1e-3):
        conorms = vonorms.conorms.set_tol(coform_tolerance)
        perms = conorms.permissible_permutations
        self._log(f"Searching through {len(perms)} permissible permutations...")
        permuted_vonorm_lists: list[tuple[VonormList, PermutationMatrix]] = []
        
        rounded_vonorms = vonorms.round(self.sorting_dec_places)
        for perm_mat in perms:
            vonorm_permutation = perm_mat.vonorm_permutation
            permuted_vlist = rounded_vonorms.apply_permutation(vonorm_permutation)
            permuted_vonorm_lists.append((permuted_vlist, perm_mat))

        sorted_vlists = sorted(permuted_vonorm_lists, key=lambda group: group[0].tuple, reverse=False)
        canonical_vonorm_list = sorted_vlists[0][0]
        equivalent_transformations = [group[1] for group in sorted_vlists if group[0].about_equal(canonical_vonorm_list)]
        return canonical_vonorm_list, equivalent_transformations

class VonormCanonicalizer():

    def __init__(self, verbose_logging=False, reduction_tolerance = 0):
        self._verbose_logging = verbose_logging
        self.reduction_tolerance = reduction_tolerance
        self.sorting_dec_places = 5

    def _log(self, msg):
        if self._verbose_logging:
            print(msg)

    def get_canonicalized_vonorms(self, vonorms: VonormList, skip_reduction=False, coform_tolerance=1e-3):

        if not skip_reduction:
            self._log(f"Performing Selling Reduction...")
            reducer = VonormListSellingReducer(
                tol=self.reduction_tolerance,
                verbose_logging=self._verbose_logging
            )

            reduction_result = reducer.reduce(vonorms)
            vonorms: VonormList = reduction_result.reduced_object
            reduction_transform = reduction_result.transform_matrix
        else:
            reduction_transform = None
        
        sorter = VonormSorter(self._verbose_logging)
        canonical_vonorm_list, equivalent_transformations = sorter.get_canonicalized_vonorms(vonorms, coform_tolerance)

        return CanonicalizedVonormResult(
            canonical_vonorm_list,
            reduction_transform,
            equivalent_transformations
        )

class CanonicalizedVonormResult():

    def __init__(self,
                 canonical_vonorm_list: VonormList,
                 selling_transform_matrix,
                 equivalent_transformations: list[PermutationMatrix]):
        self.canonical_vonorms = canonical_vonorm_list
        self.selling_transform_mat = selling_transform_matrix
        self.equivalent_transformations = equivalent_transformations
    
    def print_details(self):
        print(f"Selling Transform: {self.selling_transform_mat}")
        print(f"Identified equivalent canonicalizing transformations...")
        for eq in self.equivalent_transformations:
            print(f"Eq. Vo. Perm: {eq.vonorm_permutation}")
            for mat in eq.all_matrices:
                print(f"Mat: {mat.matrix}")
        print(f"Canonicalized vonorms: {self.canonical_vonorms}")

class LatticeNormalFormConstructor():

    def __init__(self, lattice_step_size: float, verbose_logging=False):
        self.lattice_step_size = lattice_step_size    
        self._verbose_logging = verbose_logging

    def _log(self, msg):
        if self._verbose_logging:
            print(msg)


    def build_lnf_from_pymatgen_structure(self, structure: Structure):
        return self.build_lnf_from_superbasis(Superbasis.from_pymatgen_structure(structure))

    def build_lnf_from_pymatgen_lattice(self, lattice: Lattice):
        return self.build_lnf_from_superbasis(Superbasis.from_pymatgen_lattice(lattice))

    def build_lnf_from_generating_vecs(self, generating_vecs: np.array):
        return self.build_lnf_from_superbasis(Superbasis.from_generating_vecs(generating_vecs))

    def build_lnf_from_superbasis(self, superbasis: Superbasis):
        return self.get_from_undiscretized_vnorms(superbasis.compute_vonorms())

    def get_from_undiscretized_vnorms(self, vonorms: VonormList):
        undisc = self.build_lnf_from_vonorms(vonorms)
        dvc = DiscretizedVonormComputer(self.lattice_step_size)
        disc = dvc.find_closest_valid_vonorms(undisc.lnf.vonorms)
        return self.build_lnf_from_vonorms(disc)

    def build_lnf_from_vonorms(self, vonorms: VonormList, skip_reduction = False):
        canonicalizer = VonormCanonicalizer(reduction_tolerance=1e-8, verbose_logging=self._verbose_logging)
        result = canonicalizer.get_canonicalized_vonorms(vonorms, skip_reduction=skip_reduction)
        lnf = LatticeNormalForm(result.canonical_vonorms, self.lattice_step_size)
        
        self._log(f"Found the canonical vonorms: {result.canonical_vonorms}")
        #self._log(f"With conorms: {result.canonical_vonorms.conorms}")
        #self._log(f"Found stabilizing Vonorm permutations: {[p.vonorm_permutation for p in result.canonical_vonorms.stabilizer_perms()]}")
        #self._log(f"Found stabilizing Conorm permutations: {[p.conorm_permutation for p in result.canonical_vonorms.stabilizer_perms()]}")

        return LatticeNormalFormConstructionResult(
            lnf,
            result
        )

    #@maybe_profile
    def build_lnf_from_discretized_vonorms_fast(self, vonorms: VonormList):
        """
        Fast path for building LNF from already-discretized vonorms.

        Uses exact arithmetic and avoids object creation overhead.
        Assumes vonorms are already discretized and valid.

        Args:
            vonorms: VonormList with discretized values

        Returns:
            tuple: (canonical_vonorms, lnf, selling_transform, sorting_matrices)
        """
        # Call the fast raw function
        canonical_vonorms_tuple, selling_transform, sorting_matrices = build_lnf_raw(
            vonorms.vonorms,
            self.lattice_step_size
        )

        # Wrap only what's needed
        canonical_vonorms = VonormList(canonical_vonorms_tuple)
        lnf = LatticeNormalForm(canonical_vonorms, self.lattice_step_size)

        self._log(f"Found the canonical vonorms (fast path): {canonical_vonorms}")

        return canonical_vonorms, lnf, selling_transform, sorting_matrices

class LatticeNormalFormConstructionResult():

    def __init__(self,
                 lnf: LatticeNormalForm,
                 canonical_result: CanonicalizedVonormResult):
        self.lnf = lnf
        self.canonical_result = canonical_result
    
    def stabilizer(self, tol=1e-8):
        return self.lnf.vonorms.stabilizer_matrices(tol)
    
    def sorting_transforms(self):
        tmat_perms = self.canonical_result.equivalent_transformations
        mats = [m for p in tmat_perms for m in p.all_matrices]
        return mats

    def print_details(self):
        print(f"Found LNF: {self.lnf}")
        print("LNF step: ")
        self.canonical_result.print_details()
    
    def selling_transform_mat(self):
        mat = self.canonical_result.selling_transform_mat
        if mat is None:
            return MatrixTuple.identity()
        return mat
