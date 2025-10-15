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
from .unimodular import combine_unimodular_matrices
from ..linalg import MatrixTuple


class VonormSorter():

    def __init__(self, verbose_logging=False):
        self._verbose_logging = verbose_logging
        self.sorting_dec_places = 5

    def _log(self, msg):
        if self._verbose_logging:
            print(msg)

    def get_canonicalized_vonorms(self, vonorms: VonormList, coform_tolerance=1e-3):
        conorms = vonorms.conorms
        conorms = conorms.set_tol(coform_tolerance)
        self._log(f"Searching through {len(conorms.form.permissible_permutations())} permissible permutations...")
        permuted_vonorm_lists: list[tuple[VonormList, PermutationMatrix]] = []
        for perm_mat in conorms.form.permissible_permutations():
            vonorm_permutation = perm_mat.vonorm_permutation
            permuted_vlist = vonorms.apply_permutation(vonorm_permutation)
            permuted_vonorm_lists.append((permuted_vlist, perm_mat))

        sorted_vlists = sorted(permuted_vonorm_lists, key=lambda group: tuple([round(v, self.sorting_dec_places) for v in group[0].vonorms]), reverse=False)
        canonical_vonorm_list = sorted_vlists[0][0]
        equivalent_transformations = [group[1] for group in sorted_vlists if group[0] == canonical_vonorm_list]
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
        self._log(f"Found stabilizing permutations: {[p.vonorm_permutation for p in result.canonical_vonorms.stabilizer_perms()]}")

        return LatticeNormalFormConstructionResult(
            lnf,
            result
        )

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
