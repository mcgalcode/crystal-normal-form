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
        return self.build_lnf_from_vonorms(superbasis.compute_vonorms())

    def build_lnf_from_vonorms(self, vonorms: VonormList):
        self._log("Canonicalizing RAW vonorms...")
        canonicalizer = VonormCanonicalizer(reduction_tolerance=1e-8, verbose_logging=self._verbose_logging)
        undiscretized_canonical_result = canonicalizer.get_canonicalized_vonorms(vonorms)

        self._log("Discretizing RAW vonorms...")
        dvc = DiscretizedVonormComputer(self.lattice_step_size, self._verbose_logging)
        discretized_vonorms = dvc.find_closest_valid_vonorms(undiscretized_canonical_result.canonical_vonorms)

        self._log("Canonicalizing DISCRETE vonorms...")
        discretized_canonical_result = canonicalizer.get_canonicalized_vonorms(discretized_vonorms)
        lnf = LatticeNormalForm(discretized_canonical_result.canonical_vonorms, self.lattice_step_size)

        return LatticeNormalFormConstructionResult(
            lnf,
            undiscretized_canonical_result,
            discretized_canonical_result
        )

    def build_lnf_from_discretized_vonorms(self, vonorms: VonormList, skip_reduction = True):
        canonicalizer = VonormCanonicalizer(reduction_tolerance=1e-8, verbose_logging=self._verbose_logging)
        result = canonicalizer.get_canonicalized_vonorms(vonorms, skip_reduction=skip_reduction)
        lnf = LatticeNormalForm(result.canonical_vonorms, self.lattice_step_size)
        
        self._log(f"Canonicalized the neighbor vonorms: {result.canonical_vonorms}")
        self._log(f"Found stabilizing permutations: {[p.vonorm_permutation for p in result.canonical_vonorms.stabilizer_perms()]}")

        return LatticeNormalFormConstructionResult(
            lnf,
            None,
            result
        )

class LatticeNormalFormConstructionResult():

    def __init__(self,
                 lnf: LatticeNormalForm,
                 undiscretized_canonicalization_result: CanonicalizedVonormResult,
                 discretized_canonicalization_result: CanonicalizedVonormResult):
        self.lnf = lnf
        self.undiscretized_canonicalization_result = undiscretized_canonicalization_result
        self.discretized_canonicalization_result = discretized_canonicalization_result
    
    def stabilizer(self, tol=1e-8):
        return self.discretized_canonicalization_result.canonical_vonorms.stabilizer_matrices(tol)
    
    def print_details(self):
        print(f"Found LNF: {self.lnf}")
        if self.undiscretized_canonicalization_result is None:
            print("Skipped undiscretized LNF step")
        else:
            self.undiscretized_canonicalization_result.print_details()
        print("Discretized LNF step: ")
        self.discretized_canonicalization_result.print_details()
    
    @property
    def transform_mat(self):
        transforms = []
        if self.undiscretized_canonicalization_result is not None:
            transforms.extend([
                self.undiscretized_canonicalization_result.selling_transform_mat,
                self.undiscretized_canonicalization_result.equivalent_transformations[0].matrix
            ])

        if self.discretized_canonicalization_result.selling_transform_mat is not None:
            transforms.append(self.discretized_canonicalization_result.selling_transform_mat)

        transforms.append(self.discretized_canonicalization_result.equivalent_transformations[0].matrix)
        return combine_unimodular_matrices(transforms)
