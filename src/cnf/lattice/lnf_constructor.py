import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from .vonorm_list import VonormList
from .selling import VonormListSellingReducer
from .permutations import VonormPermutation
from .rounding import DiscretizedVonormComputer
from .superbasis import Superbasis
from .lattice_normal_form import LatticeNormalForm


class VonormCanonicalizer():

    def __init__(self, verbose_logging=False, reduction_tolerance = 0):
        self._verbose_logging = verbose_logging
        self.reduction_tolerance = reduction_tolerance
    
    def get_canonicalized_vonorms(self, vonorms: VonormList):
        reducer = VonormListSellingReducer(tol=self.reduction_tolerance, verbose_logging=self._verbose_logging)
        reduction_result = reducer.reduce(vonorms)
        reduced_vonorms = reduction_result.reduced_object
        conorms = reduced_vonorms.conorms

        permuted_vonorm_lists: list[tuple[VonormList, VonormPermutation]] = []
        for conorm_permutation in conorms.permissible_permutations:
            vonorm_permutation = conorm_permutation.to_vonorm_permutation()
            permuted_vlist = reduced_vonorms.apply_permutation(vonorm_permutation)
            permuted_vonorm_lists.append((permuted_vlist, vonorm_permutation))

        sorted_vlists = sorted(permuted_vonorm_lists, key=lambda pair: pair[0].vonorms, reverse=False)
        canonical_vonorm_list = sorted_vlists[0][0]
        stabilizer_permutations = [pair[1] for pair in permuted_vonorm_lists if pair[0] == canonical_vonorm_list]
        return CanonicalizedVonormResult(
            canonical_vonorm_list,
            reduction_result.transform_matrix,
            stabilizer_permutations
        )

class CanonicalizedVonormResult():

    def __init__(self,
                 canonical_vonorm_list: VonormList,
                 selling_transform_matrix,
                 stabilizer_permutations: list[VonormPermutation]):
        self.canonical_vonorms = canonical_vonorm_list
        self.selling_transform_mat = selling_transform_matrix
        self.stabilizer_permutations = stabilizer_permutations

class LatticeNormalFormConstructor():

    def __init__(self, lattice_step_size: float, verbose_logging=False):
        self.lattice_step_size = lattice_step_size    
        self._verbose_logging = verbose_logging

    def build_lnf_from_pymatgen_structure(self, structure: Structure):
        return self.build_lnf_from_superbasis(Superbasis.from_pymatgen_structure(structure))

    def build_lnf_from_pymatgen_lattice(self, lattice: Lattice):
        return self.build_lnf_from_superbasis(Superbasis.from_pymatgen_lattice(lattice))

    def build_lnf_from_generating_vecs(self, generating_vecs: np.array):
        return self.build_lnf_from_superbasis(Superbasis.from_generating_vecs(generating_vecs))

    def build_lnf_from_superbasis(self, superbasis: Superbasis):
        return self.build_lnf_from_vonorms(superbasis.compute_vonorms())

    def build_lnf_from_vonorms(self, vonorms: VonormList):
        canonicalizer = VonormCanonicalizer(reduction_tolerance=1e-8, verbose_logging=self._verbose_logging)
        undiscretized_canonical_result = canonicalizer.get_canonicalized_vonorms(vonorms)

        dvc = DiscretizedVonormComputer(self.lattice_step_size, self._verbose_logging)
        discretized_vonorms = dvc.find_closest_valid_vonorms(undiscretized_canonical_result.canonical_vonorms)

        discretized_canonical_result = canonicalizer.get_canonicalized_vonorms(discretized_vonorms)
        lnf = LatticeNormalForm(discretized_canonical_result.canonical_vonorms, self.lattice_step_size)
        return LatticeNormalFormConstructionResult(
            lnf,
            undiscretized_canonical_result,
            discretized_canonical_result
        )

class LatticeNormalFormConstructionResult():

    def __init__(self,
                 lnf: LatticeNormalForm,
                 undiscretized_canonicalization_result: CanonicalizedVonormResult,
                 discretized_canonicalization_result: CanonicalizedVonormResult):
        self.lnf = lnf
        self.undiscretized_canonicalization_result = undiscretized_canonicalization_result
        self.discretized_canonicalization_result = discretized_canonicalization_result
