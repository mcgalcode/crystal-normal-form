import numpy as np
from pymatgen.core import Structure # Or other library
from .lattice import Superbasis
from .motif.atomic_motif import FractionalMotif, DiscretizedMotif
from .lattice.lnf_constructor import LatticeNormalFormConstructor, LatticeNormalFormConstructionResult, VonormSorter
from .lattice.voronoi import VonormList
from .lattice.selling import VonormListSellingReducer
from .lattice.lattice_normal_form import LatticeNormalForm
from .motif.basis_normal_form import BasisNormalForm
from .lattice.permutations import MatrixTuple
from .lattice.rounding import DiscretizedVonormComputer
from .motif.bnf_constructor import BNFConstructor, BNFConstructionResult
from .crystal_normal_form import CrystalNormalForm
from .lattice.unimodular import combine_unimodular_matrices

class CNFConstructionResult():

    def __init__(self,
                 cnf: CrystalNormalForm,
                 lnf_construction_result: LatticeNormalFormConstructionResult,
                 bnf_construction_result: BNFConstructionResult):
        self.cnf = cnf
        self.lnf_result = lnf_construction_result
        self.bnf_result = bnf_construction_result
    
    def print_details(self):
        self.lnf_result.print_details()
        print()
        self.bnf_result.print_details()

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
        motif = cnf.basis_normal_form.to_discretized_motif()
        return self.from_discretized_obtuse_vonorms_and_motif(disc_vns, motif)

    def from_motif_and_superbasis(self, motif: FractionalMotif, superbasis: Superbasis):
        lnf_constructor = LatticeNormalFormConstructor(self.xi, self.verbose_logging)
        lnf_construction_result = lnf_constructor.build_lnf_from_superbasis(superbasis)
        if self.verbose_logging:
            print(f"Successfully constructed LNF! {lnf_construction_result.lnf}")
            print()
        
        return self._from_lnf_construction_result(motif, lnf_construction_result)
        
    def from_discretized_obtuse_vonorms_and_motif(self,
                                           discretized_vonorms: VonormList,
                                           motif: DiscretizedMotif):
        if isinstance(motif, FractionalMotif):
            motif = motif.discretize(self.delta)

        if not discretized_vonorms.is_obtuse():
            raise ValueError(f"Provided discretized Vonorms do not represent obtuse superbasis: {discretized_vonorms.vonorms}")
        
        # Here we follow the same procedure as in #from_motif_and_superbasis but
        # we do not need to perform discretization and we do not need to perform
        # selling reduction
        lnf_constructor = LatticeNormalFormConstructor(self.xi, self.verbose_logging)
        lnf_construction_result = lnf_constructor.build_lnf_from_discretized_vonorms(discretized_vonorms, skip_reduction=True)
        if self.verbose_logging:
            print(f"Successfully constructed LNF! {lnf_construction_result.lnf}")
        return self._from_lnf_construction_result(motif, lnf_construction_result)

    def _from_lnf_construction_result(self, motif: FractionalMotif, lnf_construction_result: LatticeNormalFormConstructionResult):
        transform = lnf_construction_result.transform_mat

        if transform is not None:
            if self.verbose_logging:
                print(f"Applying transformation mat:")
                print(transform.matrix)
            motif = motif.apply_unimodular(transform)

        stabilizer_perms = lnf_construction_result.discretized_canonicalization_result.equivalent_transformations
        stabilizer_mats = [m for p in stabilizer_perms for m in p.all_matrices]
        # stabilizer_mats = lnf_construction_result.stabilizer()

        if self.verbose_logging:
            print(f"Found {len(stabilizer_mats)} stabilizers...")

        bnf_constructor = BNFConstructor(self.delta, stabilizer_mats, self.verbose_logging)
        bnf_construction_res = bnf_constructor.build(motif)

        if self.verbose_logging:
            print(f"Found BNF! {bnf_construction_res.bnf}")
            print(f"Achieved by matrix: {bnf_construction_res.sorted_bnf_candidates[0].unimodular}")
            print(f"And shift {bnf_construction_res.sorted_bnf_candidates[0].shift}")
            print(f"Based on motif:")
            bnf_construction_res.sorted_bnf_candidates[0].motif.print_details()

        cnf = CrystalNormalForm(lnf_construction_result.lnf, bnf_construction_res.bnf)
        return CNFConstructionResult(cnf, lnf_construction_result, bnf_construction_res)

    def from_motif_and_superbasis_2(self, motif: FractionalMotif, superbasis: Superbasis):
        reducer     = VonormListSellingReducer(tol=1e-8)
        sorter      = VonormSorter()
        discretizer = DiscretizedVonormComputer(self.xi)

        vonorms = superbasis.compute_vonorms()
        reduction_res = reducer.reduce(vonorms)
        vonorms = reduction_res.reduced_object
        motif = motif.apply_unimodular(reduction_res.transform_matrix)

        vonorms, transforms = sorter.get_canonicalized_vonorms(vonorms)
        motif = motif.apply_unimodular(transforms[0].matrix)

        discretized_vonorms = discretizer.find_closest_valid_vonorms(vonorms)

        reduction_res = reducer.reduce(discretized_vonorms)
        vonorms: VonormList = reduction_res.reduced_object
        motif = motif.apply_unimodular(reduction_res.transform_matrix)
        return self.from_discretized_vonorms_and_motif_2(vonorms, motif)

    def from_discretized_vonorms_and_motif_2(self, vonorms: VonormList, motif: DiscretizedMotif):

        cnf_candidates: list[CrystalNormalForm] = []
        for perm in vonorms.conorms.permissible_permutations:
            lnf_candidate = LatticeNormalForm(vonorms.apply_permutation(perm.vonorm_permutation), self.xi)
            for mat in perm.all_matrices:
                candidate_motif = motif.apply_unimodular(mat)
                if self.verbose_logging:
                    print()
                    print(f"Trying mat: {mat}")
                    candidate_motif.print_details()
                    
                if not isinstance(candidate_motif, DiscretizedMotif):
                    candidate_motif = candidate_motif.discretize(self.delta)

                sorted_elements = candidate_motif.sorted_elements
                origin_element = sorted_elements[0]

                origin_element_positions = candidate_motif.get_element_positions(origin_element)

                # For each possible origin, compute the list
                for origin_candidate in origin_element_positions:
                    shift = -origin_candidate
                    shifted_motif = candidate_motif.shift_origin(shift)
                    bnf_list = shifted_motif.to_bnf_list(element_order=sorted_elements)
                    element_list, _ = shifted_motif.to_elements_and_positions()
                    bnf = BasisNormalForm(tuple([int(c) for c in bnf_list[3:]]), element_list, self.delta)
                    cnf_candidates.append(CrystalNormalForm(lnf_candidate, bnf))

        sorted_cnfs = sorted(cnf_candidates, key=lambda cnf: cnf.coords)
        return CNFConstructionResult(sorted_cnfs[0], None, None)
    
    def from_motif_and_basis_vecs(self, motif: FractionalMotif, basis_vecs: np.array):
        superbasis = Superbasis.from_generating_vecs(basis_vecs)
        return self.from_motif_and_superbasis(motif, superbasis)

    def from_pymatgen_structure(self, struct: Structure):
        motif = FractionalMotif.from_pymatgen_structure(struct)
        superbasis = Superbasis.from_pymatgen_structure(struct)
        return self.from_motif_and_superbasis_2(motif, superbasis)