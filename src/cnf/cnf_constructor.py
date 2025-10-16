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
                 lnf_result: LatticeNormalFormConstructionResult,
                 bnf_construction_result: BNFConstructionResult):
        self.cnf = cnf
        self.lnf_result = lnf_result
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
        return self.from_vonorms_and_motif(disc_vns, motif)

    def from_motif_and_superbasis(self, motif: FractionalMotif, superbasis: Superbasis):
        vonorms = superbasis.compute_vonorms()
        return self.from_vonorms_and_motif_undiscretized(vonorms, motif)
    
    def from_vonorms_and_motif_undiscretized(self, vonorms: VonormList, motif: FractionalMotif):
        undisc_cnf = self.from_vonorms_and_motif(vonorms, motif)
        vonorms = undisc_cnf.lnf_result.lnf.vonorms

        motif = undisc_cnf.bnf_result.sorted_bnf_candidates[0].motif
        motif = motif.discretize(self.delta)

        dvc = DiscretizedVonormComputer(self.xi, self.verbose_logging)
        vonorms = dvc.find_closest_valid_vonorms(vonorms)

        return self.from_vonorms_and_motif(vonorms, motif)
        
    def from_vonorms_and_motif(self, vonorms: VonormList, motif: DiscretizedMotif | FractionalMotif):
        lnf_constructor = LatticeNormalFormConstructor(self.xi, self.verbose_logging)
        lnf_result = lnf_constructor.build_lnf_from_vonorms(vonorms)
        if self.verbose_logging:
            print(f"Successfully constructed LNF! {lnf_result.lnf}")
        motif = motif.apply_unimodular(lnf_result.selling_transform_mat())
        # Option 1 - use the transforms TO the sorted list as the search set
        stabilizer_mats = lnf_result.sorting_transforms()
        
        # Option 2 - transform via ANY ONE of the transforms to sorted list then use stabilizer as search set
        # motif = motif.apply_unimodular(lnf_result.sorting_transforms()[0])
        # stabilizer_mats = lnf_result.stabilizer()

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

        cnf = CrystalNormalForm(lnf_result.lnf, bnf_construction_res.bnf)
        return CNFConstructionResult(cnf, lnf_result, bnf_construction_res)
    
    def from_motif_and_basis_vecs(self, motif: FractionalMotif, basis_vecs: np.array):
        superbasis = Superbasis.from_generating_vecs(basis_vecs)
        return self.from_motif_and_superbasis(motif, superbasis)

    def from_pymatgen_structure(self, struct: Structure):
        motif = FractionalMotif.from_pymatgen_structure(struct)
        superbasis = Superbasis.from_pymatgen_structure(struct)
        return self.from_motif_and_superbasis(motif, superbasis)