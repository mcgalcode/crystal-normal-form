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
        return self.from_vonorms_and_motif_undiscretized(vonorms, motif)
    
    def from_vonorms_and_motif_undiscretized(self, vonorms: VonormList, motif: FractionalMotif):
        vonorms = vonorms.set_tol(1e-3)
        undisc_cnf = self.from_vonorms_and_motif(vonorms, motif)
        vonorms = undisc_cnf.lnf_result.lnf.vonorms

        motif = undisc_cnf.mnf_result.canonical_motif
        motif = motif.discretize(self.delta)

        dvc = DiscretizedVonormComputer(self.xi, self.verbose_logging)
        vonorms = dvc.find_closest_valid_vonorms(vonorms)

        return self.from_vonorms_and_motif(vonorms, motif)

    def from_vonorms_and_motif(self, vonorms: VonormList, motif: DiscretizedMotif | FractionalMotif):
        lnf_constructor = LatticeNormalFormConstructor(self.xi, self.verbose_logging)
        lnf_result = lnf_constructor.build_lnf_from_vonorms(vonorms)
        if self.verbose_logging:
            print(f"Successfully constructed LNF! {lnf_result.lnf}")

        stabilizer_1 = vonorms.stabilizer_matrices(1e-4)
        selling = [lnf_result.selling_transform_mat()]
        sorting_transforms = lnf_result.sorting_transforms()[:1]
        stabilizer_2 = lnf_result.stabilizer(1e-4)
        
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
    
    def from_motif_and_basis_vecs(self, motif: FractionalMotif, basis_vecs: np.array):
        superbasis = Superbasis.from_generating_vecs(basis_vecs)
        return self.from_motif_and_superbasis(motif, superbasis)

    def from_pymatgen_structure(self, struct: Structure):
        motif = FractionalMotif.from_pymatgen_structure(struct)
        superbasis = Superbasis.from_pymatgen_structure(struct)
        return self.from_motif_and_superbasis(motif, superbasis)