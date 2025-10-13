import numpy as np
import itertools

from .atomic_motif import FractionalMotif, DiscretizedMotif
from ..lattice.permutations import PermutationMatrix
from ..linalg import MatrixTuple
from .basis_normal_form import BasisNormalForm
from ..lattice.unimodular import combine_unimodular_matrices
    
class BNFCandidate():

    def __init__(self,
                 bnf_coords: tuple,
                 motif: DiscretizedMotif,
                 unimodular: MatrixTuple,
                 shift: np.array):
        self.bnf_coords = bnf_coords
        self.motif = motif
        self.unimodular = unimodular
        self.shift = shift
    
    def __repr__(self):
        repr = ""
        repr += f"Stabilizer Matrix: {self.unimodular.tuple}"
        repr += "\n"
        repr += f"Shift: {self.shift}"
        return repr

class BNFConstructionResult():

    def __init__(self,
                 original_motif: DiscretizedMotif,
                 delta: int,
                 stabilizers: list[MatrixTuple],
                 sorted_bnf_candidates: list[BNFCandidate]):
        self.delta = delta
        self.original_motif = original_motif
        self.stabilizers = stabilizers
        self.sorted_bnf_candidates = sorted_bnf_candidates
    
    def print_details(self):
        print(f"Found BNF: {self.bnf.coord_list}")
        print(f"Applied the following stabilizers...")
        for p in self.stabilizers:
            print(p.tuple)

        print(f"Considered stablizers:")
        for p in self.stabilizers:
            print(f"Vo Perm: {p.vonorm_permutation})")
            for m in p.all_matrices:
                print(f"Mat: {m.tuple}")
        
        print(f"Found phone-book first shift: {self.sorted_bnf_candidates[0].shift}")
    
    @property
    def bnf(self):
        canonical_candidate = self.sorted_bnf_candidates[0]
        element_list, _ = canonical_candidate.motif.to_elements_and_positions()
        canonical_bnf_coords= canonical_candidate.bnf_coords

        return BasisNormalForm(tuple([int(c) for c in canonical_bnf_coords]), element_list, self.delta)

class BNFConstructor():
    """Implements methods for taking a list of atomic positions
    in fractional coordinates and producing the Basis Normal Form string
    as described in the section "Representation of Crystalline Atomic Bases" on
    pp. 52 of David Mrdjenovich's thesis.
    """
    
    def __init__(self,
                 delta: int,
                 stabilizer: list[MatrixTuple] = None,
                 verbose_logging = False):
        self.delta = delta
        if stabilizer is None:
            stabilizer = [MatrixTuple(np.eye(3))]
        self.stabilizer = stabilizer
        self.verbose_logging = verbose_logging
    
    def build(self, original_motif: FractionalMotif):

        if self.verbose_logging:
            print(f"Initial motif positions:")
            original_motif.print_details()

        bnf_candidates: list[BNFCandidate] = []

        for mat in self.stabilizer:
            transformed_motif = original_motif.apply_unimodular(mat)
            if self.verbose_logging:
                print()
                print(f"Trying mat: {mat}")
                transformed_motif.print_details()
                
            if not isinstance(transformed_motif, DiscretizedMotif):
                transformed_motif = transformed_motif.discretize(self.delta)

            sorted_elements = transformed_motif.sorted_elements
            origin_element = sorted_elements[0]

            origin_element_positions = transformed_motif.get_element_positions(origin_element)

            # For each possible origin, compute the list
            for origin_candidate in origin_element_positions:
                shift = -origin_candidate
                shifted_motif = transformed_motif.shift_origin(shift)
                bnf_list = shifted_motif.to_bnf_list(element_order=sorted_elements)
                candidate = BNFCandidate(bnf_list[3:], shifted_motif, mat, shift)
                bnf_candidates.append(candidate)

        sorted_candidates = sorted(bnf_candidates, key=lambda c: c.bnf_coords)
        return BNFConstructionResult(
            original_motif,
            self.delta,
            self.stabilizer,
            sorted_candidates
        )