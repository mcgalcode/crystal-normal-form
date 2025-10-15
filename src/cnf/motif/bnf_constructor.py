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
    def canonical_candidate(self):
        return self.sorted_bnf_candidates[0]
    
    @property
    def canonical_motif(self):
        return self.canonical_candidate.motif
    
    @property
    def bnf(self):
        element_list, _ = self.canonical_motif.to_elements_and_positions()
        canonical_bnf_coords = self.canonical_candidate.bnf_coords

        return BasisNormalForm(canonical_bnf_coords, element_list, self.delta)

def get_all_shifted_motifs(m: FractionalMotif) -> tuple[list[FractionalMotif], list[np.ndarray]]:
    sorted_elements = m.sorted_elements
    origin_element = sorted_elements[0]

    origin_element_positions = m.get_element_positions(origin_element)
    # For each possible origin, compute the list
    shifted_motifs = []
    shifts = []
    for origin_candidate in origin_element_positions:
        shift = -origin_candidate
        shifted_motifs.append(m.shift_origin(shift))
        shifts.append(shift)
    return shifted_motifs, shifts


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
            # print(original_motif.to_bnf_list())
            # original_motif.print_details()
            transformed_motif = original_motif.apply_unimodular(mat)
            # print(transformed_motif.to_bnf_list())
            # if isinstance(transformed_motif, FractionalMotif):
                # transformed_motif = transformed_motif.discretize(self.delta)

            # if self.verbose_logging:
            # print()
            # transformed_motif.print_details()
            # transformed_motif.print_details()
            shifted_motifs, shifts = get_all_shifted_motifs(transformed_motif)

            for shifted_motif, shift in zip(shifted_motifs, shifts):
                bnf_list = shifted_motif.to_bnf_list()
                # print(bnf_list)
                candidate = BNFCandidate(bnf_list[3:], shifted_motif, mat, shift)
                bnf_candidates.append(candidate)

        sorted_candidates = sorted(bnf_candidates, key=lambda c: c.bnf_coords)
        return BNFConstructionResult(
            original_motif,
            self.delta,
            self.stabilizer,
            sorted_candidates
        )