import numpy as np

from .atomic_motif import FractionalMotif, DiscretizedMotif
from ..lattice.permutations import PermutationMatrix
from ..linalg import MatrixTuple
from .basis_normal_form import BasisNormalForm
    
class BNFCandidate():

    def __init__(self,
                 bnf_coords: tuple,
                 motif: DiscretizedMotif,
                 stabilizer_member: PermutationMatrix,
                 unimodular: MatrixTuple,
                 shift: np.array):
        self.bnf_coords = bnf_coords
        self.motif = motif
        self.stabilizer_member = stabilizer_member
        self.unimodular = unimodular
        self.shift = shift

class BNFConstructionResult():

    def __init__(self,
                 original_motif: DiscretizedMotif,
                 pretransforms: list[MatrixTuple],
                 pretransformed_motif: DiscretizedMotif,
                 sorted_bnf_candidates: list[BNFCandidate]):
        self.original_motif = original_motif
        self.pretransforms = pretransforms
        self.pretransformed_motif = pretransformed_motif
        self.sorted_bnf_candidates = sorted_bnf_candidates
    
    @property
    def bnf(self):
        canonical_candidate = self.sorted_bnf_candidates[0]
        element_list, _ = canonical_candidate.motif.to_elements_and_positions()
        canonical_bnf_coords= canonical_candidate.bnf_coords

        return BasisNormalForm(tuple([int(c) for c in canonical_bnf_coords]), element_list, canonical_candidate.motif.delta)        

class BNFConstructor():
    """Implements methods for taking a list of atomic positions
    in fractional coordinates and producing the Basis Normal Form string
    as described in the section "Representation of Crystalline Atomic Bases" on
    pp. 52 of David Mrdjenovich's thesis.
    """
    
    def __init__(self,
                 pre_transforms: list[MatrixTuple] = None,
                 stabilizer: list[PermutationMatrix] = None):
        if pre_transforms is None:
            pre_transforms = []
        
        if stabilizer is None:
            stabilizer = []

        self.pre_transforms = pre_transforms
        self.stabilizer = stabilizer

    def build_from_fractional_motif(self,
                                    motif: FractionalMotif,
                                    delta: int = 10):
        disc_motif = motif.discretize(delta)
        return self.build(disc_motif)
    

    def _add_shifted_candidates(self, candidates: list[BNFCandidate], transformed_motif: DiscretizedMotif, perm_matrix, mat):
        sorted_elements = transformed_motif.sorted_elements
        origin_element = sorted_elements[0]

        origin_element_positions = transformed_motif.get_element_positions(origin_element)
        # For each possible origin, compute the list
        for origin_candidate in origin_element_positions:
            shift = -origin_candidate
            shifted_motif = transformed_motif.shift_origin(shift)
            bnf_list = shifted_motif.to_bnf_list(element_order=sorted_elements)
            candidate = BNFCandidate(bnf_list[3:], transformed_motif, perm_matrix, mat, shift)
            candidates.append(candidate)
    
    def build(self, disc_motif: DiscretizedMotif):
        if not isinstance(disc_motif, DiscretizedMotif):
            raise ValueError("Tried to find canonical BNF for non-discretized motif!")
    
        pretransformed_motif = disc_motif
        for t in self.pre_transforms:
            pretransformed_motif = pretransformed_motif.apply_unimodular(t)

        bnf_candidates: list[BNFCandidate] = []

        if len(self.stabilizer) > 0:
            for perm_matrix in self.stabilizer:
                for mat in perm_matrix.all_matrices:
                    transformed_motif = pretransformed_motif.apply_unimodular(mat)
                    self._add_shifted_candidates(
                        bnf_candidates, transformed_motif, perm_matrix, mat
                    )
        else:
            self._add_shifted_candidates(
                bnf_candidates, pretransformed_motif, None, None
            )
                
        sorted_candidates = sorted(bnf_candidates, key=lambda c: c.bnf_coords)
        return BNFConstructionResult(
            disc_motif,
            self.pre_transforms,
            pretransformed_motif, 
            sorted_candidates
        )