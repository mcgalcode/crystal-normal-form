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
                 pretransforms: list[MatrixTuple],
                 sorted_bnf_candidates: list[BNFCandidate]):
        self.original_motif = original_motif
        self.pretransforms = pretransforms
        self.sorted_bnf_candidates = sorted_bnf_candidates
    
    def print_details(self):
        print(f"Found BNF: {self.bnf.coord_list}")
        print(f"Applied the following pretransforms...")
        for p in self.pretransforms:
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

        return BasisNormalForm(tuple([int(c) for c in canonical_bnf_coords]), element_list, canonical_candidate.motif.delta)        
    
class PreTransform():

    def __init__(self, mats: list[MatrixTuple]):
        if not isinstance(mats, list):
            raise ValueError(f"PreTransform requires a list of MatrixTuples")
        for m in mats:
            if not isinstance(m, MatrixTuple):
                raise ValueError(f"Tried to instantiate PreTransform for BNF construction with a bad object: {type(m)}")
        self.mats = mats


class BNFConstructor():
    """Implements methods for taking a list of atomic positions
    in fractional coordinates and producing the Basis Normal Form string
    as described in the section "Representation of Crystalline Atomic Bases" on
    pp. 52 of David Mrdjenovich's thesis.
    """
    
    def __init__(self,
                 pre_transforms: list[PreTransform] = None,
                 verbose_logging = False):
        if pre_transforms is None:
            pre_transforms = []

        filtered_pre_transforms = []
        for p in pre_transforms:
            if len(p.mats) == 1 and p.mats[0].is_identity():
                continue
            filtered_pre_transforms.append(p)
        
        # print(f"Using pretransforms")
        # for p in filtered_pre_transforms:
        #     print(p.mats)
        
        self.pre_transforms = filtered_pre_transforms
        self.verbose_logging = verbose_logging

    def build_from_fractional_motif(self,
                                    motif: FractionalMotif,
                                    delta: int = 10):
        disc_motif = motif.discretize(delta)
        return self.build(disc_motif)
    

    def _add_shifted_candidates(self, candidates: list[BNFCandidate], transformed_motif: DiscretizedMotif, mat):
        sorted_elements = transformed_motif.sorted_elements
        origin_element = sorted_elements[0]

        origin_element_positions = transformed_motif.get_element_positions(origin_element)
        # For each possible origin, compute the list
        for origin_candidate in origin_element_positions:
            shift = -origin_candidate
            shifted_motif = transformed_motif.shift_origin(shift)
            bnf_list = shifted_motif.to_bnf_list(element_order=sorted_elements)
            candidate = BNFCandidate(bnf_list[3:], transformed_motif, mat, shift)
            candidates.append(candidate)
    
    def build(self, disc_motif: DiscretizedMotif):
        if not isinstance(disc_motif, DiscretizedMotif):
            raise ValueError("Tried to find canonical BNF for non-discretized motif!")

        bnf_candidates: list[BNFCandidate] = []
            
        mat_sets = [p.mats for p in self.pre_transforms]
        for mat_chain in itertools.product(*mat_sets):
            if len(mat_chain) == 0:
                transformed_motif = disc_motif
            else:               
                combined = combine_unimodular_matrices(mat_chain)
                transformed_motif = disc_motif.apply_unimodular(combined)
                if self.verbose_logging:
                    print(f"Trying mat chain: {mat_chain}")
                    print(f"Equivalent single mat: {combined}")
            self._add_shifted_candidates(
                bnf_candidates, transformed_motif, mat_chain
            )                

        sorted_candidates = sorted(bnf_candidates, key=lambda c: c.bnf_coords)
        return BNFConstructionResult(
            disc_motif,
            self.pre_transforms,
            sorted_candidates
        )