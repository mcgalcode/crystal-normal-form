from .atomic_motif import FractionalMotif
from .utils import sort_number_lists

class BasisNormalForm():
    """Implements methods for taking a list of atomic positions
    in fractional coordinates and producing the Basis Normal Form string
    as described in the section "Representation of Crystalline Atomic Bases" on
    pp. 52 of David Mrdjenovich's thesis.
    """

    @classmethod
    def from_element_and_position_lists(cls, elements, positions):
        motif = FractionalMotif.from_elements_and_positions(elements, positions)
        return cls.from_motif(motif)

    @classmethod
    def from_motif(cls, motif: FractionalMotif, delta: int = 10):
        sorted_elements = motif.sorted_elements
        origin_element = sorted_elements[0]

        origin_element_positions = motif.get_element_positions(origin_element)

        all_bnf_lists = []

        # For each possible origin, compute the list
        for origin_candidate in origin_element_positions:
            shifted_motif = motif.shift_origin(-origin_candidate)
            bnf_list = shifted_motif.to_bnf_list(delta, element_order=sorted_elements)
            all_bnf_lists.append(bnf_list)
        
        sorted_bnf_lists = sort_number_lists(all_bnf_lists)

        canonical_bnf_list = sorted_bnf_lists[0]
        truncated_bnf_list = canonical_bnf_list[3:]

        element_list, _ = motif.to_elements_and_positions()

        return cls(tuple([int(c) for c in truncated_bnf_list]), element_list, delta)

    def __init__(self, coord_list, element_list, delta):
        self.coord_list = coord_list
        self.elements = element_list
        self.delta = delta

    def to_element_position_map(self):
        frac_coords = [c / self.delta for c in self.coord_list]
        separated_coord_lists = [frac_coords[start_idx:start_idx+3] for start_idx in range(0, len(frac_coords), 3)]
        separated_coord_lists = [[0, 0, 0]] + separated_coord_lists

        return FractionalMotif.from_elements_and_positions(self.elements, separated_coord_lists)
    
    def __repr__(self):
        return f"BasisNormalForm({self.coord_list},elements={self.elements},delta={self.delta})"
