from .element_position_map import ElementPositionMap
from .utils import sort_number_lists

class BasisNormalForm():
    """Implements methods for taking a list of atomic positions
    in fractional coordinates and producing the Basis Normal Form string
    as described in the section "Representation of Crystalline Atomic Bases" on
    pp. 52 of David Mrdjenovich's thesis.
    """

    @classmethod
    def from_element_and_position_list(cls, elements, positions):
        el_pos_map = ElementPositionMap.from_elements_and_positions(elements, positions)
        return cls.from_element_position_map(el_pos_map)

    @classmethod
    def from_element_position_map(cls, el_pos_map: ElementPositionMap, delta: int = 10):
        sorted_elements = el_pos_map.sorted_elements
        origin_element = sorted_elements[0]

        origin_element_positions = el_pos_map.get_element_positions(origin_element)

        all_bnf_lists = []

        # For each possible origin, compute the list
        for origin_candidate in origin_element_positions:
            shifted_el_pos_map = el_pos_map.shift_origin(-origin_candidate)
            bnf_list = shifted_el_pos_map.to_bnf_list(delta, element_order=sorted_elements)
            all_bnf_lists.append(bnf_list)
        
        sorted_bnf_lists = sort_number_lists(all_bnf_lists)

        canonical_bnf_list = sorted_bnf_lists[0]
        truncated_bnf_list = canonical_bnf_list[3:]

        element_list, _ = el_pos_map.to_elements_and_positions()

        return cls(truncated_bnf_list, element_list, delta)

    def __init__(self, coord_list, element_list, delta):
        self.coord_list = coord_list
        self.elements = element_list
        self.delta = delta

    def to_element_position_map(self):
        frac_coords = [c / self.delta for c in self.coord_list]
        separated_coord_lists = [frac_coords[start_idx:start_idx+3] for start_idx in range(0, len(frac_coords), 3)]
        separated_coord_lists = [[0, 0, 0]] + separated_coord_lists

        return ElementPositionMap.from_elements_and_positions(self.elements, separated_coord_lists)
