import numpy as np
from .utils import shift_coords, discretize_coords, sort_elements, sort_number_lists

class ElementPositionMap():

    
    @classmethod
    def from_elements_and_positions(cls, elements, positions):
        element_to_positions_map = {}

        for el, pos in zip(elements, positions):
            if el in element_to_positions_map:
                element_to_positions_map[el].append(np.array(pos))
            else:
                element_to_positions_map[el] = [np.array(pos)]
        
        return cls(element_to_positions_map)

    def __init__(self, element_to_positions_map: dict[str, list[np.array]]):
        self.map = element_to_positions_map
        self.sorted_elements = sort_elements(self.unique_elements())
        self.sorted_elements.reverse()
    
    def get_element_positions(self, el: str):
        if el not in self.map:
            raise ValueError(f"Element {el} not found in ElementPositionMap")
        
        return self.map[el]
    
    def get_discretized_positions(self, el: str, delta: int):
        unrounded = self.get_element_positions(el)
        return [discretize_coords(c, delta) for c in unrounded]

    def unique_elements(self):
        return list(self.map.keys())

    def to_elements_and_positions(self):
        elements  = []
        positions = []

        for element in self.sorted_elements:
            for position in self.get_element_positions(element):
                elements.append(element)
                positions.append(position)
        
        return elements, positions

    def __len__(self):
        return sum([len(positions) for _, positions in self.map.items()])
    
    def __repr__(self):
        return self.map.__repr__()
    
    def shift_origin(self, shift_vector: np.array):
        shift_vector = np.array(shift_vector)
        
        new_map = {}

        for el, pos_list in self.map.items():
            new_map[el] = []
            for pos in pos_list:
                new_map[el].append(shift_coords(pos, shift_vector))
        
        return ElementPositionMap(new_map)
    
    def get_sorted_discretized_positions(self, delta, element_order = None):
        if element_order is None:
            element_order = self.sorted_elements

        sorted_positions = []
        for el in element_order:
            position_list = self.get_discretized_positions(el, delta)
            sorted_positions.extend(sort_number_lists(position_list))

        return sorted_positions

    def to_bnf_list(self, delta, element_order = None):
        if element_order is None:
            element_order = self.sorted_elements

        discretized_positions_list = self.get_sorted_discretized_positions(delta, element_order)
        return [el for coord_list in discretized_positions_list for el in coord_list]