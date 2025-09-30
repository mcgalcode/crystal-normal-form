import numpy as np
from pymatgen.core.structure import Structure
from .utils import shift_coords, discretize_coords, sort_elements, sort_number_lists

from ..lattice import Superbasis
from ..linalg import MatrixTuple

class AtomicMotif():
        
    @classmethod
    def from_elements_and_positions(cls, elements, positions):
        if len(elements) != len(positions):
            raise ValueError("Must instantiate motifs with same num elements and positions")
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
        elements, positions = self.to_elements_and_positions()
        self.atoms = elements
        self.positions = positions
        self.coord_matrix = np.array(self.positions).T
    
    def get_element_positions(self, el: str):
        if el not in self.map:
            raise ValueError(f"Element {el} not found in AtomicMotif")
        
        return self.map[el]

    def unique_elements(self):
        return list(self.map.keys())

    def element_count(self, el):
        return len(self.get_element_positions(el))

    def to_elements_and_positions(self):
        elements  = []
        positions = []

        for element in self.sorted_elements:
            for position in self.get_element_positions(element):
                elements.append(element)
                positions.append(position)
        
        return elements, positions

    def shift_origin(self, shift_vector: np.array):
        if not self.is_valid_shift_vector(shift_vector):
            raise ValueError(f"Tried to shift origin in {self.__class__} using invalid shift vector: {shift_vector}")
        
        shift_vector = np.array(shift_vector)
        
        new_map = {}

        for el, pos_list in self.map.items():
            new_map[el] = []
            for pos in pos_list:
                new_map[el].append(shift_coords(pos, shift_vector))
        
        return self.__class__(new_map)

    def is_valid_shift_vector(self, shift_vector: np.array):
        return True
    
    def __len__(self):
        return sum([len(positions) for _, positions in self.map.items()])
    
    def __repr__(self):
        return self.map.__repr__()
    
    def __eq__(self, other: 'AtomicMotif'):
        els_eq = tuple(self.atoms) == tuple(other.atoms)
        if not els_eq:
            return False
        
        def lists_of_np_arrays_approx_eq(l1, l2):
            l1_tuples = set([tuple(np.round(l, 5)) for l in l1])
            l2_tuples = set([tuple(np.round(l, 5)) for l in l2])
            return l1_tuples == l2_tuples and len(l1) == len(l2)
            
        for el in self.unique_elements():
            self_positions = self.get_element_positions(el)
            other_positions = other.get_element_positions(el)
            if not lists_of_np_arrays_approx_eq(self_positions, other_positions):
                return False
        return True 
class FractionalMotif(AtomicMotif):

    @classmethod
    def from_pymatgen_structure(cls, pmg_struct: Structure):
        site_species = [site.species for site in pmg_struct.sites]
        elements = []
        for spec in site_species:
            assert spec.num_atoms == 1
            elements.append(spec.elements[0])

        coords = [site.frac_coords for site in pmg_struct.sites]
        return cls.from_elements_and_positions(elements, coords)

    def get_discretized_positions(self, el: str, delta: int):
        unrounded = self.get_element_positions(el)
        return [discretize_coords(c, delta) for c in unrounded]
    
    def transform(self, transform: np.array):
        transformed_coords = self.coord_matrix.T @ transform
        return FractionalMotif.from_elements_and_positions(self.atoms, transformed_coords)
    
    def apply_unimodular(self, unimodular: MatrixTuple):
        if not np.isclose(unimodular.determinant(), 1):
            raise ValueError(f"Tried to transform motif w matrix with det {unimodular.determinant()}")
        transformed = unimodular.inverse() @ self.coord_matrix
        return FractionalMotif.from_elements_and_positions(self.atoms, transformed.T)
    
    def compute_cartesian_coords_in_basis(self, basis: Superbasis):
        cart_coords = basis.generating_vecs().T @ self.coord_matrix
        return CartesianMotif.from_elements_and_positions(self.atoms, cart_coords.T)

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
        return tuple(el for coord_list in discretized_positions_list for el in coord_list)


class CartesianMotif(AtomicMotif):
    
    @classmethod
    def from_pymatgen_structure(cls, pmg_struct: Structure):
        elements = [site.species for site in pmg_struct.sites]
        coords = [site.coords for site in pmg_struct.sites]
        return CartesianMotif.from_elements_and_positions(elements, coords)
        
    
    
