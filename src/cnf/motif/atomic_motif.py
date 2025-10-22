import numpy as np
from pymatgen.core.structure import Structure
from .utils import shift_coords, discretize_coords, sort_elements, sort_number_lists, move_coords_into_cell

from ..lattice import Superbasis
from ..linalg import MatrixTuple
from ..lattice.permutations import apply_permutation

import itertools

def construct_el_pos_map(elements, positions):
    if len(elements) != len(positions):
        raise ValueError("Must instantiate motifs with same num elements and positions")
    element_to_positions_map = {}

    for el, pos in zip(elements, positions):
        if el in element_to_positions_map:
            element_to_positions_map[el].append(np.array(pos))
        else:
            element_to_positions_map[el] = [np.array(pos)]
    
    return element_to_positions_map


class AtomicMotif():
        
    @classmethod
    def from_elements_and_positions(cls, elements, positions):
        return cls(construct_el_pos_map(elements, positions))
    
    @classmethod
    def from_mnf_list(cls, mnf_coord_list, element_list, **kwargs):
        separated_coord_lists = [mnf_coord_list[start_idx:start_idx+3] for start_idx in range(0, len(mnf_coord_list), 3)]
        separated_coord_lists = [[0, 0, 0]] + separated_coord_lists

        return cls.from_elements_and_positions(element_list, separated_coord_lists, **kwargs)        

    def __init__(self, element_to_positions_map: dict[str, list[np.array]]):
        self.map = element_to_positions_map
        self.sorted_elements = sort_elements(self.unique_elements())
        self.sorted_elements.reverse()
        elements, positions = self.to_elements_and_positions()
        positions = [np.array(pos) for pos in positions]
        positions = [self._process_transformed_coords(c) for c in positions]
        for pos in positions:
            self.validate_positions(pos)
        self.atoms = elements
        self.positions = positions
        self.coord_matrix = np.array(self.positions).T
    
    def _get_kwargs(self):
        return {}
    
    def get_element_positions(self, el: str):
        if el not in self.map:
            raise ValueError(f"Element {el} not found in AtomicMotif")
        
        return self.map[el]

    def unique_elements(self):
        return list(self.map.keys())

    def element_count(self, el):
        return len(self.get_element_positions(el))

    def validate_positions(self, pos):
        if not isinstance(pos, np.ndarray):
            raise ValueError(f"Encountered non-np.ndarray in AtomicMotif instantiation: {pos}")
    
    def get_matching(self, other: 'AtomicMotif'):
        assert len(self.atoms) == len(other.atoms)
        for el in self.sorted_elements:
            self_positions = [tuple(pos) for pos in self.get_element_positions(el)]
            othe_positions = [tuple(pos) for pos in other.get_element_positions(el)]
            if tuple(sorted(self_positions)) != tuple(sorted(othe_positions)):
                return False
        return True
                
    def print_details(self):
        for atom, pos in zip(self.atoms, self.positions):
            print(f"{atom}: {pos.tolist()}")

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
                new_map[el].append(self.shift_coord(pos, shift_vector))
        
        return self.__class__(new_map, **self._get_kwargs())
    
    def shift_coord(self, pos, shift_vector):
        return shift_coords(pos, shift_vector, None)

    def get_sorted_positions(self, element_order = None):
        if element_order is None:
            element_order = self.sorted_elements

        sorted_positions = []
        for el in element_order:
            position_list = self.get_element_positions(el)
            sorted_positions.extend(sort_number_lists(position_list))

        return sorted_positions
    
    def to_mnf_list(self, element_order = None, sort = False):
        if sort:
            element_order = self.sorted_elements
            sorted_positions = self.get_sorted_positions(element_order)
        elif element_order is not None:
            sorted_positions = self.get_sorted_positions(element_order)
        else:
            sorted_positions = self.positions
        return tuple([self._process_mnf_list_coord(el) for coord_list in sorted_positions[1:] for el in coord_list])
    
    def _process_mnf_list_coord(self, coord):
        return coord

    def apply_unimodular(self, unimodular: MatrixTuple, skip_det_check = False):
        if not skip_det_check and not np.isclose(unimodular.determinant(), 1):
            raise ValueError(f"Tried to transform motif w matrix with det {unimodular.determinant()}")
        inv = unimodular.inverse().matrix
        transformed = inv @ self.coord_matrix
        positions = transformed.T
        positions = self._process_transformed_coords(positions)
        return self.__class__.from_elements_and_positions(self.atoms, positions, **self._get_kwargs())
    
    def transform(self, transform: np.array):
        transformed_coords = self.coord_matrix.T @ transform
        positions = self._process_transformed_coords(transformed_coords)
        return self.__class__.from_elements_and_positions(self.atoms, positions, **self._get_kwargs())
    
    def invert(self):
        inversion = MatrixTuple(-np.eye(3))
        return self.apply_unimodular(inversion, skip_det_check=True)

    def has_inversion_symmetry(self, atol=1e-6):
        return self.find_match(self.invert(), atol=atol)
    
    def find_inverted_match(self, other, atol=1e-6):
        return self.invert().find_match(other, atol=atol)

    def find_match(self, other: 'AtomicMotif', atol=1e-6):

        species1, coords1 = self.to_elements_and_positions()
        species2, coords2 = other.to_elements_and_positions()

        coords1 = np.asarray(coords1)
        coords2 = np.asarray(coords2)
        
        if coords1.shape != coords2.shape:
            return False
        
        if len(species1) != len(species2):
            return False
                
        # Build a list of (species, coords) for coords2
        atoms2 = list(zip(species2, coords2))
        
        # For each inverted atom in coords1, find matching atom in coords2
        for i, coord1 in enumerate(coords1):
            species_to_match = species1[i]
            found = False
            
            for j, (sp2, coord2) in enumerate(atoms2):
                if sp2 == species_to_match:
                    # Check if coordinates match (with PBC)
                    diff = np.abs(coord1 - coord2)
                    if np.all(diff < atol):
                        found = True
                        break
            
            if not found:
                return False
        
        return True

    def _process_transformed_coords(self, coords):
        return coords

    def is_valid_shift_vector(self, shift_vector: np.array):
        return True
    
    def __len__(self):
        return sum([len(positions) for _, positions in self.map.items()])
    
    def __repr__(self):
        return self.map.__repr__()
    
    def __hash__(self):
        return self.to_mnf_list().__hash__()
    
    def __eq__(self, other: 'AtomicMotif'):
        els_eq = tuple(self.atoms) == tuple(other.atoms)
        if not els_eq:
            return False
        
        def lists_of_np_arrays_approx_eq(l1, l2):
            l1_tuples = set([tuple(np.round(l, 7)) for l in l1])
            l2_tuples = set([tuple(np.round(l, 7)) for l in l2])
            return l1_tuples == l2_tuples and len(l1) == len(l2)
            
        for el in self.unique_elements():
            self_positions = self.get_element_positions(el)
            other_positions = other.get_element_positions(el)
            if not lists_of_np_arrays_approx_eq(self_positions, other_positions):
                return False
        return True
    
    @property
    def position_tuple_list(self):
        tups = []
        for p in self.positions:
            tup = tuple([self._process_mnf_list_coord(i) for i in p])
            tups.append(tup)
        return tups
    
    @property
    def num_origin_atoms(self):
        return self.element_count(self.sorted_elements[0])

    
class PeriodicMotif(AtomicMotif):

    @classmethod
    def from_elements_and_positions(cls, elements, positions, mod):
            return cls(construct_el_pos_map(elements, positions), mod)

    def __init__(self, element_to_positions_map: dict[str, list[np.array]], mod = None):
        if mod is None:
            raise ValueError(f"Tried to instantiate {self.__class__} without mod")
        
        self._mod = mod
        super().__init__(element_to_positions_map)

    def validate_positions(self, pos):
        if np.any(np.array(pos) < 0):
            raise ValueError(f"AtomicMotif instantiated with negative position: {pos}")

    def shift_coord(self, pos, shift_vector):
        return shift_coords(pos, shift_vector, self._mod)
    
    def _process_transformed_coords(self, coords):
        return move_coords_into_cell(coords, self._mod)

    def _get_kwargs(self):
        return { "mod": self._mod }
    
class FractionalMotif(PeriodicMotif):

    @classmethod
    def from_elements_and_positions(cls, elements, positions):
        return cls(construct_el_pos_map(elements, positions))       

    def __init__(self, element_to_positions_map: dict[str, list[np.array]]):
        FRACTIONAL_MOD = 1
        super().__init__(element_to_positions_map, FRACTIONAL_MOD)

    @classmethod
    def from_pymatgen_structure(cls, pmg_struct: Structure):
        site_species = [site.species for site in pmg_struct.sites]
        elements = []
        for spec in site_species:
            assert spec.num_atoms == 1
            elements.append(spec.elements[0])

        coords = [site.frac_coords for site in pmg_struct.sites]
        return cls.from_elements_and_positions(elements, coords)

    def _process_mnf_list_coord(self, coord):
        return round(float(coord), 6)
    
    def compute_cartesian_coords_in_basis(self, basis: Superbasis):
        cart_coords = basis.generating_vecs().T @ self.coord_matrix
        return CartesianMotif.from_elements_and_positions(self.atoms, cart_coords.T)
    
    def discretize(self, delta):
        discretized = discretize_coords(self.coord_matrix.T, delta)
        res =  DiscretizedMotif.from_elements_and_positions(self.atoms, discretized, delta)
        return res  
    
    def _get_kwargs(self):
        return {}

class DiscretizedMotif(PeriodicMotif):

    @classmethod
    def from_elements_and_positions(cls, elements, positions, delta):
        return cls(construct_el_pos_map(elements, positions), delta)

    def __init__(self, element_to_positions_map: dict[str, list[np.array]], delta = None):
        if not isinstance(delta, int):
            raise ValueError(f"Tried to instantiate discretized motif w bad delta: {delta}")
        self.delta = delta
        super().__init__(element_to_positions_map, delta)

    def is_valid_shift_vector(self, shift_vector: np.array):
        return np.all(shift_vector < self._mod)

    def _process_mnf_list_coord(self, coord):
        return int(coord)
    
    def to_fractional_motif(self):
        atoms = self.atoms
        positions = np.array(self.positions) / self.delta
        return FractionalMotif.from_elements_and_positions(atoms, positions)

    def _get_kwargs(self):
        return { "delta": self.delta }


class CartesianMotif(AtomicMotif):
    
    @classmethod
    def from_pymatgen_structure(cls, pmg_struct: Structure):
        elements = [site.species for site in pmg_struct.sites]
        coords = [site.coords for site in pmg_struct.sites]
        return CartesianMotif.from_elements_and_positions(elements, coords)
        
    
    
