import numpy as np

from pymatgen.core.composition import Element
from ..lattice.permutations import apply_permutation

def sort_elements(elements: list[str]) -> list[str]:
    return list(sorted(elements, key=lambda e: Element(e).number))

def move_coords_into_cell(frac_coords: np.array, mod) -> np.array:
    return np.mod(frac_coords, mod)

def shift_coords(frac_coords: np.array, shift_vector: np.array, mod) -> np.array:
    moved = frac_coords + shift_vector
    if mod is not None:
        return move_coords_into_cell(moved, mod)
    else:
        return moved

def sort_number_lists(list_of_lists: list[np.array], simultaneously_sort = None) -> list[np.array]:
    """Sorts a list of lists of numbers. This is achieved by leveraging
    the fact that python sorts tuples in lexicographical order

    Parameters
    ----------
    list_of_lists : list[np.array]
        The list of lists to sort (note, only the outer list is sorted.)

    Returns
    -------
    list[np.array]
        The sorted list of lists
    """
    number_tuples = [(tuple(number_list), idx) for idx, number_list in enumerate(list_of_lists)]
    sorted_pairs = sorted(number_tuples, key=lambda x: x[0])
    sorted_lists = [p[0] for p in sorted_pairs]
    idxs = tuple([p[1] for p in sorted_pairs])
    numpy_num_lists = [np.array(numbers) for numbers in sorted_lists]
    if simultaneously_sort is None:
        return numpy_num_lists
    else:
        return numpy_num_lists, [apply_permutation(l, idxs) for l in simultaneously_sort]

def discretize_coords(frac_coords: np.array, num_discretization_intervals: int):
    interval_size = 1 / num_discretization_intervals
    rounded_coords = np.round(frac_coords / interval_size) * interval_size
    coords_in_cell = move_coords_into_cell(rounded_coords, mod = 1)
    return (coords_in_cell / interval_size).astype(np.int64)