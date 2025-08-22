import numpy as np

from pymatgen.core.composition import Element

def sort_elements(elements: list[str]) -> list[str]:
    return list(sorted(elements, key=lambda e: Element(e).number))

def move_coords_into_cell(frac_coords: np.array) -> np.array:
    return np.mod(frac_coords, 1)

def shift_coords(frac_coords: np.array, shift_vector: np.array) -> np.array:
    return move_coords_into_cell(frac_coords + shift_vector)

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
    number_tuples = [tuple(number_list) for number_list in list_of_lists]
    if simultaneously_sort is None:
        return [np.array(numbers) for numbers in sorted(number_tuples)]
    else:
        sorted_idxs = sorted(range(len(number_tuples)), key=number_tuples.__getitem__)
        sorted_list_of_lists = list(map(list_of_lists.__getitem__, sorted_idxs))
        other_sorted = [list(map(other_list.__getitem__, sorted_idxs)) for other_list in simultaneously_sort]
        return sorted_list_of_lists, other_sorted

def discretize_coords(frac_coords: np.array, num_discretization_intervals: int):
    interval_size = 1 / num_discretization_intervals
    rounded_coords = np.round(frac_coords / interval_size) * interval_size
    coords_in_cell = move_coords_into_cell(rounded_coords)
    return (coords_in_cell / interval_size).astype(np.int64)