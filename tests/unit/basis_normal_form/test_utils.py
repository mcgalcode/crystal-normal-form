import pytest
import numpy as np

import cnf.motif.utils as bnf_utils

@pytest.mark.parametrize(
    "unnormalized_coords,expected_coords",
    [
        (np.array([1.1, 0.5, -0.1]), np.array([0.1, 0.5, 0.9])),
        (np.array([0.0, 0.5, 0.7]), np.array([0.0, 0.5, 0.7])),
        (np.array([-0.0, -1.0, -2.1]), np.array([0.0, 0.0, 0.9])),
    ]
)
def test_can_move_coords_into_pbc_cell(unnormalized_coords, expected_coords):
    computed_cell_coords = bnf_utils.move_coords_into_cell(unnormalized_coords, mod=1)
    assert np.all(np.isclose(computed_cell_coords, expected_coords, 1e-16))

@pytest.mark.parametrize(
    "unsorted_element_list,expected_sorted_list",
    [
        (["Pt", "H", "Sb"], ["H", "Sb", "Pt"]),
        (["W", "Ta", "Hf", "Cs"], ["Cs", "Hf", "Ta", "W"]),
        (["Ca", "Mg", "Be"], ["Be", "Mg", "Ca"])
    ]
)
def test_can_sort_elements(unsorted_element_list, expected_sorted_list):
    bnf_sorted = bnf_utils.sort_elements(unsorted_element_list)
    for actual, expected in zip(bnf_sorted, expected_sorted_list):
        assert actual == expected

@pytest.mark.parametrize(
    "original_coords,shift_vec,expected_coords",
    [
        (
            np.array([0.5, 0.5, 0.6]),
            np.array([0.8, -0.8, 1.6]),
            np.array([0.3, 0.7, 0.2])
        )
    ]
)
def test_can_shift_coords(original_coords, shift_vec, expected_coords):
    shifted = bnf_utils.shift_coords(original_coords, shift_vec, mod=1)
    assert np.all(np.isclose(shifted, expected_coords))


@pytest.mark.parametrize(
    "unsorted_positions,expected_positions",
    [
        (
            [
                np.array([0.3, 0.5, 0.5]),
                np.array([0.3, 0.3, 0.5]),
                np.array([0.2, 0.7, 0.5]),
            ],
            [
                np.array([0.2, 0.7, 0.5]),
                np.array([0.3, 0.3, 0.5]),
                np.array([0.3, 0.5, 0.5]),
            ],
        ),
        (
            [
                np.array([0.9, 0.0, 0.0]),
                np.array([0.3, 0.3, 0.3]),
                np.array([0.9, 0.0, 0.0]),
            ],
            [
                np.array([0.3, 0.3, 0.3]),
                np.array([0.9, 0.0, 0.0]),
                np.array([0.9, 0.0, 0.0]),
            ],
        ),        
        (
            [
                np.array([0.3, 0.5, 0.5]),
            ],
            [
                np.array([0.3, 0.5, 0.5]),
            ],
        ),
        (
            [
                np.array([0.3, 0.5, 0.5]),
                np.array([0.3, 0.3, 0.5]),
            ],
            [
                np.array([0.3, 0.3, 0.5]),
                np.array([0.3, 0.5, 0.5]),
            ],
        )       
    ]
)
def test_can_sort_positions(unsorted_positions, expected_positions):
    sorted_positions = bnf_utils.sort_number_lists(unsorted_positions)

    for actual, expected in zip(sorted_positions, expected_positions):
        assert np.isclose(actual, expected, 1e-16).all()


def test_can_simultaneouslyt_sort():
    list_of_lists = [
        [1, 1, 3],
        [1, 0, 3],
        [0, 0, 3],
    ]

    other_lists = [
        ["a", "b", "c"],
        ["x", "v", "f"]
    ]
    sorted_list_of_lists, sorted_other_lists = bnf_utils.sort_number_lists(list_of_lists, other_lists)

    assert (sorted_list_of_lists[0] == np.array([0, 0, 3])).all()
    assert (sorted_list_of_lists[1] == np.array([1, 0, 3])).all()
    assert (sorted_list_of_lists[2] == np.array([1, 1, 3])).all()

    assert sorted_other_lists[0][0] == "c"
    assert sorted_other_lists[0][1] == "b"
    assert sorted_other_lists[0][2] == "a"

    assert sorted_other_lists[1][0] == "f"
    assert sorted_other_lists[1][1] == "v"
    assert sorted_other_lists[1][2] == "x"


@pytest.mark.parametrize("raw_coords,num_intervals,expected_ints",[
    (
        np.array([0.13, 0.27, 0.98]),
        10,
        np.array([1, 3, 0])
    ),
    (
        np.array([0.13, 0.27, 0.98]),
        20,
        np.array([3, 5, 0])
    ),
    (
        np.array([0.13, 0.27, 0.98]),
        50,
        np.array([6, 14, 49])
    ),
])
def test_can_discretize_coords(raw_coords, num_intervals, expected_ints):
    actual_ints = bnf_utils.discretize_coords(raw_coords, num_intervals)
    assert (actual_ints == expected_ints).all()

def test_move_coords_into_cell_shouldnt_leave_limit():
    coords = np.array([[1.0, 1.0, 1.0]])
    result = bnf_utils.move_coords_into_cell(coords, 1)
    assert np.all(result == np.array([0,0,0]))