import pytest
import numpy as np

from pymatgen.core.composition import Element
from cnf.basis_normal_form import ElementPositionMap
import cnf.basis_normal_form.utils as bnf_utils

def test_can_initialize(sn2_o4_els_and_positions):
    els, positions = sn2_o4_els_and_positions
    el_pos_map = ElementPositionMap.from_elements_and_positions(els, positions)

    for el, pos in zip(els, positions):
        stored_positions = el_pos_map.get_element_positions(el)
        found_match = False
        for stored_pos in stored_positions:
            found_match = np.all(np.isclose(stored_pos, pos, 1e-16))
            if found_match:
                break

        assert found_match, f"No match found for {el} at {pos}"

def test_can_report_length(sn2_o4_el_pos_map):
    assert len(sn2_o4_el_pos_map) == 6

def test_can_report_elements(sn2_o4_el_pos_map):
    assert "Sn" in sn2_o4_el_pos_map.unique_elements()
    assert "O" in sn2_o4_el_pos_map.unique_elements()
    assert len(sn2_o4_el_pos_map.unique_elements()) == 2

def test_can_sort_positions():
    elements = ["H", "Br", "O", "O"]
    positions = [
        (0.5, 0.6, 0.4),
        (0, 0.2, 0.3),
        (0.1, 0.1, 0.1),
        (0.2, 0.1, 0.8)
    ]
    el_pos_map = ElementPositionMap.from_elements_and_positions(elements, positions)
    sorted_positions = el_pos_map.get_sorted_discretized_positions(10)
    assert np.all(sorted_positions[0] == [5, 6, 4])
    assert np.all(sorted_positions[1] == [1, 1, 1])
    assert np.all(sorted_positions[2] == [2, 1, 8])
    assert np.all(sorted_positions[3] == [0, 2, 3])

@pytest.mark.parametrize("shift_vector", [
    (np.array([-0.5, 0, -0.5]))
])
def test_can_shift_positions(
    shift_vector,
    sn2_o4_els_and_positions
):
    els, original_positions = sn2_o4_els_and_positions
    shifted_positions = [bnf_utils.shift_coords(np.array(c), shift_vector) for c in original_positions]

    original_map = ElementPositionMap.from_elements_and_positions(els, original_positions)
    shifted_map = original_map.shift_origin(shift_vector)

    for el, shifted_pos in zip(els, shifted_positions):
        stored_positions = shifted_map.get_element_positions(el)
        found_match = False
        for stored_pos in stored_positions:
            found_match = np.all(np.isclose(stored_pos, shifted_pos, 1e-16))
            if found_match:
                break

        assert found_match, f"No match found for {el} at shifted position: {shifted_pos}"


def test_from_element_position():
    elements = ["O", "O", "Sn", "O", "Sn", "O"]
    positions = [
        (0, 0.3, 0.3),
        (0.5, 0.8, 0.2),
        (0.5, 0.5, 0.5),
        (0, 0.7, 0.7),
        (0, 0, 0),
        (0.5, 0.2, 0.8)
    ]