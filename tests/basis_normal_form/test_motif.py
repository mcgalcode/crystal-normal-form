import pytest
import numpy as np
import helpers
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from cnf.motif import FractionalMotif
from cnf.lattice.permutations import VONORM_PERMUTATION_TO_CONORM_PERMUTATION, VonormPermutation
from cnf.lattice.superbasis import Superbasis
from cnf import CrystalNormalForm
from cnf.cnf_constructor import CNFConstructor
from cnf.lattice.rounding import DiscretizedVonormComputer
from cnf.linalg import MatrixTuple

import cnf.motif.utils as bnf_utils

def test_can_initialize(sn2_o4_els_and_positions):
    els, positions = sn2_o4_els_and_positions
    el_pos_map = FractionalMotif.from_elements_and_positions(els, positions)

    for el, pos in zip(els, positions):
        stored_positions = el_pos_map.get_element_positions(el)
        found_match = False
        for stored_pos in stored_positions:
            found_match = np.all(np.isclose(stored_pos, pos, 1e-16))
            if found_match:
                break

        assert found_match, f"No match found for {el} at {pos}"

def test_can_report_length(sn2_o4_motif: FractionalMotif):
    assert len(sn2_o4_motif) == 6

def test_can_report_elements(sn2_o4_motif: FractionalMotif):
    assert "Sn" in sn2_o4_motif.unique_elements()
    assert "O" in sn2_o4_motif.unique_elements()
    assert len(sn2_o4_motif.unique_elements()) == 2

def test_can_sort_positions():
    elements = ["H", "Br", "O", "O"]
    positions = [
        (0.5, 0.6, 0.4),
        (0, 0.2, 0.3),
        (0.1, 0.1, 0.1),
        (0.2, 0.1, 0.8)
    ]
    motif = FractionalMotif.from_elements_and_positions(elements, positions)
    motif = motif.discretize(10)
    sorted_positions = motif.get_sorted_positions()
    assert np.all(sorted_positions[3] == [5, 6, 4])
    assert np.all(sorted_positions[1] == [1, 1, 1])
    assert np.all(sorted_positions[2] == [2, 1, 8])
    assert np.all(sorted_positions[0] == [0, 2, 3])

@pytest.mark.parametrize("shift_vector", [
    (np.array([-0.5, 0, -0.5]))
])
def test_can_shift_positions(
    shift_vector,
    sn2_o4_els_and_positions
):
    els, original_positions = sn2_o4_els_and_positions
    shifted_positions = [bnf_utils.shift_coords(np.array(c), shift_vector, 1) for c in original_positions]

    original_map = FractionalMotif.from_elements_and_positions(els, original_positions)
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

def test_can_get_cartesian_coords_after_transform():
    lattice_vecs = [[0,2,1], [1, 0, 0], [1, 0, 2]]
    sb = Superbasis.from_generating_vecs(lattice_vecs)
    motif = FractionalMotif.from_elements_and_positions(["Li", "Li"], [(0.25, 0.25, 0.25), (0.5, 0.5, 0)])

    cartesian_coords = motif.compute_cartesian_coords_in_basis(sb)
    assert (cartesian_coords.positions[0] == np.array([0.5, 0.5, 0.75])).all()
    assert (cartesian_coords.positions[1] == np.array([0.5, 1, 0.5])).all()

    transform = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ])

    transformed_sb = sb.apply_matrix_transform(transform)
    transformed_motif = motif.transform(transform)

    transformed_cart_coords = transformed_motif.compute_cartesian_coords_in_basis(transformed_sb)
    assert (transformed_cart_coords.positions[0] == np.array([0.5, 0.5, 0.75])).all()
    assert (transformed_cart_coords.positions[1] == np.array([0.5, 1, 0.5])).all()

def test_compare_equality():
    motif1 = FractionalMotif.from_elements_and_positions(["Li", "Li"], [(0.25, 0.25, 0.25), (0.5, 0.5, 0)])
    motif2 = FractionalMotif.from_elements_and_positions(["Li", "Li"], [(0.25, 0.25, 0.25), (0.5, 0.5, 0)])
    assert motif1 == motif2

    motif1 = FractionalMotif.from_elements_and_positions(["Li", "Li"], [(0.25, 0.25, 0.25), (0.5, 0.5, 0)])
    motif2 = FractionalMotif.from_elements_and_positions(["Li", "Li"], [(0.5, 0.5, 0), (0.25, 0.25, 0.25)])
    assert motif1 == motif2

    motif1 = FractionalMotif.from_elements_and_positions(["Li", "Si"], [(0.25, 0.25, 0.25), (0.5, 0.5, 0)])
    motif2 = FractionalMotif.from_elements_and_positions(["Li", "Li"], [(0.25, 0.25, 0.25), (0.5, 0.5, 0)])
    assert not motif1 == motif2

    motif1 = FractionalMotif.from_elements_and_positions(["Li", "Si"], [(0.25, 0.25, 0.25), (0.5, 0.5, 0)])
    motif2 = FractionalMotif.from_elements_and_positions(["Li", "Si"], [(0.25, 0.35, 0.25), (0.5, 0.5, 0)])
    assert not motif1 == motif2

    motif1 = FractionalMotif.from_elements_and_positions(["Li", "Si"], [(0.25, 0.35, 0.25), (0.5, 0.5, 0)])
    motif2 = FractionalMotif.from_elements_and_positions(["Li", "Si"], [(0.25, 0.35, 0.25), (0.5, 0.5, 0)])
    assert motif1 == motif2

def test_cartesian_coords_not_changed_by_unimodular():
    lattice = Lattice.orthorhombic(1.0, 2.0, 1.5)
    sb = Superbasis.from_pymatgen_lattice(lattice)
    motif = FractionalMotif.from_elements_and_positions(["Li", "Li"], [(0.25, 0.25, 0.25), (0.5, 0.5, 0)])
    original_cart_coords = motif.compute_cartesian_coords_in_basis(sb)
    original = Structure(sb.generating_vecs(), motif.atoms, motif.positions)

    for perm in sb.compute_vonorms().conorms.permissible_permutations:
        permuted_sb = sb.apply_matrix_transform(perm.matrix.matrix)
        permuted_motif = motif.apply_unimodular(perm.matrix)
        transformed_cart_corods = permuted_motif.compute_cartesian_coords_in_basis(permuted_sb)
        transformed = Structure(permuted_sb.generating_vecs(), permuted_motif.atoms, permuted_motif.positions)

        helpers.assert_identical_by_pdd_distance(original, transformed)

def test_finds_simple_match():
    motif1 = FractionalMotif.from_elements_and_positions(
        ["Li", "Li"],
        [
            (0.25, 0.5, 0.75),
            (0.1, 0.2, 0.3)
        ]
    )

    motif2 = FractionalMotif.from_elements_and_positions(
        ["Li", "Li"],
        [
            (0.1, 0.2, 0.3),
            (0.25, 0.5, 0.75),
        ]
    )

    assert motif1.find_match(motif2)

def test_doesnt_match_different_els():
    motif1 = FractionalMotif.from_elements_and_positions(
        ["Li", "Li"],
        [
            (0.25, 0.5, 0.75),
            (0.1, 0.2, 0.3)
        ]
    )

    motif2 = FractionalMotif.from_elements_and_positions(
        ["Li", "Mg"],
        [
            (0.1, 0.2, 0.3),
            (0.25, 0.5, 0.75),
        ]
    )

    assert not motif1.find_match(motif2)

def test_matches_multiple_els_complex():
    motif1 = FractionalMotif.from_elements_and_positions(
        ["Li", "Mg", "Li", "Mg", "As"],
        [
            (0.25, 0.5, 0.75),
            (0.1, 0.2, 0.3),
            (0.2, 0.5, 0.7),
            (0.1, 0.4, 0.5),
            (0.9, 0.9, 0.9)
        ]
    )

    motif2 = FractionalMotif.from_elements_and_positions(
        ["Li", "Li", "Mg", "As", "Mg"],
        [
            (0.25, 0.5, 0.75),
            (0.2, 0.5, 0.7),
            (0.1, 0.2, 0.3),
            (0.9, 0.9, 0.9),
            (0.1, 0.4, 0.5),
        ]
    )

    assert motif1.find_match(motif2)

    # same els, reordered positions
    motif3 = FractionalMotif.from_elements_and_positions(
        ["Li", "Mg", "Li", "Mg", "As"],
        [
            (0.1, 0.2, 0.3),
            (0.25, 0.5, 0.75),
            (0.1, 0.4, 0.5),
            (0.2, 0.5, 0.7),
            (0.9, 0.9, 0.9)
        ]
    )
    assert not motif1.find_match(motif3)

def test_identifies_inverted_match():
    motif1 = FractionalMotif.from_elements_and_positions(
        ["Li", "Mg"],
        [
            (0.4, 0.6, 0.2),
            (0.8, 0.8, 0.1),
        ]
    )

    motif2 = FractionalMotif.from_elements_and_positions(
        ["Li", "Mg"],
        [
            (0.6, 0.4, 0.8),
            (0.2, 0.2, 0.9),
        ]
    )

    assert motif1.find_inverted_match(motif2)
    assert motif2.find_inverted_match(motif1)

def test_identifies_inversion_symmetry():
    motif1 = FractionalMotif.from_elements_and_positions(
        ["Li"],
        [
            (0.5, 0.5, 0.5),
        ]
    )

    assert motif1.has_inversion_symmetry()

def test_verify_positive_det():
    motif1 = FractionalMotif.from_elements_and_positions(
        ["Li", "Mg"],
        [
            (0.4, 0.6, 0.2),
            (0.8, 0.8, 0.1),
        ]
    )
    neg_det = -np.eye(3)
    with pytest.raises(ValueError) as err:
        motif1.apply_unimodular(MatrixTuple(neg_det))
    assert "matrix with det" in err.value.__repr__()