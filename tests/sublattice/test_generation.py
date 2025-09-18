import pytest
import numpy as np

from cnf.sublattice.generation import transform_lattice_vecs, MotifTranslationSet, transform_basis_position
from cnf.sublattice.gamma_matrices import GammaMatrixGroup, GammaMatrixTuple
from cnf.motif import FractionalMotif

from pymatgen.core.lattice import Lattice

@pytest.fixture
def various_lattices():
    return [
        Lattice.cubic(1.0),
        Lattice.cubic(1.5),
        Lattice.orthorhombic(1.2, 0.3, 1.4),
        Lattice.hexagonal(2.1, 3.1),
        Lattice.hexagonal(3.1, 2.1),
        Lattice.monoclinic(1.4, 2.1, 0.9, 36),
        Lattice.from_parameters(0.3, 1.2, 0.8, 56, 121, 95),
        Lattice.from_parameters(2.0, 1.0, 1.4, 45, 55, 65),
    ]

def test_can_transform_lattice_generators():
    generators = np.array([
        [1, 2, 0],
        [0, 2, 0],
        [0, 1, 3],
    ])

    transform = GammaMatrixTuple(np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 1],
    ]))
    transformed_generators = transform_lattice_vecs(generators, transform)
    assert np.all(np.isclose(transformed_generators[0], [2, 4, 0]))
    assert np.all(np.isclose(transformed_generators[1], [0, 4, 0]))
    assert np.all(np.isclose(transformed_generators[2], [0, 1, 3]))

def test_generating_matrices_yield_correct_volume_change(various_lattices):
    for N in range(1, 10):
        mat_group = GammaMatrixGroup.for_index(N)
        for l in various_lattices:            
            for member in mat_group.matrices:
                transformed = transform_lattice_vecs(l.matrix, member)
                assert np.isclose(np.linalg.det(transformed) / l.volume, N)


def test_generates_correct_motif_translations():
    for N in range(1, 10):
        mat_group = GammaMatrixGroup.for_index(N)
        for m in mat_group.matrices:
            vectors = MotifTranslationSet.from_gamma_matrix(m)
            assert len(vectors) == N - 1

            for v in vectors.vecs:
                assert np.all(v < 1.0)
                assert np.all(v >= 0)

def test_can_transform_basis_position():
    mat = GammaMatrixTuple(np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
    ]))
    pos = np.array((0.5, 0.5, 0.5))
    transformed_pos = transform_basis_position(pos, mat)
    assert np.all(np.isclose(transformed_pos, np.array((0.25, 0.25, 0.25))))

    mat = GammaMatrixTuple(np.array([
        [2, 0, 0],
        [0, 1, 0],
        [0, 0, 2],
    ]))
    pos = np.array((0.5, 0.5, 0.5))
    transformed_pos = transform_basis_position(pos, mat)
    assert np.all(np.isclose(transformed_pos, np.array((0.25, 0.5, 0.25))))

def test_motif_translations_for_simple_gamma():
    mat = np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
    ])
    translations = MotifTranslationSet.from_gamma_matrix(GammaMatrixTuple(mat))
    pos = np.array((0.5, 0.5, 0.5))
    images = translations.apply_to_coord(pos)
    images = set([tuple(i) for i in images])
    assert len(images) == len(translations) + 1

    assert (0.25, 0.25, 0.25) in images
    assert (0.25, 0.25, 0.75) in images
    assert (0.25, 0.75, 0.25) in images
    assert (0.75, 0.25, 0.25) in images
    assert (0.25, 0.75, 0.75) in images
    assert (0.75, 0.25, 0.75) in images
    assert (0.75, 0.75, 0.25) in images
    assert (0.75, 0.75, 0.75) in images

def test_motif_translations_for_simple_motif_1():
    motif = FractionalMotif({
        "Li": [[0.5, 0.5, 0.5]],
        "C": [[0.25, 0.5, 0.75]],
    })
    
    mat = np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
    ])
    translations = MotifTranslationSet.from_gamma_matrix(GammaMatrixTuple(mat))
    extended_motif = translations.apply_to_motif(motif)
    assert len(extended_motif) == 8 * len(motif)

    new_li_positions = [tuple(pos) for pos in extended_motif.get_element_positions("Li")]
    assert (0.25, 0.25, 0.25) in new_li_positions
    assert (0.25, 0.25, 0.75) in new_li_positions
    assert (0.25, 0.75, 0.25) in new_li_positions
    assert (0.75, 0.25, 0.25) in new_li_positions
    assert (0.25, 0.75, 0.75) in new_li_positions
    assert (0.75, 0.25, 0.75) in new_li_positions
    assert (0.75, 0.75, 0.25) in new_li_positions
    assert (0.75, 0.75, 0.75) in new_li_positions

    new_c_positions = [tuple(pos) for pos in extended_motif.get_element_positions("C")]
    assert (0.125, 0.25, 0.375) in new_c_positions
    assert (0.125, 0.25, 0.875) in new_c_positions
    assert (0.125, 0.75, 0.375) in new_c_positions
    assert (0.625, 0.25, 0.375) in new_c_positions
    assert (0.125, 0.75, 0.875) in new_c_positions
    assert (0.625, 0.25, 0.875) in new_c_positions
    assert (0.625, 0.75, 0.375) in new_c_positions
    assert (0.625, 0.75, 0.875) in new_c_positions

def test_motif_translations_for_simple_motif_2():
    motif = FractionalMotif({
        "Li": [[0.5, 0.5, 0.5]],
        "C": [[0.25, 0.5, 0.75]],
    })
    
    mat = np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 1],
    ])
    translations = MotifTranslationSet.from_gamma_matrix(GammaMatrixTuple(mat))
    extended_motif = translations.apply_to_motif(motif)
    assert len(extended_motif) == 4 * len(motif)

    new_li_positions = [tuple(pos) for pos in extended_motif.get_element_positions("Li")]
    assert (0.25, 0.25, 0.5) in new_li_positions
    assert (0.25, 0.75, 0.5) in new_li_positions
    assert (0.75, 0.25, 0.5) in new_li_positions
    assert (0.75, 0.75, 0.5) in new_li_positions

    new_c_positions = [tuple(pos) for pos in extended_motif.get_element_positions("C")]
    assert (0.125, 0.25, 0.75) in new_c_positions
    assert (0.125, 0.75, 0.75) in new_c_positions
    assert (0.625, 0.25, 0.75) in new_c_positions
    assert (0.625, 0.75, 0.75) in new_c_positions
