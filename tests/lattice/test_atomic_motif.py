import pytest
import numpy as np

from cnf.lattice.atomic_motif import AtomicMotif
from cnf.lattice.superbasis import Superbasis
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

lat = Lattice.from_parameters(1.0, 1.5, 1.25, 50, 70, 120)
struct = Structure(lat, ["Li", "Li"], [(0.25, 0.25, 0.25), (0.5, 0.5, 0)])

def test_can_get_cartesian_coords_after_transform():
    lattice_vecs = [[0,2,1], [1, 0, 0], [1, 0, 2]]
    sb = Superbasis.from_generating_vecs(lattice_vecs)
    motif = AtomicMotif([(0.25, 0.25, 0.25), (0.5, 0.5, 0)], ["Li", "Li"])

    cartesian_coords = motif.compute_cartesian_coords_in_basis(sb)
    
    assert (cartesian_coords[0] == np.array([0.5, 0.5, 0.75])).all()
    assert (cartesian_coords[1] == np.array([0.5, 1, 0.5])).all()

    transform = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ])

    transformed_sb = sb.apply_matrix_transform(transform)
    transformed_motif = motif.transform(transform)

    transformed_cart_coords = transformed_motif.compute_cartesian_coords_in_basis(transformed_sb)
    assert (transformed_cart_coords[0] == np.array([0.5, 0.5, 0.75])).all()
    assert (transformed_cart_coords[1] == np.array([0.5, 1, 0.5])).all()