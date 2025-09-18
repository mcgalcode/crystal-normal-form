import numpy as np

from .gamma_matrices import GammaMatrixTuple

def transform_lattice_vecs(lvecs: np.ndarray, gmat: GammaMatrixTuple):
    """Converts lattice vecs (as rows) to supercell lattice vecs
    according to the provided gamma matrix

    Parameters
    ----------
    lvecs : np.array
        The lattice vecs for which a supercell should be generated
        using gmat
    """
    return (lvecs.T @ gmat.matrix).T

# def recover_gmat_from_supercell_and_unit_cell(supercellvecs: np.ndarray, unit_cell_vecs: np.ndarray):


def transform_basis_position(basis_position: np.ndarray, gmat: GammaMatrixTuple):
    return gmat.inverse() @ basis_position

