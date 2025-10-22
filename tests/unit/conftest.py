import helpers
import pytest
import os

import numpy as np
from pathlib import Path
from pymatgen.core.structure import Structure


@pytest.fixture(scope='package')
def zr_bcc_primitive_lattice_vecs():
    zr_lattice_param = 3.52
    bcc_primitive_vecs = np.array([
        [-1, 1, 1],
        [1, -1, 1],
        [1, 1, -1],
    ]) * zr_lattice_param / 2
    return bcc_primitive_vecs

@pytest.fixture(scope='package')
def zr_fcc_primitive_lattice_vecs():
    zr_lattice_param = 4.46
    # fcc_primitive_vecs = np.array([
    #     [1, 1, -1],
    #     [-1, 1, 0],
    #     [0, 0, 1]
    # ]) * zr_lattice_param / 2
    fcc_primitive_vecs = np.array([
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1]
    ]) * zr_lattice_param / 2
    return fcc_primitive_vecs


@pytest.fixture
def zr_hcp():
    return helpers.load_specific_cif("Zr_HCP.cif")

@pytest.fixture
def zr_bcc():
    return helpers.load_specific_cif("Zr_BCC.cif")