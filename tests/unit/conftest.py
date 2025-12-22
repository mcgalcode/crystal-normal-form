import helpers
import pytest
import os

from cnf.lattice import Superbasis
from cnf.motif.atomic_motif import FractionalMotif
from cnf import UnitCell
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
def zr_bcc_manual_unit_cell(zr_bcc_primitive_lattice_vecs):
    sb = Superbasis.from_generating_vecs(zr_bcc_primitive_lattice_vecs)
    motif = FractionalMotif.from_elements_and_positions(["Zr"], [[0, 0, 0]])
    return UnitCell(sb, motif)

@pytest.fixture
def zr_fcc_manual_unit_cell(zr_fcc_primitive_lattice_vecs):
    sb = Superbasis.from_generating_vecs(zr_fcc_primitive_lattice_vecs)
    motif = FractionalMotif.from_elements_and_positions(["Zr"], [[0, 0, 0]])
    return UnitCell(sb, motif)

@pytest.fixture
def zr_hcp_mp():
    return helpers.load_specific_cif("Zr_HCP.cif")

@pytest.fixture
def zr_bcc_mp():
    return helpers.load_specific_cif("Zr_BCC.cif")

@pytest.fixture
def ti_o2_anatase():
    return helpers.load_specific_cif("TiO2_anatase.cif")

@pytest.fixture
def ti_o2_rutile():
    return helpers.load_specific_cif("TiO2_rutile.cif")