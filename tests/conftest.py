import pytest

def pytest_addoption(parser):
    """Adds the --run-debug command-line option to pytest."""
    parser.addoption(
        "--run-debug", action="store_true", default=False, help="run debug tests"
    )

def pytest_configure(config):
    """Registers the 'debug' marker to avoid warnings."""
    config.addinivalue_line("markers", "debug: mark test to run only with --run-debug")

def pytest_runtest_setup(item):
    """
    Hook to skip tests marked with 'debug' unless --run-debug is given.
    This runs before the test setup.
    """
    if "debug" in item.keywords and not item.config.getoption("--run-debug"):
        pytest.skip("skipping debug test; enable with --run-debug")

def pytest_report_teststatus(report):
    """
    Hook to change the reporting output for skipped debug tests.
    It changes the character from 's' (skip) to 'd' (debug).
    """
    # Check if the test was skipped during the setup phase
    if report.when == "setup" and report.skipped:
        # Check if the 'debug' marker was on the test
        if "debug" in report.keywords:
            # Return a tuple: (category, short_letter, verbose_word)
            return "skipped", "d", "DEBUG"

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