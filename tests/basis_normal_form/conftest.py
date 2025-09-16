import pytest

from cnf.motif.atomic_motif import FractionalMotif

SN_2_O_4_POSITIONS = [
    (0, 0.3, 0.3),
    (0.5, 0.8, 0.2),
    (0.5, 0.5, 0.5),
    (0, 0.7, 0.7),
    (0, 0, 0),
    (0.5, 0.2, 0.8)
]

@pytest.fixture
def sn2_o4_els_and_positions():
    elements = ["O", "O", "Sn", "O", "Sn", "O"]
    return elements, SN_2_O_4_POSITIONS

@pytest.fixture
def sn2_o4_motif(sn2_o4_els_and_positions):
    els, positions = sn2_o4_els_and_positions
    return FractionalMotif.from_elements_and_positions(els, positions)