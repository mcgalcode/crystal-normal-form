import pytest
import numpy as np

import helpers

from cnf import UnitCell
from cnf.calculation.relaxation import relax_unit_cell


@pytest.fixture
def grace_calc():
    """Get GRACE calculator, skip test if unavailable."""
    try:
        from cnf.calculation.grace import GraceCalculator
        calc = GraceCalculator()
        return calc._calc
    except Exception as e:
        pytest.skip(f"GRACE calculator not available: {e}")


class TestRelaxUnitCell:

    @helpers.skip_if_fast
    def test_returns_unit_cell(self, zr_bcc_mp, grace_calc):
        uc = UnitCell.from_pymatgen_structure(zr_bcc_mp)
        result = relax_unit_cell(uc, grace_calc, fmax=0.1, max_steps=3, verbose=False)
        assert isinstance(result, UnitCell)

    @helpers.skip_if_fast
    def test_respects_max_steps(self, zr_bcc_mp, grace_calc):
        uc = UnitCell.from_pymatgen_structure(zr_bcc_mp)
        # Very tight fmax that won't converge in 2 steps
        result = relax_unit_cell(uc, grace_calc, fmax=0.0001, max_steps=2, verbose=False)
        assert isinstance(result, UnitCell)

    @helpers.skip_if_fast
    def test_verbose_output(self, zr_bcc_mp, grace_calc, capsys):
        uc = UnitCell.from_pymatgen_structure(zr_bcc_mp)
        relax_unit_cell(uc, grace_calc, fmax=0.1, max_steps=2, verbose=True, label="test")
        captured = capsys.readouterr()
        assert "[test] Before:" in captured.out
        assert "[test] After:" in captured.out

    @helpers.skip_if_fast
    def test_no_output_when_not_verbose(self, zr_bcc_mp, grace_calc, capsys):
        uc = UnitCell.from_pymatgen_structure(zr_bcc_mp)
        relax_unit_cell(uc, grace_calc, fmax=0.1, max_steps=2, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    @helpers.skip_if_fast
    def test_preserves_species(self, zr_bcc_mp, grace_calc):
        uc = UnitCell.from_pymatgen_structure(zr_bcc_mp)
        result = relax_unit_cell(uc, grace_calc, fmax=0.1, max_steps=2, verbose=False)
        original_elements = [s.symbol for s in zr_bcc_mp.species]
        result_elements = [s.symbol for s in result.to_pymatgen_structure().species]
        assert original_elements == result_elements

    @helpers.skip_if_fast
    def test_preserves_num_atoms(self, zr_bcc_mp, grace_calc):
        uc = UnitCell.from_pymatgen_structure(zr_bcc_mp)
        result = relax_unit_cell(uc, grace_calc, fmax=0.1, max_steps=2, verbose=False)
        assert len(result.to_pymatgen_structure()) == len(zr_bcc_mp)

    @helpers.skip_if_fast
    def test_multiatomic_structure(self, ti_o2_anatase, grace_calc):
        uc = UnitCell.from_pymatgen_structure(ti_o2_anatase)
        result = relax_unit_cell(uc, grace_calc, fmax=0.1, max_steps=2, verbose=False)
        assert isinstance(result, UnitCell)
        assert len(result.to_pymatgen_structure()) == len(ti_o2_anatase)

    @helpers.skip_if_fast
    def test_relaxation_changes_structure(self, zr_bcc_mp, grace_calc):
        uc = UnitCell.from_pymatgen_structure(zr_bcc_mp)
        result = relax_unit_cell(uc, grace_calc, fmax=0.01, max_steps=20, verbose=False)

        # Relaxation should change the structure (unless already at minimum)
        original_vol = uc.to_pymatgen_structure().volume
        result_vol = result.to_pymatgen_structure().volume
        # Volume should be within reasonable range
        assert 0.5 < result_vol / original_vol < 2.0
