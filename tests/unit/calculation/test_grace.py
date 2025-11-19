from cnf.calculation.grace import GraceCalculator
from cnf import CrystalNormalForm

def test_can_load_calc(zr_bcc_mp):
    cnf = CrystalNormalForm.from_pmg_struct(zr_bcc_mp, xi=1.5, delta=1)
    e = GraceCalculator().calculate_energy(cnf)
    assert e is not None
    assert isinstance(e, float)