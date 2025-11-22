from cnf.calculation.grace import GraceCalculator
from cnf import CrystalNormalForm

def test_can_load_calc(zr_bcc_mp):
    cnf = CrystalNormalForm.from_pmg_struct(zr_bcc_mp, xi=1.5, delta=1)
    e = GraceCalculator().calculate_energy(cnf)
    assert e is not None
    assert isinstance(e, float)

def test_relax(zr_bcc_mp):
    cnf = CrystalNormalForm.from_pmg_struct(zr_bcc_mp, xi=1.5, delta=10)
    min_cnf, min_e, num_iters = GraceCalculator().relax(cnf)
    assert min_cnf != cnf
    assert min_e < GraceCalculator().calculate_energy(cnf)
    assert num_iters > 0