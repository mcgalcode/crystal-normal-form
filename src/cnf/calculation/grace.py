from tensorpotential.calculator.foundation_models import grace_fm, GRACEModels
from ..crystal_normal_form import CrystalNormalForm
from ase import Atoms
from .base_calculator import BaseCalculator


ASE_CALC = grace_fm(GRACEModels.GRACE_1L_OAM) # for better code completion

class GraceCalculator(BaseCalculator):

    def __init__(self):
        self._calc = ASE_CALC

    def calculate_energy(self, cnf: CrystalNormalForm):
        atoms = cnf.reconstruct().to_ase_atoms()
        self._calc.calculate(atoms, properties = ['energy'])
        return self._calc.results['energy']
    