from tensorpotential.calculator.foundation_models import grace_fm, GRACEModels
from ..crystal_normal_form import CrystalNormalForm
from ..navigation.neighbor_finder import NeighborFinder
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
    
    def relax(self, cnf: CrystalNormalForm):
        min_e = self.calculate_energy(cnf)
        min_cnf = cnf
        num_iters = 0
        while True:
            print(f"Energy: {min_e}")
            nbs = NeighborFinder(min_cnf).find_neighbors()
            nb_es = sorted([(self.calculate_energy(n), n) for n in nbs])
            min_nb_e, min_nb = nb_es[0]
            if min_nb_e < min_e:
                min_e = min_nb_e
                min_cnf = min_nb
            else:
                break
            num_iters += 1
        return min_cnf, min_e, num_iters

            
    