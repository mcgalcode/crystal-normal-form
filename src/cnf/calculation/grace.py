import logging
import math
from pathlib import Path

from tensorpotential.calculator import TPCalculator
from tensorpotential.calculator.foundation_models import grace_fm, GRACEModels

from ..crystal_normal_form import CrystalNormalForm
from ..navigation import find_neighbors
from .base_calculator import BaseCalculator

logger = logging.getLogger(__name__)

DEFAULT_MODEL = GRACEModels.GRACE_FS_OAM

class GraceCalculator(BaseCalculator):

    def __init__(self, model_string: str = DEFAULT_MODEL, model_path: str = None):
        if model_path is not None:
            self.model_string = str(model_path)
            logger.info(f"Loading GRACE model from path: {model_path}")
            self._calc = TPCalculator(model_path)
        else:
            self.model_string = model_string
            logger.info(f"Loading GRACE foundation model: {model_string}")
            self._calc = grace_fm(model_string)

    def calculate_energy(self, cnf: CrystalNormalForm) -> float:
        atoms = cnf.reconstruct().to_ase_atoms()
        self._calc.calculate(atoms, properties=['energy'])
        return self._calc.results['energy']

    def relax(self, cnf: CrystalNormalForm, max_iters = None) -> tuple[CrystalNormalForm, float, int]:
        min_e = self.calculate_energy(cnf)
        min_cnf = cnf
        num_iters = 0
        if max_iters is None:
            max_iters = math.inf

        while num_iters < max_iters:
            print(f"Energy: {min_e}")
            nbs = find_neighbors(min_cnf)
            nb_es = sorted([(self.calculate_energy(n), n) for n in nbs], key=lambda x: x[0])
            min_nb_e, min_nb = nb_es[0]
            if min_nb_e < min_e:
                min_e = min_nb_e
                min_cnf = min_nb
            else:
                break
            num_iters += 1
        return min_cnf, min_e, num_iters

    def identifier(self):
        return f"GraceMLIPCalculator(model={self.model_string})"
