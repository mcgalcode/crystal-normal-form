import numpy as np

from ..crystal_normal_form import CrystalNormalForm
from ..lattice import LatticeNormalForm
from .lattice_step import LatticeStepResult

class Neighbor():

    def __init__(self,
                 lnf: LatticeNormalForm | CrystalNormalForm,
                 step_results: list[LatticeStepResult] = None):
        self.point = lnf
        if step_results is not None:
            self.step_results = step_results
        else:
            self.step_results = []
        
    def add_step(self, step_result: LatticeStepResult):
        self.step_results.append(step_result)

    def __eq__(self, other: 'Neighbor'):
        return self.point == other.point
    
    def __hash__(self):
        return self.point.__hash__()