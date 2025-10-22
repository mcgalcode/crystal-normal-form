from ..crystal_normal_form import CrystalNormalForm
from cnf.utils.pdd import pdd
from abc import ABC, abstractmethod
from ..unit_cell import UnitCell

class ScoreFunction(ABC):

    @abstractmethod
    def score(self, pt: CrystalNormalForm) -> float:
        pass

class PDDScorer(ScoreFunction):

    def __init__(self, target_structs: list[UnitCell]):
        self.target_structs = [t.to_pymatgen_structure() for t in target_structs]
    
    def score(self, pt: CrystalNormalForm) -> float:
        pt_struct = pt.reconstruct()
        dist = min([pdd(pt_struct, target, k=100) for target in self.target_structs])
        return dist