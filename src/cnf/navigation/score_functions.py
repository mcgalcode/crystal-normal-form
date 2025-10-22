from ..crystal_normal_form import CrystalNormalForm
from cnf.utils.pdd import pdd
from abc import ABC, abstractmethod
from pymatgen.core.structure import Structure


class ScoreFunction(ABC):

    @abstractmethod
    def score(self, pt: CrystalNormalForm) -> float:
        pass

class PDDScorer(ScoreFunction):

    def __init__(self, target_structs: list[Structure]):
        self.target_structs = target_structs
    
    def score(self, pt: CrystalNormalForm) -> float:
        pt_struct = pt.reconstruct()
        dist = min([pdd(pt_struct, target, k=100) for target in self.target_structs])
        return dist