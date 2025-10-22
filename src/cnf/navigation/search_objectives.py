from abc import ABC, abstractmethod

from ..crystal_normal_form import CrystalNormalForm


class SearchObjective(ABC):

    @abstractmethod
    def objective_complete(self, explorer) -> bool:
        pass

class LocateAnyTargetStruct(SearchObjective):

    def __init__(self, target_cnfs: list[CrystalNormalForm]):
        self.target_cnfs = target_cnfs
        self.located_endpt = None
    
    def objective_complete(self, explorer):
        for cnf in self.target_cnfs:
            if cnf in explorer.map:
                self.located_endpt = cnf       
                return True