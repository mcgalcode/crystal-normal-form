from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
from ..crystal_normal_form import CrystalNormalForm

if TYPE_CHECKING:
    from .crystal_explorer import CrystalExplorer

class SearchObjective(ABC):

    @abstractmethod
    def objective_complete(self, explorer: CrystalExplorer) -> bool:
        pass

class LocateAnyTargetStruct(SearchObjective):

    def __init__(self, target_cnfs: list[CrystalNormalForm]):
        self.target_cnfs = target_cnfs
    
    def objective_complete(self, explorer: CrystalExplorer):
        return any([explorer.map.get_point_id(t) is not None for t in self.target_cnfs])
        