from abc import abstractmethod, ABC
from ..crystal_normal_form import CrystalNormalForm

class BaseCalculator(ABC):
    
    @abstractmethod
    def calculate_energy(self, cnf: CrystalNormalForm):
        pass