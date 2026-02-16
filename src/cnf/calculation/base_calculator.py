from abc import abstractmethod, ABC
from ..crystal_normal_form import CrystalNormalForm

class BaseCalculator(ABC):

    @abstractmethod
    def calculate_energy(self, cnf: CrystalNormalForm):
        pass

    def calculate_energies_batch(self, cnfs: list[CrystalNormalForm]) -> list[float]:
        """Calculate energies for multiple CNFs. Override for batched implementations."""
        return [self.calculate_energy(cnf) for cnf in cnfs]

    @abstractmethod
    def identifier(self) -> str:
        pass