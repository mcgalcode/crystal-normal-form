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


class CalcProvider(ABC):
    """Factory for creating calculators in worker processes.

    CalcProvider instances must be picklable (store only primitive types).
    When called, they create a fresh calculator instance. This enables
    multiprocessing where each worker gets its own calculator.

    Example:
        provider = GraceCalcProvider(model_path="/path/to/model")
        # In worker process:
        calc = provider()  # Creates GraceCalculator with loaded model
    """

    @abstractmethod
    def __call__(self) -> BaseCalculator:
        """Create and return a new calculator instance."""
        pass

    @abstractmethod
    def identifier(self) -> str:
        """Return a string identifying this provider's configuration."""
        pass