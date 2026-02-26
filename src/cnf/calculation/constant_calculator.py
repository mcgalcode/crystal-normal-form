from .base_calculator import BaseCalculator, CalcProvider


class ConstantCalcProvider(CalcProvider):
    """Picklable factory for ConstantCalculator."""

    def __init__(self, val: float):
        self.val = val

    def __call__(self) -> "ConstantCalculator":
        return ConstantCalculator(self.val)

    def identifier(self) -> str:
        return f"ConstantCalcProvider(value={self.val})"


class ConstantCalculator(BaseCalculator):

    def __init__(self, val):
        self.val = val

    def calculate_energy(self, _):
        return self.val

    def identifier(self):
        return f"ConstantCalculator(value={self.val})"