from .base_calculator import BaseCalculator

class ConstantCalculator(BaseCalculator):

    def __init__(self, val):
        self.val = val
    
    def calculate_energy(self, _):
        return self.val

    def identifier(self):
        return f"ConstantCalculator(value={self.val})"