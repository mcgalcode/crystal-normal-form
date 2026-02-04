import tqdm
from ...calculation.grace import GraceCalculator

def get_energies(cnf_points, prog=False):
    calc = GraceCalculator()
    energies = []
    for cnf in tqdm.tqdm(cnf_points, disable=not prog):
        energies.append(calc.calculate_energy(cnf))
    return energies