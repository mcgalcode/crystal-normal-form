"""Path energy evaluation utilities."""

from cnf import CrystalNormalForm


def evaluate_path_energies(path_tuples, elements, xi, delta, calc, cache):
    """Evaluate energies along a path, using and populating the shared cache."""
    energies = []
    for pt_tup in path_tuples:
        if pt_tup not in cache:
            cnf = CrystalNormalForm.from_tuple(pt_tup, elements, xi, delta)
            cache[pt_tup] = calc.calculate_energy(cnf)
        energies.append(cache[pt_tup])
    return energies


def path_barrier(energies):
    """Barrier = max energy along path."""
    return max(energies)
