"""Path energy evaluation utilities."""

from cnf import CrystalNormalForm


def evaluate_path_energies(path_tuples, elements, xi, delta, calc, cache, verbose=False):
    """Evaluate energies along a path, using and populating the shared cache."""
    n_pts = len(path_tuples)
    n_cached = 0
    n_computed = 0

    if verbose:
        print(f"      Evaluating energies for {n_pts} points...", flush=True)

    energies = []
    for i, pt in enumerate(path_tuples):
        # Convert to tuple for cache key (astar_rust returns lists)
        pt_tup = tuple(pt) if isinstance(pt, list) else pt
        if pt_tup not in cache:
            cnf = CrystalNormalForm.from_tuple(pt_tup, elements, xi, delta)
            cache[pt_tup] = calc.calculate_energy(cnf)
            n_computed += 1
            if verbose and n_computed % 50 == 0:
                print(f"        computed {n_computed}/{n_pts - n_cached} energies...", flush=True)
        else:
            n_cached += 1
        energies.append(cache[pt_tup])

    if verbose:
        print(f"      Done: {n_computed} computed, {n_cached} from cache", flush=True)

    return energies


def path_barrier(energies):
    """Barrier = max energy along path."""
    return max(energies)
