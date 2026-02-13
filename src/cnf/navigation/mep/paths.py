"""
Path alignment and resampling utilities for crystal structure trajectories.

Provides unimodular matrix search + Hungarian atom assignment for aligning
sequences of pymatgen Structures, plus SSNEB-distance-aware resampling.
"""

import numpy as np
import tqdm
from scipy.optimize import linear_sum_assignment

from pymatgen.core import Structure, Lattice

from cnf.linalg.unimodular import load_unimodular
from cnf.crystal_normal_form import CrystalNormalForm


_UNIMODULAR_MATRICES = None

def _get_unimodular_matrices():
    global _UNIMODULAR_MATRICES
    if _UNIMODULAR_MATRICES is None:
        _UNIMODULAR_MATRICES = [m.matrix for m in load_unimodular("unimodular.json")]
    return _UNIMODULAR_MATRICES


def hungarian_atom_assignment(ref_frac, cand_frac, ref_cell, species):
    """Find optimal atom permutation via Hungarian algorithm, within same-species groups.

    Args:
        ref_frac: Reference fractional coordinates, shape (n_atoms, 3).
        cand_frac: Candidate fractional coordinates, shape (n_atoms, 3).
        ref_cell: Reference cell matrix (3x3), used for Cartesian cost.
        species: List of species strings, length n_atoms.

    Returns:
        Tuple of (aligned_frac, total_cost) where aligned_frac has PBC-unwrapped
        coordinates relative to ref_frac.
    """
    n = len(species)
    perm = list(range(n))

    species_groups = {}
    for i, sp in enumerate(species):
        species_groups.setdefault(sp, []).append(i)

    total_cost = 0.0
    for sp, indices in species_groups.items():
        ref_sub = ref_frac[indices]
        cand_sub = cand_frac[indices]
        m = len(indices)
        cost = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                diff = cand_sub[j] - ref_sub[i]
                diff = diff - np.round(diff)
                cart = diff @ ref_cell
                cost[i, j] = np.dot(cart, cart)
        row_ind, col_ind = linear_sum_assignment(cost)
        total_cost += cost[row_ind, col_ind].sum()
        for ri, ci in zip(row_ind, col_ind):
            perm[indices[ri]] = indices[ci]

    permuted_frac = cand_frac[perm]
    diff = permuted_frac - ref_frac
    diff = diff - np.round(diff)
    aligned_frac = ref_frac + diff
    return aligned_frac, total_cost


def align_structure_to_reference(ref_struct, curr_struct):
    """Align curr_struct to ref_struct using unimodular matrix search + Hungarian atom assignment.

    Searches over all precomputed unimodular matrices to find the lattice
    orientation that minimizes cell difference + atom displacement cost,
    then applies Hungarian assignment within each species group.

    Args:
        ref_struct: Reference pymatgen Structure.
        curr_struct: Structure to align.

    Returns:
        A new pymatgen Structure aligned to the reference.
    """
    ref_cell = ref_struct.lattice.matrix
    ref_frac = ref_struct.frac_coords
    curr_cell = curr_struct.lattice.matrix
    curr_frac = curr_struct.frac_coords
    species = [str(s) for s in curr_struct.species]

    best_score = float('inf')
    best_cell = None
    best_frac = None

    for U in _get_unimodular_matrices():
        candidate_cell = U @ curr_cell
        cell_diff_sq = np.sum((candidate_cell - ref_cell) ** 2)
        if cell_diff_sq >= best_score:
            continue

        U_inv = np.linalg.inv(U)
        candidate_frac = curr_frac @ U_inv

        aligned_frac, atom_cost = hungarian_atom_assignment(
            ref_frac, candidate_frac, ref_cell, species
        )

        score = cell_diff_sq + atom_cost
        if score < best_score:
            best_score = score
            best_cell = candidate_cell
            best_frac = aligned_frac

    return Structure(Lattice(best_cell), species, best_frac, coords_are_cartesian=False)


def align_path(structs, verbose=True):
    """Sequentially align a list of structures, each to its predecessor.

    Args:
        structs: List of pymatgen Structures.
        verbose: Show progress bar.

    Returns:
        List of aligned pymatgen Structures.
    """
    aligned = [structs[0]]
    iterator = structs[1:]
    if verbose:
        iterator = tqdm.tqdm(iterator, desc="Aligning structures")
    for s in iterator:
        aligned.append(align_structure_to_reference(aligned[-1], s))
    return aligned


def align_cnf_path(cnfs, verbose=True):
    """Reconstruct and align a list of CrystalNormalForm objects.

    Args:
        cnfs: List of CrystalNormalForm instances.
        verbose: Show progress bar.

    Returns:
        List of aligned pymatgen Structures.
    """
    structs = [cnf.reconstruct() for cnf in cnfs]
    return align_path(structs, verbose=verbose)

def resample_path_by_distance(structs, num_images, weight=1.0):
    """Resample an aligned path so images are equally spaced in SSNEB distance.

    Computes the cumulative SSNEB distance along the path, then picks
    num_images points at uniform distance intervals, interpolating linearly
    between the nearest bracketing structures.

    Args:
        structs: List of aligned pymatgen Structures.
        num_images: Number of output images (including endpoints).
        weight: Relative weight of cell vs atomic degrees of freedom.

    Returns:
        List of num_images pymatgen Structures, equally spaced in SSNEB
        distance metric.
    """
    from .ssneb import compute_ssneb_distances
    distances = compute_ssneb_distances(structs, weight=weight)
    cumulative = np.zeros(len(structs))
    for i in range(len(distances)):
        cumulative[i + 1] = cumulative[i] + distances[i]
    total_dist = cumulative[-1]

    # Target distances for uniform spacing
    target_dists = np.linspace(0, total_dist, num_images)

    resampled = []
    for td in target_dists:
        # Find bracketing indices
        idx = np.searchsorted(cumulative, td, side='right') - 1
        idx = max(0, min(idx, len(structs) - 2))

        seg_start = cumulative[idx]
        seg_end = cumulative[idx + 1]
        seg_len = seg_end - seg_start

        if seg_len < 1e-12:
            resampled.append(structs[idx])
            continue

        t = (td - seg_start) / seg_len

        # Linear interpolation of cell and fractional coordinates
        s1 = structs[idx]
        s2 = structs[idx + 1]
        cell_interp = (1 - t) * s1.lattice.matrix + t * s2.lattice.matrix
        frac_interp = (1 - t) * s1.frac_coords + t * s2.frac_coords
        species = [str(sp) for sp in s1.species]

        resampled.append(Structure(
            Lattice(cell_interp), species, frac_interp,
            coords_are_cartesian=False
        ))

    return resampled


def make_uniform_path(structs, weight=1.0):
    """Resample a path to the same number of frames but with uniform SSNEB spacing.

    Equivalent to resample_path_by_distance(structs, len(structs)), but
    returned as a list that can be cheaply subsampled with integer indexing.

    Args:
        structs: List of aligned pymatgen Structures.
        weight: Relative weight of cell vs atomic degrees of freedom.

    Returns:
        List of len(structs) Structures, equally spaced in SSNEB distance.
    """
    return resample_path_by_distance(structs, len(structs), weight=weight)


def subsample_uniform_path(uniform_structs, num_images):
    """Pick num_images frames from a uniformly-spaced path by integer indexing.

    Because uniform_structs is already evenly spaced in SSNEB distance,
    the subsampled frames are also approximately evenly spaced.

    Args:
        uniform_structs: List of uniformly-spaced Structures (from make_uniform_path).
        num_images: Number of output images (including endpoints).

    Returns:
        List of num_images Structures.
    """
    n = len(uniform_structs)
    indices = np.round(np.linspace(0, n - 1, num_images)).astype(int)
    return [uniform_structs[i] for i in indices]
