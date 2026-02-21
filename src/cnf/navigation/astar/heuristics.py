
import numpy as np

from typing import List

from cnf import CrystalNormalForm
from ...utils.pdd import pdd_for_cnfs, pdd_amd_for_cnfs
from ...lattice.permutations import VonormPermutation, CONORM_PERMUTATION_TO_VONORM_PERMUTATION, ZERO_CONORM_SETS_TO_PERMUTATIONS_TO_UNIMOD_MATS
from ...lattice.vonorm_unimodular import VonormPermutationMatrix
from ...lattice.voronoi.vonorm_list import VonormList, VONORM_TO_DOT_PRODUCTS
from ...motif.mnf_constructor import sort_motif_coord_arr, move_coords_into_bounds, get_mnf_strs_from_coord_mats

def pdd_heuristic(cnf: tuple, goals: list[CrystalNormalForm]) -> float:
    xi = goals[0].xi
    delta = goals[0].delta
    els = goals[0].elements

    pt = CrystalNormalForm.from_tuple(cnf, els, xi, delta)
    dists = [pdd_for_cnfs(pt, g, k=20) for g in goals]
    return (min(dists) * 100) ** 2

def pdd_and_manhattan(cnf: tuple, goals: list[CrystalNormalForm]) -> float:
    xi = goals[0].xi
    delta = goals[0].delta
    els = goals[0].elements

    pt = CrystalNormalForm.from_tuple(cnf, els, xi, delta)
    dists = [pdd_for_cnfs(pt, g, k=20) for g in goals]
    return 10000 * min(dists) + manhattan_distance(cnf, goals)

def pdd_amd_heuristic(cnf: tuple, goals: list[CrystalNormalForm]) -> float:
    xi = goals[0].xi
    delta = goals[0].delta
    els = goals[0].elements

    pt = CrystalNormalForm.from_tuple(cnf, els, xi, delta)
    dists = [pdd_amd_for_cnfs(pt, g, k=20) for g in goals]
    return (min(dists) * 100) ** 3 


def manhattan_distance(cnf: tuple, goals: list[CrystalNormalForm]) -> float:
    manhattan_dist = float('inf')
    current_coords = np.array(cnf)

    for goal in goals:
        goal_coords = np.array(goal.coords)
        curr_dist = np.sum(np.abs(current_coords - goal_coords))
        manhattan_dist = min(manhattan_dist, curr_dist)

    return manhattan_dist * 2

def manhattan_dist_cnfs(cnf: CrystalNormalForm, cnf2: CrystalNormalForm) -> float:
    return np.sum(np.abs(np.array(cnf.coords) - np.array(cnf2.coords)))

def squared_euclidean_heuristic(cnf: tuple, goals: List[CrystalNormalForm]) -> float:

    min_dist_sq = float('inf')

    for goal in goals:
        goal_coords = np.array(goal.coords)
        dist_sq = np.sum((np.array(cnf) - goal_coords) ** 2)
        min_dist_sq = min(min_dist_sq, dist_sq)

    return min_dist_sq


# ---------------------------------------------------------------------------
# Unimodular-aware Manhattan heuristic
# ---------------------------------------------------------------------------
#
# The plain Manhattan heuristic ignores the fact that the canonical ordering
# of vonorms (and motif coordinates) can change across Voronoi-class
# boundaries.  Two CNFs that are one graph step apart may look very far
# apart in Manhattan distance because they sit in different canonical
# orderings.
#
# This heuristic pre-computes all 168 vonorm-permutation variants of each
# goal CNF (permuted vonorms + unimodular-transformed & re-sorted motif).
# During search it returns the *minimum* Manhattan distance over all
# variants, giving a much tighter estimate of the true graph distance.
# ---------------------------------------------------------------------------

def _precompute_goal_variants(goal: CrystalNormalForm):
    """Pre-compute all vonorm-permuted + motif-transformed variants of a goal.

    For each of the 168 vonorm permutations we:
      1. Permute the 7 vonorms.
      2. Apply the corresponding inverse unimodular matrix to the motif
         coordinate matrix (integer arithmetic, mod delta).
      3. Try every valid origin shift (one per atom of the first element).
      4. Sort atoms lexicographically by (element, x, y, z).
      5. Flatten to the CNF tuple layout: vonorms ++ motif_coords.

    Returns:
        np.ndarray of shape (N_variants, tuple_len) containing all goal
        representations, ready for vectorised Manhattan-distance computation.
    """
    vonorms = np.array(goal.coords[:7])
    delta = goal.delta
    elements = goal.elements

    # Reconstruct motif coord matrix (3, N-1) from the MNF tuple
    mnf_tuple = goal.coords[7:]
    n_stored = len(mnf_tuple) // 3   # atoms stored (excludes origin)
    n_atoms = n_stored + 1
    stored_coords = np.array(mnf_tuple, dtype=np.int64).reshape(n_stored, 3)  # (N-1, 3)
    # Full coord matrix including origin at (0,0,0)
    full_coords = np.vstack([np.zeros((1, 3), dtype=np.int64), stored_coords])  # (N, 3)
    coord_matrix = full_coords.T  # (3, N)

    # Atom labels for sorting (group by element)
    atom_labels = []
    el_num = 0
    prev_el = elements[0]
    for el in elements:
        if el != prev_el:
            el_num += 1
            prev_el = el
        atom_labels.append(el_num)
    atom_labels = np.array(atom_labels)

    # Number of origin-shift candidates (atoms of the first element)
    num_origin_atoms = sum(1 for el in elements if el == elements[0])

    all_perms = VonormPermutation.all_vonorm_perm_tuples()

    variants = []
    for perm_tuple in all_perms:
        perm_arr = np.array(perm_tuple)
        permuted_vonorms = vonorms[perm_arr]

        # Unimodular matrix from positions [1:4] of the vonorm permutation
        U = VonormPermutationMatrix.from_vector_idxs(perm_tuple[1:4]).matrix
        U_inv = np.round(np.linalg.inv(U)).astype(np.int64)

        # Transform motif coords: new_coords = U_inv @ coord_matrix
        transformed = (U_inv @ coord_matrix)  # (3, N)
        transformed = np.mod(transformed, delta).astype(np.int64)

        # Try each origin shift
        for origin_idx in range(num_origin_atoms):
            shift = -transformed[:, origin_idx]  # (3,)
            shifted = transformed + shift[:, np.newaxis]
            shifted = np.mod(shifted, delta).astype(np.int64)

            # Sort by (element, x, y, z)
            sorted_cm = sort_motif_coord_arr(shifted, atom_labels)  # (3, N)

            # Extract MNF-style tuple: skip first atom (origin), flatten
            motif_tuple = sorted_cm.T[1:].flatten()

            variant = np.concatenate([permuted_vonorms, motif_tuple])
            variants.append(variant)

    return np.array(variants, dtype=np.int64)


def _precompute_goal_variants_full(goal: CrystalNormalForm, partial: bool = False):
    """Pre-compute goal variants using unimodular matrices across ALL coforms.

    Unlike ``_precompute_goal_variants`` which derives a single 3x3 matrix
    from each of 168 vonorm permutations, this version iterates over every
    conorm zero-set and every permissible conorm permutation therein,
    retrieving unimodular matrices that realise each permutation.

    This is necessary because a lattice neighbor may sit in a different
    Voronoi class than the goal, so we need transformations from all classes.

    Args:
        goal: The goal CNF to generate variants for.
        partial: If True, use only one matrix per (zero_set, conorm_perm)
            instead of all matrices. Much faster with some loss of accuracy.
    """
    vonorms = np.array(goal.coords[:7])
    delta = goal.delta
    elements = goal.elements

    # Reconstruct motif coord matrix (3, N) from the MNF tuple
    mnf_tuple = goal.coords[7:]
    n_stored = len(mnf_tuple) // 3
    stored_coords = np.array(mnf_tuple, dtype=np.int64).reshape(n_stored, 3)
    full_coords = np.vstack([np.zeros((1, 3), dtype=np.int64), stored_coords])
    coord_matrix = full_coords.T  # (3, N)

    # Atom labels for sorting
    atom_labels = []
    el_num = 0
    prev_el = elements[0]
    for el in elements:
        if el != prev_el:
            el_num += 1
            prev_el = el
        atom_labels.append(el_num)
    atom_labels = np.array(atom_labels)

    num_origin_atoms = sum(1 for el in elements if el == elements[0])

    # Iterate over ALL zero-sets (all coforms) so that cross-boundary
    # neighbors are covered regardless of Voronoi class.
    seen = set()
    variants = []
    for zero_set, perm_to_mats in ZERO_CONORM_SETS_TO_PERMUTATIONS_TO_UNIMOD_MATS.items():
        for conorm_perm, mat_list in perm_to_mats.items():
            vonorm_perm = CONORM_PERMUTATION_TO_VONORM_PERMUTATION[conorm_perm]
            perm_arr = np.array(vonorm_perm)
            permuted_vonorms = vonorms[perm_arr]

            mats_to_use = [mat_list[0]] if partial else mat_list
            for mat_tuple in mats_to_use:
                # Deduplicate by matrix identity
                mat_key = mat_tuple.tuple
                if mat_key in seen:
                    continue
                seen.add(mat_key)

                U = mat_tuple.matrix
                U_inv = np.round(np.linalg.inv(U)).astype(np.int64)

                transformed = np.mod(U_inv @ coord_matrix, delta).astype(np.int64)

                for origin_idx in range(num_origin_atoms):
                    shift = -transformed[:, origin_idx]
                    shifted = np.mod(transformed + shift[:, np.newaxis], delta).astype(np.int64)

                    sorted_cm = sort_motif_coord_arr(shifted, atom_labels)
                    motif_tuple = sorted_cm.T[1:].flatten()

                    variant = np.concatenate([permuted_vonorms, motif_tuple])
                    variants.append(variant)

    return np.unique(np.array(variants, dtype=np.int64), axis=0)


def make_heuristic(mode: str, weight: float = 0.5):
    """Factory function for A* heuristics.

    Args:
        mode: One of "manhattan", "unimodular_light", "unimodular_partial",
            "unimodular_full".
        weight: Weight for unimodular heuristics.

    Returns:
        A callable heuristic function.
    """
    if mode == "manhattan":
        return manhattan_distance
    elif mode == "unimodular_light":
        return UnimodularManhattanHeuristic(weight=weight, full=False, partial=False)
    elif mode == "unimodular_partial":
        return UnimodularManhattanHeuristic(weight=weight, full=False, partial=True)
    elif mode == "unimodular_full":
        return UnimodularManhattanHeuristic(weight=weight, full=True, partial=False)
    else:
        raise ValueError(f"Unknown heuristic mode: {mode}")


class UnimodularManhattanHeuristic:
    """Manhattan heuristic that accounts for vonorm reorderings.

    On first call, pre-computes all permuted variants of the goal CNFs.
    Subsequent calls are a single vectorised Manhattan distance computation.

    Usage::

        heuristic = UnimodularManhattanHeuristic()
        state = astar_pathfind(starts, goals, heuristic=heuristic)
    """

    def __init__(self, weight: float = 0.5, full: bool = True, partial: bool = False):
        self._goal_variants = None   # (N_variants, tuple_len) array
        self._goals_id = None        # identity check for cache invalidation
        self.weight = weight
        self.full = full
        self.partial = partial

    def _ensure_precomputed(self, goals: list[CrystalNormalForm]):
        goals_id = id(goals)
        if self._goal_variants is not None and self._goals_id == goals_id:
            return
        if self.full:
            precompute = lambda g: _precompute_goal_variants_full(g, partial=self.partial)
        else:
            precompute = _precompute_goal_variants
        variant_list = []
        for goal in goals:
            variant_list.append(precompute(goal))
        self._goal_variants = np.vstack(variant_list)
        self._goals_id = goals_id

    def __call__(self, cnf: tuple, goals: list[CrystalNormalForm]) -> float:
        self._ensure_precomputed(goals)
        current = np.array(cnf, dtype=np.int64)
        dists = np.sum(np.abs(self._goal_variants - current), axis=1)
        return float(np.min(dists)) * self.weight