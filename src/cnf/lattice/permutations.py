from itertools import permutations
import numpy as np
from .constants import VONORM_TO_CONORM_TRANSFORM_NSO_SUPERBASIS
import json
import tqdm

from importlib.resources import files

from .unimodular import get_unimodular_matrix_from_voronoi_vector_idxs


VONORM_PERMUTATION_TO_CONORM_PERMUTATION = None
CONORM_PERMUTATION_TO_VONORM_PERMUTATION = None

class Permutation(tuple):

    def __init__(self, perm):
        self.perm = perm
        super().__init__()
    
    def compose(self, other: 'Permutation'):
        composed = compose_permutations(self.perm, other.perm)
        return self.__class__(composed)

class ConormPermutation(Permutation):

    def to_vonorm_permutation(self):
        return VonormPermutation(CONORM_PERMUTATION_TO_VONORM_PERMUTATION[self.perm])

    def to_unimodular_matrix(self):
        return self.to_vonorm_permutation().to_unimodular_matrix()
    
class VonormPermutation(Permutation):

    def to_conorm_permutation(self):
        return ConormPermutation(VONORM_PERMUTATION_TO_CONORM_PERMUTATION[self.perm])
    
    def to_unimodular_matrix(self):
        generating_vec_indices = self.perm[1:4]
        return get_unimodular_matrix_from_voronoi_vector_idxs(generating_vec_indices)


def permutation_to_matrix(permutation):
    rows = []
    for idx in permutation:
        row = np.zeros(len(permutation))
        row[idx] = 1
        rows.append(row)
    return np.array(rows)

def load_matching_perms():
    perm_map = {}

    data = files("cnf.lattice").joinpath("data", "matching_perms.json").read_text()
    perm_pairs = json.loads(data)
    for pair in perm_pairs:
        vonorm_permutation = tuple(pair[0])
        conorm_permutation = tuple(pair[1])
        perm_map[vonorm_permutation] = conorm_permutation
    
    return perm_map

if VONORM_PERMUTATION_TO_CONORM_PERMUTATION is None:
    VONORM_PERMUTATION_TO_CONORM_PERMUTATION = load_matching_perms()
    CONORM_PERMUTATION_TO_VONORM_PERMUTATION = { v: k for k, v in VONORM_PERMUTATION_TO_CONORM_PERMUTATION.items() }

def find_matching_permutations():
    s7_permutations = list(permutations(range(7)))
    matching_pairs = []
    for p_v in tqdm.tqdm(s7_permutations):
        p_v_matrix = permutation_to_matrix(p_v)
        for p_c in s7_permutations:
            p_c_matrix = np.linalg.inv(permutation_to_matrix(p_c))
            if (p_c_matrix @ VONORM_TO_CONORM_TRANSFORM_NSO_SUPERBASIS @ p_v_matrix == VONORM_TO_CONORM_TRANSFORM_NSO_SUPERBASIS).all():
                matching_pairs.append([list(p_v), list(p_c)])
                # print("Found matching pair!")
                # print("P_v = ", p_v)
                # print("P_c = ", p_c)
    
    with open("matching_perms.json", 'w+') as f:
        json.dump(matching_pairs, f)

def compose_permutations(p1, p2):
    return tuple(apply_permutation(p2, p1))

def apply_permutation(to_permute, permutation):
    permuted_vals = []
    for p in permutation:
        permuted_vals.append(to_permute[p])
    return permuted_vals

def is_permutation_set_closed(permutations):
    for p1 in permutations:
        for p2 in permutations:
            composed = compose_permutations(p1, p2)
            
            if composed not in permutations:
                return False
    return True