from itertools import permutations
import numpy as np
from .constants import VONORM_TO_CONORM_TRANSFORM_NSO_SUPERBASIS
import json
import tqdm

from importlib.resources import files

from .vonorm_unimodular import VonormPermutationMatrix
from ..linalg import MatrixTuple


VONORM_PERMUTATION_TO_CONORM_PERMUTATION = None
CONORM_PERMUTATION_TO_VONORM_PERMUTATION = None
CONORM_PERMUTATION_TO_VONORM_PERMUTATION_ARRAY = None

class Permutation(tuple):

    def __init__(self, perm):
        self.perm = perm
        super().__init__()
    
    def compose(self, other: 'Permutation'):
        composed = compose_permutations(self.perm, other.perm)
        return self.__class__(composed)

class ConormPermutation(Permutation):

    @staticmethod
    def all_conorm_perm_tuples():
        return list(CONORM_PERMUTATION_TO_VONORM_PERMUTATION.keys())
    
    @staticmethod
    def all_conorm_perms():
        return [ConormPermutation(p) for p in ConormPermutation.all_conorm_perm_tuples()]

    def to_vonorm_permutation(self):
        return VonormPermutation(CONORM_PERMUTATION_TO_VONORM_PERMUTATION[self.perm])
    
class VonormPermutation(Permutation):

    @staticmethod
    def all_vonorm_perm_tuples():
        return list(CONORM_PERMUTATION_TO_VONORM_PERMUTATION.values())
    
    @staticmethod
    def all_vonorm_perms():
        return [VonormPermutation(p) for p in VonormPermutation.all_vonorm_perm_tuples()]

    def to_conorm_permutation(self):
        return ConormPermutation(VONORM_PERMUTATION_TO_CONORM_PERMUTATION[self.perm])

class PermutationMatrix():

    @classmethod
    def identity(cls):
        vperm = VonormPermutation((0,1,2,3,4,5,6))
        mat = MatrixTuple.identity()
        return cls(vperm, mat, [mat])

    def __init__(self, perm: Permutation, all_matrices: list[MatrixTuple]):
        self.perm = perm
        self.matrix = all_matrices[0]
        self.all_matrices = all_matrices

    @property
    def vonorm_permutation(self):
        if isinstance(self.perm, VonormPermutation):
            return self.perm
        else:
            return self.perm.to_vonorm_permutation()
        
    @property
    def conorm_permutation(self):
        if isinstance(self.perm, ConormPermutation):
            return self.perm
        else:
            return self.perm.to_conorm_permutation()


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
    # Pre-convert vonorm permutations to numpy arrays for fast indexing
    CONORM_PERMUTATION_TO_VONORM_PERMUTATION_ARRAY = { k: np.array(v, dtype=np.intp) for k, v in CONORM_PERMUTATION_TO_VONORM_PERMUTATION.items() }
    CONORM_PERMUTATION_TO_VONORM_PERMUTATION_LIST = { k: list(v) for k, v in CONORM_PERMUTATION_TO_VONORM_PERMUTATION.items() }

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

def load_unimod_mats_to_perms():
    data = files("cnf.lattice").joinpath("data", "unimodular_mats_max_6_det_1_to_perms.json").read_text()
    values = json.loads(data)
    
    results = {}
    for item in values:
        
        zeros = tuple(item[0])
        if zeros not in results:
            results[zeros] = {}
        
        mat = MatrixTuple.from_tuple(tuple(item[1]))
        perms = [tuple(p) for p in item[2]]
        for perm in perms:
            if perm not in results[zeros]:
                results[zeros][perm] = []

            results[zeros][perm].append(mat)
    return results

ZERO_CONORM_SETS_TO_PERMUTATIONS_TO_UNIMOD_MATS = load_unimod_mats_to_perms()

class UnimodPermMapper:

    @staticmethod
    def all_zero_sets():
        return list(ZERO_CONORM_SETS_TO_PERMUTATIONS_TO_UNIMOD_MATS.keys())

    @staticmethod
    def all_unimodular_matrices() -> list[MatrixTuple]:
        all_mats = []
        for zero_set, perm_map in ZERO_CONORM_SETS_TO_PERMUTATIONS_TO_UNIMOD_MATS.items():
            for perm, mats in perm_map.items():
                all_mats = all_mats + mats
        return all_mats

    @staticmethod
    def get_perms_for_zero_set(zeros: tuple):
        return [ConormPermutation(t) for t in list(ZERO_CONORM_SETS_TO_PERMUTATIONS_TO_UNIMOD_MATS[zeros].keys())]
    
    @staticmethod
    def get_matrices_for_zero_set_and_perm(zeros: tuple, perm: tuple):
        return ZERO_CONORM_SETS_TO_PERMUTATIONS_TO_UNIMOD_MATS[zeros][perm]


def compose_permutations(p1, p2):
    return tuple(apply_permutation(p2, p1))

def apply_permutation(to_permute, permutation):
    permuted_vals = []
    for p in permutation:
        permuted_vals.append(to_permute[p])
    return permuted_vals

def apply_permutation_np(to_permute: np.ndarray, permutation: np.ndarray):
    return to_permute[permutation]

def is_permutation_set_closed(permutations):
    for p1 in permutations:
        for p2 in permutations:
            composed = compose_permutations(p1, p2)
            
            if composed not in permutations:
                return False
    return True