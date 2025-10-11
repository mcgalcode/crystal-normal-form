import json
import tqdm

from cnf.linalg.matrix_tuple import MatrixTuple
from itertools import product
from importlib.resources import files



def build_unimodular_matrices(max_entry_val):
    entry_choices = range(-max_entry_val, max_entry_val + 1)
    print(f"I expect {len(entry_choices) ** 9} matrices")
    combinations = list(product(entry_choices, repeat=9))

    all_matrices = set([MatrixTuple.from_tuple(tuple(c)) for c in tqdm.tqdm(combinations, "Constructing matrix tuples...")])
    print(f"Found {len(all_matrices)} distinct matrices")

    all_unimodular = [m for m in tqdm.tqdm(all_matrices, "Testing unimodularity...") if m.is_unimodular()]
    print(f"Of which, {len(all_unimodular)} were unimodular")

    det_one = [m for m in tqdm.tqdm(all_unimodular, "Testing det=1") if m.determinant() == 1]
    print(f"Of which, {len(det_one)} had determinant == 1")

    for m in det_one:
        assert m.determinant() == 1

    with open(f"unimodular_{max_entry_val}.json", 'w') as f:
        json.dump([m.to_list() for m in det_one], f)

def load_unimodular(fname = "unimodular.json"):
    data = files("cnf.linalg").joinpath("data", fname).read_text()
    matrix_lists = json.loads(data)
    matrices = [MatrixTuple.from_tuple(tuple(l)) for l in matrix_lists]
    return matrices

UNIMODULAR_MATRICES = load_unimodular()
UNIMODULAR_MATRICES_MAX_2 = load_unimodular("unimodular_2.json")
