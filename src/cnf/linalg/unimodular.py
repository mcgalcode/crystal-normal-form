import json

from cnf.linalg.matrix_tuple import MatrixTuple
from itertools import product
from importlib.resources import files

def build_unimodular_matrices():
    entry_choices = [-1, 0, 1]
    print(f"I expect {3 ** 9} matrices")
    combinations = list(product(entry_choices, repeat=9))
    all_matrices = [MatrixTuple.from_tuple(tuple(c)) for c in combinations]
    print(f"Found {len(all_matrices)} distinct matrices")
    all_unimodular = [m for m in all_matrices if m.is_unimodular()]
    print(f"Of which, {len(all_unimodular)} were unimodular")
    det_one = [m for m in all_matrices if m.is_unimodular() and m.determinant() == 1]
    print(f"Of which, {len(det_one)} had determinant == 1")
    for m in det_one:
        assert m.determinant() == 1

    with open("unimodular.json", 'w') as f:
        json.dump([m.to_list() for m in det_one], f)

def load_unimodular():
    data = files("cnf.linalg").joinpath("data", "unimodular.json").read_text()
    matrix_lists = json.loads(data)
    matrices = [MatrixTuple.from_tuple(tuple(l)) for l in matrix_lists]
    return matrices

UNIMODULAR_MATRICES = load_unimodular()


if __name__ == "__main__":
    build_unimodular_matrices()