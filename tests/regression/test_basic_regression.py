import pytest
import json
from helpers import data as hd
import tqdm

from cnf import UnitCell, CrystalNormalForm
from cnf.navigation.neighbor_finder import NeighborFinder
from pymatgen.core.structure import Structure

XI=1.5
DELTA=10


def _make_neighbor_data():
    structs = hd._ALL_MP_STRUCTURES[::50]
    nb_data = {
        "xi": XI,
        "delta": DELTA,
        "examples": []
    }
    for s in tqdm.tqdm(structs):
        cnf = UnitCell.from_pymatgen_structure(s).to_cnf(xi=XI, delta=DELTA)
        nbs = NeighborFinder.from_cnf(cnf).find_neighbors(cnf)
        
        nb_data["examples"].append({
            "point": cnf.coords,
            "elements": cnf.elements,
            "neighbors": [list(n.coords) for n in nbs]
        })
    
    fp = hd.get_data_file_path("regression_data/neighbors.json")
    with open(fp, 'w+') as f:
        json.dump(nb_data, f)


def _make_cnf_data():
    structs: list[Structure] = hd._ALL_MP_STRUCTURES[::10]
    nb_data = {
        "xi": XI,
        "delta": DELTA,
        "examples": []
    }
    for s in tqdm.tqdm(structs):
        cnf = UnitCell.from_pymatgen_structure(s).to_cnf(xi=XI, delta=DELTA)
        
        nb_data["examples"].append({
            "point": cnf.coords,
            "structure": s.as_dict()
        })
    
    fp = hd.get_data_file_path("regression_data/to_cnf.json")
    with open(fp, 'w+') as f:
        json.dump(nb_data, f)
    


def test_neighbor_id_regression():
    nb_path = hd.get_data_file_path("regression_data/neighbors.json")
    with open(nb_path, 'r+') as f:
        dataset = json.load(f)

    xi = dataset["xi"]
    delta = dataset["delta"]
    for idx, ex in enumerate(dataset["examples"]):
        point = tuple(ex["point"])
        els = ex["elements"]
        cnf_pt = CrystalNormalForm.from_tuple(point, els, xi, delta)
        expected_nbs = [CrystalNormalForm.from_tuple(list(nb), els, xi, delta) for nb in ex["neighbors"]]
        actual_nbs = NeighborFinder.from_cnf(cnf_pt).find_neighbors(cnf_pt)
        assert len(expected_nbs) == len(actual_nbs), f"Different #s of neighbors found in regression test for example {idx}"
        assert set(expected_nbs) == set(actual_nbs), f"Different neighbor sets found in regression test for example {idx}"

def test_cnf_construction_regression():
    nb_path = hd.get_data_file_path("regression_data/to_cnf.json")
    with open(nb_path, 'r+') as f:
        dataset = json.load(f)

    xi = dataset["xi"]
    delta = dataset["delta"]
    passes = 0
    different_cnfs = []
    for idx, ex in enumerate(dataset["examples"]):
        original_cnf = tuple(ex["point"])
        struct = Structure.from_dict(ex["structure"])
        current_cnf = UnitCell.from_pymatgen_structure(struct).to_cnf(xi, delta)
        if original_cnf == current_cnf.coords:
            passes += 1
        else:
            different_cnfs.append((idx, original_cnf, current_cnf.coords))

    total = len(dataset["examples"])
    pass_rate = passes / total
    assert pass_rate >= 0.97, f"Different CNFs computed in regression test for examples {[d[0] for d in different_cnfs]}"


if __name__ == "__main__":
    _make_cnf_data()