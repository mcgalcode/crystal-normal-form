import pytest
import helpers
from rustworkx import all_shortest_paths
from itertools import product

from pymatgen.core.trajectory import Trajectory, Structure
from cnf.navigation.crystal_explorer import CrystalExplorer, get_endpoints_from_structs

@pytest.fixture
def zr_hcp():
    return helpers.load_specific_cif("Zr_HCP.cif")

@pytest.fixture
def zr_bcc():
    return helpers.load_specific_cif("Zr_BCC.cif")

def test_can_get_path_from_exploration_result(zr_bcc, zr_hcp):
    dpath = helpers.get_data_file_path("explorations/exploration.json")
    exp = CrystalExplorer.from_json(dpath)
    print(len(exp.map.all_node_ids()))
    print(len(exp._explored_pts))
    print(len(exp._unexplored_pts))

    xi = exp.map.xi
    delta = exp.map.delta

    start_structs, end_structs = get_endpoints_from_structs(zr_bcc, zr_hcp)
    start_cnfs = [s.to_cnf(xi, delta) for s in start_structs]
    end_cnfs = [s.to_cnf(xi, delta) for s in end_structs]    
    
    start_idxs = [exp.map.get_point_id(pt) for pt in start_cnfs]
    end_ids = [exp.map.get_point_id(pt) for pt in end_cnfs]

    # find the path
    pairs = product(start_idxs, end_ids)

    all_paths = []
    for p in pairs:
        all_paths.extend(all_shortest_paths(exp.map._graph, p[0], p[1]))
    
    all_paths = sorted(all_paths, key=len)
    best_path = all_paths[0]

    path_cnfs = [exp.map.get_point_by_id(i) for i in best_path]
    for idx, c in enumerate(path_cnfs):
        print(idx, c.coords)
    path_structs: list[Structure] = [cnf.reconstruct() for cnf in path_cnfs]
    path_structs = [s.make_supercell((3,3,3)) for s in path_structs]
    t = Trajectory.from_structures(path_structs, constant_lattice=False)
    t.write_Xdatcar("XDATCAR")