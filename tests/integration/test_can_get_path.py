import pytest
import numpy as np
import helpers
from rustworkx import all_shortest_paths
from itertools import product
import tqdm
from pymatgen.core.trajectory import Trajectory, Structure
from cnf.navigation.crystal_explorer import CrystalExplorer
from cnf.navigation.utils import get_endpoints_from_unit_cells
from cnf.linalg.unimodular import get_unimodulars_col_max
from cnf import UnitCell
from pymatgen.analysis.structure_matcher import StructureMatcher
from cnf.viz.trajectory import TrajectoryVisualizer

def test_can_get_path_from_exploration_result(zr_bcc_manual_unit_cell, zr_fcc_manual_unit_cell):
    dpath = helpers.get_data_file_path("explorations/result.json")
    exp = CrystalExplorer.from_json(dpath)
    print(len(exp.map.all_node_ids()))
    print(len(exp._explored_pts))
    print(len(exp._unexplored_pts))

    xi = exp.map.xi
    delta = exp.map.delta

    start_structs = zr_bcc_manual_unit_cell.supercells(2) #[:1]
    end_structs = zr_fcc_manual_unit_cell.supercells(2)
    start_cnfs = set([s.to_cnf(xi, delta) for s in start_structs])
    end_cnfs = set([s.to_cnf(xi, delta) for s in end_structs])
    
    start_idxs = [exp.map.get_point_id(pt) for pt in start_cnfs]
    end_ids = [exp.map.get_point_id(pt) for pt in end_cnfs]
    end_ids = [i for i in end_ids if i is not None]
    print(start_idxs)
    print(end_ids)

    # find the path
    pairs = product(start_idxs, end_ids)

    all_paths = []
    for p in pairs:
        print(f"Finding paths for {p}")
        all_paths.extend(all_shortest_paths(exp.map._graph, p[0], p[1]))
    
    all_paths = sorted(all_paths, key=len)
    best_path = all_paths[0]

    path_cnfs = [exp.map.get_point_by_id(i) for i in best_path]
    for idx, c in enumerate(path_cnfs):
        print(idx, c.coords)
    
    tv = TrajectoryVisualizer("XDATCAR", (3,3,3))
    tv.save_trajectory_from_cnfs(path_cnfs)
