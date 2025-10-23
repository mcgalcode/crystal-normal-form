import helpers
import pytest
from cnf import CrystalNormalForm, UnitCell
from cnf.navigation.crystal_map import CrystalMap
from cnf.navigation.neighbor_finder import NeighborFinder
from cnf.navigation.crystal_explorer import CrystalExplorer
from cnf.navigation.search_objectives import LocateAnyTargetStruct
from cnf.navigation.score_functions import PDDScorer, NullScore
from cnf.navigation.utils import get_endpoints_from_unit_cells
from cnf.navigation.search_filters import SimpleVolumeAndOverlapFilter
from rustworkx import all_shortest_paths
from itertools import product
from cnf.navigation.crystal_explorer import CrystalExplorer
from cnf.viz.trajectory import TrajectoryVisualizer

@pytest.fixture(scope='module')
def point_set():
    struct = helpers.ALL_MP_STRUCTURES(1)[0]
    xi = 1.5
    delta = 10
    cnf = UnitCell.from_pymatgen_structure(struct).to_cnf(xi=xi, delta=delta)
    nf = NeighborFinder(cnf)
    nbs = nf.find_neighbors()
    return xi, delta, nbs


@pytest.fixture
def ti_o2_anatase():
    return helpers.load_specific_cif("TiO2_anatase.cif")


def test_explorer_tracks_exploration_state(point_set):
    xi, delta, cnfs = point_set
    cmap = CrystalMap(xi, delta, cnfs[0].elements)

    cnf1 = cnfs[0]
    cnf2 = cnfs[1]

    cmap.add_point(cnf1)
    cmap.add_point(cnf2)

    target = cnfs[2]
    volumes = [cnf.reconstruct().volume for cnf in [cnf1, cnf2, target]]
    min_vol = min(volumes) * 0.8
    max_vol = max(volumes) * 1.2

    explorer = CrystalExplorer(cmap, min_vol, max_vol, [target])

    # Points should start as unexplored but scored
    assert not explorer.is_point_explored(cnf1)
    assert not explorer.is_point_explored(cnf2)
    assert explorer.score_for_point(cmap.get_point_id(cnf1)) is not None
    assert explorer.score_for_point(cmap.get_point_id(cnf2)) is not None


def test_explorer_marks_points_as_explored(point_set):
    xi, delta, cnfs = point_set
    cmap = CrystalMap(xi, delta, cnfs[0].elements)

    cnf = cnfs[0]
    cmap.add_point(cnf)

    target = cnfs[1]
    volumes = [cnf.reconstruct().volume, target.reconstruct().volume]
    min_vol = min(volumes) * 0.8
    max_vol = max(volumes) * 1.2

    explorer = CrystalExplorer(cmap, min_vol, max_vol, [target])

    assert not explorer.is_point_explored(cnf)

    pid = cmap.get_point_id(cnf)
    explorer.explore_point(pid)

    assert explorer.is_point_explored(cnf)


def test_explore_point_adds_neighbors(ti_o2_anatase):
    xi = 1.5
    delta = 10
    cnf = CrystalNormalForm.from_pmg_struct(ti_o2_anatase, xi, delta)

    cmap = CrystalMap.from_cnf(cnf)
    struct = ti_o2_anatase
    min_vol = struct.volume * 0.8
    max_vol = struct.volume * 1.2

    explorer = CrystalExplorer(cmap, min_vol, max_vol, [cnf])

    assert not explorer.is_point_explored(cnf)

    pid = cmap.get_point_id(cnf)
    new_pt_ids = explorer.explore_point(pid)

    for nid in new_pt_ids:
        pt = cmap.get_point_by_id(nid)
        assert pt in cmap
        assert cmap.connection_exists(cnf, pt)
        assert not explorer.is_id_explored(nid)

    assert explorer.is_point_explored(cnf)


def test_explorer_handles_unscored_points(point_set):
    xi, delta, cnfs = point_set
    cmap = CrystalMap(xi, delta, cnfs[0].elements)

    cnf1 = cnfs[0]
    cmap.add_point(cnf1)

    target = cnfs[1]
    volumes = [cnf1.reconstruct().volume, target.reconstruct().volume]
    min_vol = min(volumes) * 0.8
    max_vol = max(volumes) * 1.2

    # Create explorer with skip_scoring=True
    explorer = CrystalExplorer(cmap, min_vol, max_vol, [target], skip_scoring=True)

    # Point should exist but not be scored
    pid = cmap.get_point_id(cnf1)
    assert pid not in explorer.scores

    # Manually score it
    score = explorer.score_pt(cnf1)
    assert pid in explorer.scores
    assert explorer.score_for_point(pid) == score


def test_best_current_score(point_set):
    xi, delta, cnfs = point_set
    cmap = CrystalMap(xi, delta, cnfs[0].elements)

    for cnf in cnfs[:3]:
        cmap.add_point(cnf)

    target = cnfs[3]
    volumes = [cnf.reconstruct().volume for cnf in cnfs[:4]]
    min_vol = min(volumes) * 0.8
    max_vol = max(volumes) * 1.2

    explorer = CrystalExplorer(cmap, min_vol, max_vol, [target])

    best_score = explorer.best_current_score()
    assert best_score is not None

    # Best score should be the minimum of all scores
    all_scores = [explorer.score_for_point(cmap.get_point_id(cnf)) for cnf in cnfs[:3]]
    assert best_score == min(all_scores)


def test_unexplored_points_list(point_set):
    xi, delta, cnfs = point_set
    cmap = CrystalMap(xi, delta, cnfs[0].elements)

    cnf1 = cnfs[0]
    cnf2 = cnfs[1]
    cmap.add_point(cnf1)
    cmap.add_point(cnf2)

    target = cnfs[2]
    volumes = [cnf.reconstruct().volume for cnf in [cnf1, cnf2, target]]
    min_vol = min(volumes) * 0.8
    max_vol = max(volumes) * 1.2

    explorer = CrystalExplorer(cmap, min_vol, max_vol, [target])

    unexplored = explorer.unexplored_points()
    assert len(unexplored) == 2

    # Explore one point
    explorer.explore_point(unexplored[0])

    unexplored_after = explorer.unexplored_points()
    assert len(unexplored_after) < len(unexplored)

@pytest.fixture
def path_find_start_structs(zr_bcc_mp):
    return [UnitCell.from_pymatgen_structure(zr_bcc_mp)]

@pytest.fixture
def path_find_end_structs(zr_hcp_mp):
    return [UnitCell.from_pymatgen_structure(zr_hcp_mp)]
    

FNAME = "explorations/result3.json"
@pytest.mark.skip
def test_can_connect_two_points(path_find_start_structs, path_find_end_structs):
    xi = 1.5
    delta = 5

    start_cnfs = list(set([s.to_cnf(xi, delta) for s in path_find_start_structs]))
    end_cnfs = list(set([s.to_cnf(xi, delta) for s in path_find_end_structs]))
    cmap = CrystalMap.from_cnfs(start_cnfs)
    

    search_filter = SimpleVolumeAndOverlapFilter.from_endpoint_structs(
        [*path_find_start_structs, *path_find_end_structs],
        0.85
    )
    score_fun = PDDScorer(path_find_end_structs[:1])
    score_fun = NullScore()
    objective = LocateAnyTargetStruct(end_cnfs)

    explorer = CrystalExplorer(cmap, search_filter, score_fun)
    
    explorer.search(objective)
    assert objective.objective_complete(explorer)
    print(f"Started with {len(start_cnfs)} structures")
    print(f"Searched for {len(end_cnfs)} structures")
    print(objective.located_endpt.coords)
    dpath = helpers.get_data_file_path(FNAME)
    explorer.to_json(dpath)

@pytest.mark.skip
def test_can_get_path_from_exploration_result(path_find_start_structs, path_find_end_structs):
    dpath = helpers.get_data_file_path(FNAME)
    exp = CrystalExplorer.from_json(dpath)
    print(len(exp.map.all_node_ids()))
    print(len(exp._explored_pts))
    print(len(exp._unexplored_pts))

    xi = exp.map.xi
    delta = exp.map.delta

    start_cnfs = set([s.to_cnf(xi, delta) for s in path_find_start_structs])
    end_cnfs = set([s.to_cnf(xi, delta) for s in path_find_end_structs])
    
    path_cnfs = exp.map.find_path(start_cnfs, end_cnfs)
    for pt in path_cnfs:
        print(pt.coords)
    
    tv = TrajectoryVisualizer("XDATCAR", (3,3,3))
    tv.save_trajectory_from_cnfs(path_cnfs)
