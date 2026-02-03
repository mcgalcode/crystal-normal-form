import pytest

from cnf.navigation.astar.core import astar_pathfind
from cnf.navigation.astar import astar_rust
from cnf.navigation.endpoints import get_endpoint_cnfs
from cnf.navigation.astar.heuristics import manhattan_distance
from cnf.navigation.search_filters import MinDistanceFilter, FilterSet

def test_astar_can_find_path(zr_bcc_mp, zr_hcp_mp):
    xi = 1.5
    delta = 10


    start_cnfs, goal_cnfs = get_endpoint_cnfs(zr_hcp_mp, zr_bcc_mp, xi, delta)

    search_state = astar_pathfind(start_cnfs,
                                  goal_cnfs,
                                  manhattan_distance,
                                  verbose=True
    )
    
    assert search_state.path is not None
