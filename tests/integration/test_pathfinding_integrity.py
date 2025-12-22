import pytest
import helpers

from cnf.db import PartitionedDB
from cnf.navigation import find_neighbors

def test_pathfind_database_is_correct():
    search_id = 1
    db_path = helpers.get_data_file_path("search/zr_basic")
    pdb = PartitionedDB(db_path, search_id)

    

    assert pdb.search_metadata.search_id == search_id
    assert len(pdb.search_metadata.end_cnfs) > 0
    assert len(pdb.search_metadata.start_cnfs) > 0
    assert pdb.is_search_complete()
    assert pdb.get_current_water_level() < -16
    assert pdb.get_current_water_level() > -17

    all_known_points = set()
    unexplored_pts = []
    seen_cnfs = {}    
    for i in pdb.partition_range:
        search_store = pdb.get_search_store_by_idx(i)
        map_store = pdb.get_map_store_by_idx(i)

        for point in map_store.get_all_explored_points():
            nonlocal_neighbors = map_store.get_nonlocal_neighbor_cnfs(point.id)
            for nb_cnf in nonlocal_neighbors:
                # Check that the neighbor exists in its correct partition
                nb_partition = pdb.get_partition_idx(nb_cnf)
                nb_map_store = pdb.get_map_store_by_idx(nb_partition)
                nb_search_store = pdb.get_search_store_by_idx(nb_partition)
                nb_point = nb_map_store.get_point_by_cnf(nb_cnf)
                if nb_point is None:
                    nb_inbox = nb_search_store.peek_incoming_points(search_id)
                    assert nb_cnf in nb_inbox, f"Cross-partition neighbor that isn't in map store should be in inbox!"
                else:
                    assert nb_point is not None, f"Cross-partition neighbor should exist in partition {nb_partition}"

        frontier_points = search_store.get_frontier_points_in_search(search_id, limit=10000)
        for point in frontier_points:
            assert point.value is not None, "Frontier points should have energy computed"
            assert not search_store.is_point_searched(search_id, point.cnf), "Frontier points should not be searched"

        for point in search_store.get_searched_points_in_search(search_id):
            assert point.explored

        for point in map_store.get_all_points():
            expected_partition = pdb.get_partition_idx(point.cnf)
            assert expected_partition == i, f"Point in partition {i} but hashes to {expected_partition}"

            # Check for duplicates
            cnf_key = str(point.cnf.coords)
            assert cnf_key not in seen_cnfs, f"Duplicate CNF found: {cnf_key}"
            seen_cnfs[cnf_key] = i     

            if point.explored:
                assert point.value is not None
                known_nbs = map_store.get_neighbor_cnfs(point.id)
                computed_nbs = find_neighbors(point.cnf)
                assert set(known_nbs) == set(computed_nbs)
                assert len(known_nbs) == len(computed_nbs)
                all_known_points = all_known_points.union(known_nbs)
            else:
                unexplored_pts.append(point.cnf)
    
    assert set(unexplored_pts).issubset(all_known_points)