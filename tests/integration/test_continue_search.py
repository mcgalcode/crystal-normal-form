import pytest
import helpers
import tempfile
from cnf.db.setup import setup_cnf_db
from cnf.search import instantiate_search, continue_search
from cnf.db.crystal_map_store import CrystalMapStore
from cnf.db.search_store import SearchProcessStore
from cnf.navigation.neighbor_finder import NeighborFinder
from cnf.calculation.base_calculator import BaseCalculator
from cnf import UnitCell, CrystalNormalForm

@pytest.fixture
def zr_hcp_mp():
    return helpers.load_specific_cif("Zr_HCP.cif")

def test_can_find_path(zr_hcp_mp):
    xi = 1.5
    delta = 10

    zr_hcp_cnf = CrystalNormalForm.from_pmg_struct(zr_hcp_mp, xi, delta)
    elements = zr_hcp_cnf.elements

    with tempfile.NamedTemporaryFile() as tf:
        setup_cnf_db(tf.name, xi, delta, elements)
        cmap_store = CrystalMapStore.from_file(tf.name)
        start_pt = zr_hcp_cnf
        known_pts = set([zr_hcp_cnf])
        all_nbs1 = NeighborFinder(zr_hcp_cnf).find_neighbors()
        new_nbs1 = set(all_nbs1).difference(known_pts)
        known_pts = known_pts.union(all_nbs1)
        path_pt_1 = list(new_nbs1)[0]
        all_nbs2 = NeighborFinder(path_pt_1).find_neighbors()
        new_nbs2 = set(all_nbs2).difference(known_pts)
        known_pts = known_pts.union(all_nbs2)

        path_pt_2 = list(new_nbs2)[0]
        all_nbs3 = NeighborFinder(path_pt_2).find_neighbors()
        new_nbs3 = set(all_nbs3).difference(known_pts)
        endpt = list(new_nbs3)[0]
        # print("path:")
        # print("startpt: ", start_pt.coords)
        # print(path_pt_1.coords)
        # print(path_pt_2.coords)
        # print("endpoint: ", endpt.coords)

        class DummyCalc(BaseCalculator):

            def calculate_energy(self, cnf):
                if cnf in [path_pt_1, path_pt_2, endpt]:
                    return 0
                else:
                    return 1000
        
        search_id = instantiate_search("dummy search", [start_pt], [endpt], tf.name)

        continue_search(search_id, tf.name, DummyCalc(), 100)

        search_store = SearchProcessStore.from_file(tf.name)
        endpts = search_store.get_endpoint_ids_in_frontier(search_id)
        assert len(endpts) == 1
    
        pt = cmap_store.get_point_by_id(endpts[0])
        assert pt.cnf == endpt


        