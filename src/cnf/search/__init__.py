from ..crystal_normal_form import CrystalNormalForm
from ..calculation.grace import GraceCalculator
from ..db.crystal_map_store import CrystalMapStore
from ..db.partitioned_db import PartitionedDB
from ..db.search_store import SearchProcessStore
from ..navigation.neighbor_finder import NeighborFinder
from ..navigation.search_filters import VolumeLimitFilter, SearchFilter
from ..calculation.base_calculator import BaseCalculator
from ..utils.log import Logger
import time
import math

FRONTIER_WIDTH = 0.002 # eV

def explore_pt(map_store: CrystalMapStore, pt_id: int, filters: list[SearchFilter] = None, log_lvl=1):
    if filters is None:
        filters = []

    pt = map_store.get_point_by_id(pt_id)

    nbs = NeighborFinder.from_cnf(pt.cnf).find_neighbors(pt.cnf)
    
    all_nb_ids = []
    new_nb_ids = []
    
    edges_to_add_to_point_store = []
    
    for nb in nbs:
    
        if len(filters) > 0:
            struct = nb.reconstruct()
            if not all([f.should_add_pt(nb, struct) for f in filters]):
                continue

        nb_id = map_store.add_point(nb)

        if nb_id is not None:
            new_nb_ids.append(nb_id)
        else:
            nb_id = map_store.get_point_by_cnf(nb).id

        edges_to_add_to_point_store.append((pt.id, nb_id, None))

        all_nb_ids.append(nb_id)
    
    map_store.bulk_add_edges(edges_to_add_to_point_store)

    map_store.mark_point_explored(pt.id)

    return all_nb_ids, new_nb_ids

