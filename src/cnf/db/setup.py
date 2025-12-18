import sqlite3
import json

from .queries import crystal_map as crystal_map_queries
from .queries import search_process as search_process_queries
from .queries import meta as meta_queries

from .search_store import SearchProcessStore
from .crystal_map_store import CrystalMapStore
from ..calculation.base_calculator import BaseCalculator
from ..crystal_normal_form import CrystalNormalForm

def setup_cnf_db(dbfname: str, xi: float, delta: int, element_list: list[str]):
    conn = sqlite3.connect(dbfname)
    cur = conn.cursor()

    # Enable WAL mode for better concurrent write performance
    cur.execute("PRAGMA journal_mode=WAL")

    # Create tables
    cur.execute(crystal_map_queries.create_point_table)
    cur.execute(crystal_map_queries.create_edge_table)
    cur.execute(crystal_map_queries.create_metadata_table)
    cur.execute(crystal_map_queries.create_lock_table)

    cur.execute(search_process_queries.create_search_process_table)
    cur.execute(search_process_queries.create_search_start_point_table)
    cur.execute(search_process_queries.create_search_end_point_table)
    cur.execute(search_process_queries.create_search_frontier_member_table)
    cur.execute(search_process_queries.create_searched_point_table)

    # Create indexes for performance
    cur.execute(crystal_map_queries.create_index_edge_source)
    cur.execute(crystal_map_queries.create_index_edge_target)
    cur.execute(crystal_map_queries.create_index_edge_target_cnf)
    cur.execute(crystal_map_queries.create_index_point_value)
    cur.execute(crystal_map_queries.create_index_point_cnf)

    cur.execute(search_process_queries.create_index_frontier_search_point)
    cur.execute(search_process_queries.create_index_searched_search_point)
    cur.execute(search_process_queries.create_index_start_point_search)
    cur.execute(search_process_queries.create_index_end_point_search)

    # Set metadata
    el_str = json.dumps(element_list)
    cur.execute(
        crystal_map_queries.set_metadata,
        (delta, xi, el_str)
    )
    conn.commit()
    return dbfname

def setup_meta_db(dbfname: str):
    conn = sqlite3.connect(dbfname)
    cur = conn.cursor()

    # Enable WAL mode for better concurrent write performance
    cur.execute("PRAGMA journal_mode=WAL")

    # Create tables
    cur.execute(meta_queries.create_partition_status_table)

def instantiate_search(search_description: str,
                       start_cnfs: list[CrystalNormalForm],
                       end_cnfs: list[CrystalNormalForm],
                       store_file: str,
                       calculator: BaseCalculator):
    all_cnfs = start_cnfs + end_cnfs
    xis = [cnf.xi for cnf in all_cnfs]
    if len(set(xis)) > 1:
        raise ValueError("Tried to instantiate search with CNFs having different xi values!")
    
    deltas = [cnf.delta for cnf in all_cnfs]
    if len(set(deltas)) > 1:
        raise ValueError("Tried to instantiate search with CNFs having different delta values!")

    element_list = start_cnfs[0].elements
    for cnf in all_cnfs:
        if cnf.elements != element_list:
            raise ValueError("Tried to instantiate search with CNFs having different element lists!")
        
    crystal_map_store = CrystalMapStore.from_file(store_file)
    for cnf in all_cnfs:
        pt_id = crystal_map_store.add_point(cnf)
        crystal_map_store.set_point_value(pt_id, value=calculator.calculate_energy(cnf))

    search_store = SearchProcessStore.from_file(store_file)
    search_id = search_store.create_search_process(search_description)

    start_point_ids = crystal_map_store.get_point_ids(start_cnfs)
    end_point_ids = crystal_map_store.get_point_ids(end_cnfs)

    for sid in start_point_ids:
        search_store.add_search_start_point(search_id, sid)
        search_store.add_to_search_frontier_by_id(search_id, sid)
    
    for eid in end_point_ids:
        search_store.add_search_end_point(search_id, eid)
    return search_id