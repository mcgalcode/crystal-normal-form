import sqlite3
import json

from .queries import crystal_map as crystal_map_queries
from .queries import search_process as search_process_queries

def setup_cnf_db(dbfname: str, xi: float, delta: int, element_list: list[str]):
    conn = sqlite3.connect(dbfname)
    cur = conn.cursor()
    cur.execute(crystal_map_queries.create_point_table)
    cur.execute(crystal_map_queries.create_edge_table)
    cur.execute(crystal_map_queries.create_metadata_table)
    cur.execute(crystal_map_queries.create_lock_table)
    cur.execute(search_process_queries.create_search_process_table)
    cur.execute(search_process_queries.create_search_frontier_member_table)
    cur.execute(search_process_queries.create_searched_point_table)

    el_str = json.dumps(element_list)
    cur.execute(
        crystal_map_queries.set_metadata,
        (delta, xi, el_str)
    )
    conn.commit()
    return dbfname