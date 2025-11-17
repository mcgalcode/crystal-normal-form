from . import constants

create_search_process_table = f"""
CREATE TABLE {constants.SEARCH_PROCESS_TABLE_NAME} (
    id INTEGER PRIMARY KEY,
    start_cnf TEXT,
    end_cnf TEXT
)
"""

create_search_frontier_member_table = f"""
CREATE TABLE {constants.SEARCH_FRONTIER_MEMBER_TABLE_NAME} (
    search_id INTEGER,
    point_id INTEGER
)
"""

create_searched_point_table = f"""
CREATE TABLE {constants.SEARCHED_POINT_TABLE_NAME} (
    search_id INTEGER
    point_id INTEGER
)
"""