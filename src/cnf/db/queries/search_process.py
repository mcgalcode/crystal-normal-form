from . import constants

create_search_process_table = f"""
CREATE TABLE {constants.SEARCH_PROCESS_TABLE_NAME} (
    id INTEGER PRIMARY KEY,
    description TEXT
)
"""

create_search_start_point_table = f"""
CREATE TABLE {constants.SEARCH_START_POINT_TABLE_NAME} (
    search_id INTEGER,
    start_point_id INTEGER
)
"""

create_search_end_point_table = f"""
CREATE TABLE {constants.SEARCH_END_POINT_TABLE_NAME} (
    search_id INTEGER,
    end_point_id INTEGER
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

insert_search_process = f"""
INSERT INTO {constants.SEARCH_PROCESS_TABLE_NAME}
    (description)
VALUES (?)
"""

insert_search_start_point = f"""
INSERT INTO {constants.SEARCH_START_POINT_TABLE_NAME}
    (search_id, start_point_id)
VALUES (?, ?)
"""

insert_search_end_point = f"""
INSERT INTO {constants.SEARCH_END_POINT_TABLE_NAME}
    (search_id, end_point_id)
VALUES (?, ?)
"""

select_search_end_points = f"""
SELECT pt.* FROM {constants.SEARCH_END_POINT_TABLE_NAME} AS sep
LEFT JOIN {constants.POINT_TABLE_NAME} AS pt
ON pt.id = sep.end_point_id
WHERE sep.search_id = ?
"""

select_search_start_points = f"""
SELECT pt.* FROM {constants.SEARCH_START_POINT_TABLE_NAME} AS ssp
LEFT JOIN {constants.POINT_TABLE_NAME} AS pt
ON pt.id = ssp.start_point_id
WHERE ssp.search_id = ?
"""