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
    point_id INTEGER,
    UNIQUE(search_id, point_id)
)
"""

create_searched_point_table = f"""
CREATE TABLE {constants.SEARCHED_POINT_TABLE_NAME} (
    search_id INTEGER,
    point_id INTEGER,
    UNIQUE(search_id, point_id)
)
"""

create_incoming_point_table = f"""
CREATE TABLE {constants.INCOMING_POINT_TABLE_NAME} (
    search_id INTEGER,
    cnf TEXT
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

mark_point_searched = f"""
INSERT OR IGNORE INTO {constants.SEARCHED_POINT_TABLE_NAME} (search_id, point_id)
VALUES (?, ?)
"""

add_point_to_frontier = f"""
INSERT OR IGNORE INTO {constants.SEARCH_FRONTIER_MEMBER_TABLE_NAME} (search_id, point_id)
VALUES (?, ?)
"""

rm_point_from_frontier = f"""
DELETE FROM {constants.SEARCH_FRONTIER_MEMBER_TABLE_NAME}
WHERE search_id = ? AND point_id = ?
"""

select_searched_points = f"""
SELECT pt.* FROM {constants.SEARCHED_POINT_TABLE_NAME} AS searched_pts
LEFT JOIN {constants.POINT_TABLE_NAME} AS pt
ON pt.id = searched_pts.point_id
WHERE searched_pts.search_id = ?
"""

select_frontier_points = f"""
SELECT pt.* FROM {constants.SEARCH_FRONTIER_MEMBER_TABLE_NAME} AS frontier_pts
LEFT JOIN {constants.POINT_TABLE_NAME} AS pt
ON pt.id = frontier_pts.point_id
WHERE frontier_pts.search_id = ?
ORDER BY pt.value ASC
LIMIT ?
"""

select_min_frontier_energy = f"""
SELECT MIN(pt.value) FROM {constants.SEARCH_FRONTIER_MEMBER_TABLE_NAME} AS frontier_pts
LEFT JOIN {constants.POINT_TABLE_NAME} AS pt
ON pt.id = frontier_pts.point_id
WHERE frontier_pts.search_id = ? AND pt.value IS NOT NULL
"""

select_frontier_points_with_max_energy = f"""
SELECT pt.* FROM {constants.SEARCH_FRONTIER_MEMBER_TABLE_NAME} AS frontier_pts
LEFT JOIN {constants.POINT_TABLE_NAME} AS pt
ON pt.id = frontier_pts.point_id
WHERE frontier_pts.search_id = ? AND pt.value IS NOT NULL AND pt.value <= ?
ORDER BY pt.value ASC
LIMIT ?
"""

select_frontier_point_ids = f"""
SELECT point_id FROM {constants.SEARCH_FRONTIER_MEMBER_TABLE_NAME}
WHERE search_id = ?
"""

select_endpt_ids_in_frontier = f"""
SELECT ft.point_id
FROM {constants.SEARCH_FRONTIER_MEMBER_TABLE_NAME} as ft
WHERE ft.search_id = ?
INTERSECT
SELECT sep.end_point_id FROM {constants.SEARCH_END_POINT_TABLE_NAME} AS sep
WHERE sep.search_id = ?
"""

# Index creation queries for performance optimization
create_index_frontier_search_point = f"""
CREATE INDEX IF NOT EXISTS idx_frontier_search_point
ON {constants.SEARCH_FRONTIER_MEMBER_TABLE_NAME} (search_id, point_id)
"""

create_index_searched_search_point = f"""
CREATE INDEX IF NOT EXISTS idx_searched_search_point
ON {constants.SEARCHED_POINT_TABLE_NAME} (search_id, point_id)
"""

create_index_start_point_search = f"""
CREATE INDEX IF NOT EXISTS idx_start_point_search
ON {constants.SEARCH_START_POINT_TABLE_NAME} (search_id)
"""

create_index_end_point_search = f"""
CREATE INDEX IF NOT EXISTS idx_end_point_search
ON {constants.SEARCH_END_POINT_TABLE_NAME} (search_id)
"""

select_all_incoming_points = f"""
SELECT cnf FROM {constants.INCOMING_POINT_TABLE_NAME}
WHERE search_id = ?
"""

delete_all_incoming_points = f"""
DELETE FROM {constants.INCOMING_POINT_TABLE_NAME}
WHERE search_id = ?
"""

insert_incoming_point = f"""
INSERT INTO {constants.INCOMING_POINT_TABLE_NAME}
    (search_id, cnf)
VALUES
    (?, ?)
"""