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

create_search_point_status_table = f"""
CREATE TABLE {constants.SEARCH_POINT_STATUS_TABLE_NAME} (
    search_id INTEGER,
    point_id INTEGER,
    point_status TEXT,
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

upsert_search_point_status = f"""
INSERT INTO {constants.SEARCH_POINT_STATUS_TABLE_NAME}
    (search_id, point_id, point_status)
VALUES (?, ?, "{constants.POINT_STATUS_OPEN}")
ON CONFLICT(search_id, point_id) DO UPDATE SET
    point_status =  "{constants.POINT_STATUS_OPEN}"
"""

delete_search_point_status = f"""
DELETE FROM {constants.SEARCH_POINT_STATUS_TABLE_NAME}
WHERE search_id = ? AND point_id = ?
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

set_point_closed = f"""
UPDATE {constants.SEARCH_POINT_STATUS_TABLE_NAME}
SET point_status = "{constants.POINT_STATUS_CLOSED}"
WHERE search_id = ? AND point_id = ?
"""

set_point_open = f"""
UPDATE {constants.SEARCH_POINT_STATUS_TABLE_NAME}
SET point_status = "{constants.POINT_STATUS_OPEN}"
WHERE search_id = ? AND point_id = ?
"""

select_searched_points = f"""
SELECT pt.* FROM {constants.SEARCH_POINT_STATUS_TABLE_NAME} AS status
LEFT JOIN {constants.POINT_TABLE_NAME} AS pt
ON pt.id = status.point_id
WHERE status.search_id = ? AND status.point_status = "{constants.POINT_STATUS_CLOSED}"
"""

select_frontier_points = f"""
SELECT pt.* FROM {constants.SEARCH_POINT_STATUS_TABLE_NAME} AS status
LEFT JOIN {constants.POINT_TABLE_NAME} AS pt
ON pt.id = status.point_id
WHERE status.search_id = ? AND status.point_status = "{constants.POINT_STATUS_OPEN}"
ORDER BY pt.value ASC
LIMIT ?
"""

select_min_frontier_energy = f"""
SELECT MIN(pt.value) FROM {constants.SEARCH_POINT_STATUS_TABLE_NAME} AS status
INNER JOIN {constants.POINT_TABLE_NAME} AS pt
ON pt.id = status.point_id
WHERE status.search_id = ? AND
      status.point_status = "{constants.POINT_STATUS_OPEN}" AND
      pt.value IS NOT NULL
"""

select_frontier_points_with_max_energy = f"""
SELECT pt.* FROM {constants.SEARCH_POINT_STATUS_TABLE_NAME} AS status
INNER JOIN {constants.POINT_TABLE_NAME} AS pt
ON pt.id = status.point_id
WHERE status.search_id = ? AND
      pt.value IS NOT NULL AND
      pt.value <= ? AND
      status.point_status = "{constants.POINT_STATUS_OPEN}"
ORDER BY pt.value ASC
LIMIT ?
"""

select_frontier_point_ids = f"""
SELECT point_id FROM {constants.SEARCH_POINT_STATUS_TABLE_NAME}
WHERE search_id = ? AND point_status = "{constants.POINT_STATUS_OPEN}"
"""

select_endpt_ids_in_frontier = f"""
SELECT status.point_id
FROM {constants.SEARCH_POINT_STATUS_TABLE_NAME} as status
WHERE status.search_id = ? AND status.point_status = "{constants.POINT_STATUS_OPEN}"
INTERSECT
SELECT sep.end_point_id FROM {constants.SEARCH_END_POINT_TABLE_NAME} AS sep
WHERE sep.search_id = ?
"""

create_index_start_point_search = f"""
CREATE INDEX IF NOT EXISTS idx_start_point_search
ON {constants.SEARCH_START_POINT_TABLE_NAME} (search_id)
"""

create_idx_point_status_composite = f"""
CREATE INDEX idx_status_composite 
ON {constants.SEARCH_POINT_STATUS_TABLE_NAME} (point_id, search_id, point_status);
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