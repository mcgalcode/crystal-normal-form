from . import constants

create_point_table = f"""
CREATE TABLE {constants.POINT_TABLE_NAME} (
    id INTEGER PRIMARY KEY,
    cnf TEXT UNIQUE,
    external_id TEXT,
    value REAL,
    explored INTEGER
);
"""

create_edge_table = f"""
CREATE TABLE {constants.EDGE_TABLE_NAME} (
    source_id INTEGER,
    target_id INTEGER
)
"""

create_metadata_table = f"""
CREATE TABLE {constants.METADATA_TABLE_NAME} (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    delta INTEGER,
    xi REAL,
    element_list TEXT
)
"""

create_lock_table = f"""
CREATE TABLE {constants.LOCK_TABLE_NAME} (
    point_id INTEGER UNIQUE
)
"""

set_metadata =  f"""
INSERT INTO {constants.METADATA_TABLE_NAME} 
    (delta, xi, element_list) 
VALUES (?, ?, ?)
"""

get_metadata = f"""
SELECT delta, xi, element_list FROM {constants.METADATA_TABLE_NAME} WHERE id = 1;
"""

get_all_cnfs = f"""
SELECT cnf FROM {constants.POINT_TABLE_NAME}
"""

insert_point = f"""
INSERT OR IGNORE INTO {constants.POINT_TABLE_NAME}
    (cnf, external_id, value, explored)
VALUES (?, ?, ?, ?)
"""

get_point_by_id = f"""
SELECT id, cnf, external_id, value, explored FROM {constants.POINT_TABLE_NAME}
WHERE id = ?
"""

get_point_by_cnf_str = f"""
SELECT id, cnf, external_id, value, explored FROM {constants.POINT_TABLE_NAME}
WHERE cnf = ?
"""

def get_points_ids(cnf_pts: list[str]):
    placeholders = ','.join(['?'] * len(cnf_pts))
    return f"""
SELECT id, cnf FROM {constants.POINT_TABLE_NAME} WHERE
cnf IN ({placeholders})
"""


delete_point_by_id = f"""
DELETE FROM {constants.POINT_TABLE_NAME}
WHERE id = ?
"""

delete_point_by_point = f"""
DELETE FROM {constants.POINT_TABLE_NAME}
WHERE cnf = ?
"""

create_connection = f"""
INSERT INTO {constants.EDGE_TABLE_NAME}
(source_id, target_id)
VALUES (?, ?)
"""

get_connection_by_ids = f"""
SELECT * FROM {constants.EDGE_TABLE_NAME}
WHERE (source_id = ? AND target_id = ?) OR (target_id = ? AND source_id = ?)
"""

delete_connection_by_ids = f"""
DELETE FROM {constants.EDGE_TABLE_NAME}
WHERE (source_id = ? AND target_id = ?) OR (target_id = ? AND source_id = ?)
"""

select_neighbors = f"""
SELECT pt2.* FROM {constants.POINT_TABLE_NAME} AS pt1
INNER JOIN {constants.EDGE_TABLE_NAME} AS edge ON pt1.id = edge.source_id 
INNER JOIN {constants.POINT_TABLE_NAME} AS pt2 ON edge.target_id = pt2.id
WHERE pt1.id = ?
UNION
SELECT pt2.* FROM {constants.POINT_TABLE_NAME} AS pt1
INNER JOIN {constants.EDGE_TABLE_NAME} AS edge ON pt1.id = edge.target_id 
INNER JOIN {constants.POINT_TABLE_NAME} AS pt2 ON edge.source_id = pt2.id
WHERE pt1.id = ?
"""

mark_point_explored = f"""
UPDATE {constants.POINT_TABLE_NAME} AS pt
SET explored = 1
WHERE pt.id = ?
"""

mark_point_unexplored = f"""
UPDATE {constants.POINT_TABLE_NAME} AS pt
SET explored = 0
WHERE pt.id = ?
"""

add_lock_for_point = f"""
INSERT OR IGNORE INTO {constants.LOCK_TABLE_NAME} (point_id)
VALUES (?)
"""

rm_lock_for_point = f"""
DELETE FROM {constants.LOCK_TABLE_NAME}
WHERE point_id = ?
"""

get_lock_for_point = f"""
SELECT * FROM {constants.LOCK_TABLE_NAME}
WHERE point_id = ?
"""

set_value_for_point = f"""
UPDATE {constants.POINT_TABLE_NAME} AS pt
SET value = ?
WHERE pt.id = ?
"""

get_point_value = f"""
SELECT pt.value
FROM {constants.POINT_TABLE_NAME} AS pt
WHERE pt.id = ?
"""

# Index creation queries for performance optimization
create_index_edge_source = f"""
CREATE INDEX IF NOT EXISTS idx_edge_source
ON {constants.EDGE_TABLE_NAME} (source_id)
"""

create_index_edge_target = f"""
CREATE INDEX IF NOT EXISTS idx_edge_target
ON {constants.EDGE_TABLE_NAME} (target_id)
"""

create_index_point_value = f"""
CREATE INDEX IF NOT EXISTS idx_point_value
ON {constants.POINT_TABLE_NAME} (value)
"""

create_index_point_cnf = f"""
CREATE INDEX IF NOT EXISTS idx_point_cnf
ON {constants.POINT_TABLE_NAME} (cnf)
"""