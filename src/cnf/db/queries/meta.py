from . import constants


create_partition_status_table = f"""
CREATE TABLE {constants.PARTITION_STATUS_TABLE_NAME} (
    partition_number INTEGER,
    min_frontier_energy REAL,
    search_id INTEGER,
    worker_heartbeat TIMESTAMP,
    total_points INTEGER DEFAULT 0,
    points_with_energy INTEGER DEFAULT 0,
    explored_points INTEGER DEFAULT 0,
    total_edges INTEGER DEFAULT 0,
    frontier_points INTEGER DEFAULT 0,
    searched_points INTEGER DEFAULT 0,
    inbox_size INTEGER DEFAULT 0,
    min_energy REAL,
    max_energy REAL,
    max_searched_energy REAL,
    stats_updated_at TIMESTAMP,
    UNIQUE(partition_number, search_id)
)
"""

create_search_status_table = f"""
CREATE TABLE {constants.SEARCH_STATUS_TABLE_NAME} (
    search_id INTEGER UNIQUE,
    is_complete INTEGER
)
"""

create_partition_entry = f"""
INSERT INTO {constants.PARTITION_STATUS_TABLE_NAME}
    (partition_number, search_id, worker_heartbeat)
VALUES (?, ?, CURRENT_TIMESTAMP)
"""

get_global_water_level = f"""
SELECT MIN(min_frontier_energy)
FROM {constants.PARTITION_STATUS_TABLE_NAME}
WHERE search_id = ?;
"""

get_partition_water_level = f"""
SELECT min_frontier_energy
FROM {constants.PARTITION_STATUS_TABLE_NAME}
WHERE partition_number = ? AND search_id = ?
"""


update_min_water_level = f"""
UPDATE {constants.PARTITION_STATUS_TABLE_NAME} SET
min_frontier_energy = ?,
worker_heartbeat = CURRENT_TIMESTAMP
WHERE partition_number = ? AND search_id = ?;
"""

insert_search_status = f"""
INSERT INTO {constants.SEARCH_STATUS_TABLE_NAME}
    (search_id, is_complete)
VALUES
    (?, ?)
"""

update_search_status = f"""
UPDATE {constants.SEARCH_STATUS_TABLE_NAME} SET
is_complete = ?
WHERE search_id = ?
"""

select_search_status = f"""
SELECT is_complete
FROM {constants.SEARCH_STATUS_TABLE_NAME}
WHERE search_id = ?
"""

update_partition_stats = f"""
UPDATE {constants.PARTITION_STATUS_TABLE_NAME} SET
    total_points = ?,
    points_with_energy = ?,
    explored_points = ?,
    total_edges = ?,
    frontier_points = ?,
    searched_points = ?,
    inbox_size = ?,
    min_energy = ?,
    max_energy = ?,
    max_searched_energy = ?,
    stats_updated_at = CURRENT_TIMESTAMP,
    worker_heartbeat = CURRENT_TIMESTAMP
WHERE partition_number = ? AND search_id = ?
"""

get_partition_stats = f"""
SELECT
    total_points,
    points_with_energy,
    explored_points,
    total_edges,
    frontier_points,
    searched_points,
    inbox_size,
    min_energy,
    max_energy,
    max_searched_energy,
    stats_updated_at
FROM {constants.PARTITION_STATUS_TABLE_NAME}
WHERE partition_number = ? AND search_id = ?
"""

get_all_partition_stats = f"""
SELECT
    partition_number,
    total_points,
    points_with_energy,
    explored_points,
    total_edges,
    frontier_points,
    searched_points,
    inbox_size,
    min_energy,
    max_energy,
    max_searched_energy,
    min_frontier_energy,
    stats_updated_at
FROM {constants.PARTITION_STATUS_TABLE_NAME}
WHERE search_id = ?
ORDER BY partition_number
"""