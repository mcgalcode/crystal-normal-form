from . import constants


create_partition_status_table = f"""
CREATE TABLE {constants.PARTITION_STATUS_TABLE_NAME} (
    partition_number INTEGER,
    min_frontier_energy REAL,
    search_id INTEGER,
    worker_heartbeat TIMESTAMP,
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