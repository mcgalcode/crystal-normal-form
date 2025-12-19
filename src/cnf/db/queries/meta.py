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