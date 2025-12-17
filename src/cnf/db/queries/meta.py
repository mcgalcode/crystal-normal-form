from . import constants


create_partition_status_table = f"""
CREATE TABLE {constants.PARTITION_STATUS_TABLE_NAME} (
    partition_id INTEGER,
    min_frontier_energy REAL,
    worker_heartbeat TIMESTAMP
)
"""

create_partition_entry = f"""
INSERT INTO {constants.PARTITION_STATUS_TABLE_NAME}
    (partition_id, worker_heartbeat)
VALUES (?, NOW())
"""

get_global_water_level = f"""
SELECT MIN(min_frontier_energy) FROM {constants.PARTITION_STATUS_TABLE_NAME};
"""

update_min_water_level = f"""
UPDATE {constants.PARTITION_STATUS_TABLE_NAME} SET
min_frontier_energy = ?,
worker_heartbeat = NOW()
WHERE partition_id = ?;
"""