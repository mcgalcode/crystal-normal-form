from . import constants

table_exists = """
SELECT name
FROM sqlite_master
WHERE type='table'
AND name='{table_name}'
"""

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
    id INTEGER PRIMARY KEY CHECK (id = 0),
    delta INTEGER,
    xi REAL,
    element_list TEXT
)
"""
