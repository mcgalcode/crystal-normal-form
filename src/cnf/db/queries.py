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
    id INTEGER PRIMARY KEY CHECK (id = 1),
    delta INTEGER,
    xi REAL,
    element_list TEXT
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
INSERT INTO {constants.POINT_TABLE_NAME}
    (cnf, external_id, value, explored)
VALUES (?, ?, ?, ?)
"""

get_point_by_id = f"""
SELECT * FROM {constants.POINT_TABLE_NAME}
WHERE id = ?
"""

get_point_by_cnf_str = f"""
SELECT id, cnf, external_id, value, explored FROM {constants.POINT_TABLE_NAME}
WHERE cnf = ?
"""