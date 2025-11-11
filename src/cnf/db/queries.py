
table_exists = """
SELECT name
FROM sqlite_master
WHERE type='table'
AND name='{table_name}'
"""

create_point_table = """
CREATE TABLE point (
    id INTEGER PRIMARY KEY,
    cnf TEXT UNIQUE,
    external_id TEXT,
    value REAL,
    explored INTEGER
)
"""