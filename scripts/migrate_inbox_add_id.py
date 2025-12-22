#!/usr/bin/env python3
"""Migration script to add autoincrementing ID column to inbox table.

This migration:
1. Creates a new inbox table with the ID column
2. Copies existing data from the old table
3. Drops the old table and renames the new one
4. Runs on all partition databases in a given directory
"""

import sqlite3
import pathlib
import sys

def migrate_inbox_table(db_path: str):
    """Migrate a single database file's inbox table."""
    print(f"Migrating {db_path}...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if the table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='incoming_point'")
        if not cursor.fetchone():
            print(f"  No incoming_point table found, skipping...")
            return

        # Check if ID column already exists
        cursor.execute("PRAGMA table_info(incoming_point)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]

        if 'id' in column_names:
            print(f"  ID column already exists, skipping...")
            return

        # Count existing rows
        cursor.execute("SELECT COUNT(*) FROM incoming_point")
        row_count = cursor.fetchone()[0]
        print(f"  Found {row_count} rows to migrate...")

        # Create new table with ID column
        cursor.execute("""
            CREATE TABLE incoming_point_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                search_id INTEGER,
                cnf TEXT
            )
        """)

        # Copy data from old table to new table
        cursor.execute("""
            INSERT INTO incoming_point_new (search_id, cnf)
            SELECT search_id, cnf FROM incoming_point
        """)

        # Drop old table
        cursor.execute("DROP TABLE incoming_point")

        # Rename new table
        cursor.execute("ALTER TABLE incoming_point_new RENAME TO incoming_point")

        conn.commit()
        print(f"  ✓ Successfully migrated {row_count} rows")

    except Exception as e:
        conn.rollback()
        print(f"  ✗ Error: {e}")
        raise
    finally:
        conn.close()

def migrate_directory(db_dir: str):
    """Migrate all partition databases in a directory."""
    directory = pathlib.Path(db_dir)

    if not directory.exists():
        print(f"Error: Directory {db_dir} does not exist")
        sys.exit(1)

    # Find all partition database files
    partition_files = list(directory.glob("*.partition.db"))

    if not partition_files:
        print(f"Error: No partition database files found in {db_dir}")
        sys.exit(1)

    print(f"Found {len(partition_files)} partition databases to migrate")
    print("=" * 60)

    for partition_file in sorted(partition_files):
        migrate_inbox_table(str(partition_file))

    print("=" * 60)
    print(f"✓ Migration complete for all {len(partition_files)} databases")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python migrate_inbox_add_id.py <db_directory>")
        print("Example: python migrate_inbox_add_id.py experiments/tio2/tio2_ar_test")
        sys.exit(1)

    db_dir = sys.argv[1]
    migrate_directory(db_dir)
