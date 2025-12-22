#!/usr/bin/env python3
"""Migration script to add statistics columns to partition_status table in metastore.

This migration adds the following columns to partition_status:
- total_points, points_with_energy, explored_points, total_edges
- frontier_points, searched_points, inbox_size
- min_energy, max_energy, max_searched_energy
- stats_updated_at
"""

import sqlite3
import sys
import os


def migrate_metastore(meta_db_path: str):
    """Migrate a metastore database to add statistics columns."""
    print(f"Migrating metastore: {meta_db_path}...")

    if not os.path.exists(meta_db_path):
        print(f"  Error: {meta_db_path} does not exist")
        return False

    conn = sqlite3.connect(meta_db_path)
    cursor = conn.cursor()

    try:
        # Check if partition_status table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='partition_status'")
        if not cursor.fetchone():
            print(f"  No partition_status table found, skipping...")
            return True

        # Check if stats columns already exist
        cursor.execute("PRAGMA table_info(partition_status)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]

        if 'total_points' in column_names:
            print(f"  Statistics columns already exist, skipping...")
            return True

        print(f"  Adding statistics columns...")

        # SQLite doesn't support adding multiple columns at once, so add them one by one
        new_columns = [
            ('total_points', 'INTEGER DEFAULT 0'),
            ('points_with_energy', 'INTEGER DEFAULT 0'),
            ('explored_points', 'INTEGER DEFAULT 0'),
            ('total_edges', 'INTEGER DEFAULT 0'),
            ('frontier_points', 'INTEGER DEFAULT 0'),
            ('searched_points', 'INTEGER DEFAULT 0'),
            ('inbox_size', 'INTEGER DEFAULT 0'),
            ('min_energy', 'REAL'),
            ('max_energy', 'REAL'),
            ('max_searched_energy', 'REAL'),
            ('stats_updated_at', 'TIMESTAMP'),
        ]

        for col_name, col_type in new_columns:
            try:
                cursor.execute(f"ALTER TABLE partition_status ADD COLUMN {col_name} {col_type}")
                print(f"    Added column: {col_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    print(f"    Column {col_name} already exists, skipping")
                else:
                    raise

        conn.commit()
        print(f"  ✓ Successfully migrated metastore")
        return True

    except Exception as e:
        conn.rollback()
        print(f"  ✗ Error: {e}")
        return False
    finally:
        conn.close()


def main():
    if len(sys.argv) != 2:
        print("Usage: python migrate_metastore_add_stats.py <path_to_meta.db>")
        print("Example: python migrate_metastore_add_stats.py experiments/tio2/tio2_ar_test/meta.db")
        sys.exit(1)

    meta_db_path = sys.argv[1]
    success = migrate_metastore(meta_db_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
