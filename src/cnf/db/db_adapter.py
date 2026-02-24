import sqlite3

class DBAdapter():

    def __init__(self, dbfname: str):
        self.db_filename = dbfname
        self.conn = sqlite3.connect(self.db_filename, timeout=30)
        self.cursor = self.conn.cursor()

    def close(self):
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
            self.cursor = None
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        self.close()