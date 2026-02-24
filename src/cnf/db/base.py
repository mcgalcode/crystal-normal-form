from .db_adapter import DBAdapter

class BaseStore():

    @classmethod
    def from_file(cls, db_fname: str):
        return cls(DBAdapter(db_fname))

    def __init__(self, adapter: DBAdapter):
        self.db_filename = adapter.db_filename
        self.adapter = adapter

    @property
    def cursor(self):
        return self.adapter.cursor

    @property
    def conn(self):
        return self.adapter.conn

    def close(self):
        """Close the database connection."""
        if self.adapter:
            self.adapter.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False