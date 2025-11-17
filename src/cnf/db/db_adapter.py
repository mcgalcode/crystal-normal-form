import sqlite3

class DBAdapter():

    def __init__(self, dbfname: str):
        self.db_filename = dbfname
        self.conn = sqlite3.connect(self.db_filename)
        self.cursor = self.conn.cursor()