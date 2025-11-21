import os

FATAL = 0
SEVERE = 1
WARN = 2
INFO = 3
DEBUG = 4

DEFAULT = SEVERE

CNF_LOG_LEVEL_ENV_VAR = "CNF_LOG_LEVEL"
ENV_LOG_LEVEL = os.getenv(CNF_LOG_LEVEL_ENV_VAR, DEFAULT)

class Logger():

    def __init__(self, lvl = ENV_LOG_LEVEL):
        self.lvl = lvl

    def log(self, msg, lvl=1):
        if lvl <= self.lvl:
            print(f"[worker_id={os.getpid()}] " + msg)