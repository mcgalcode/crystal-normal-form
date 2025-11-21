import os

FATAL = 0
SEVERE = 1
WARN = 2
INFO = 3
DEBUG = 4

DEFAULT = SEVERE

CNF_LOG_LEVEL_ENV_VAR = "CNF_LOG_LEVEL"
ENV_LOG_LEVEL = int(os.getenv(CNF_LOG_LEVEL_ENV_VAR, DEFAULT))

class Logger():

    def __init__(self, lvl: int = ENV_LOG_LEVEL):
        self.lvl = lvl

    def log(self, msg: str, lvl=1) -> None:
        if lvl <= self.lvl:
            print(f"[worker_id={os.getpid()}] " + msg)

    def fatal(self, msg: str) -> None:
        self.log(msg, FATAL)
    
    def severe(self, msg: str) -> None:
        self.log(msg, SEVERE)
    
    def warn(self, msg: str) -> None:
        self.log(msg, WARN)
    
    def info(self, msg: str) -> None:
        self.log(msg, INFO)
    
    def debug(self, msg: str) -> None:
        self.log(msg, DEBUG)