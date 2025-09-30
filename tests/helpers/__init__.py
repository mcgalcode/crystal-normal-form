import os

from .assertions import *


IS_FAST = int(os.getenv("CNF_FAST_TEST", 0)) == 1
