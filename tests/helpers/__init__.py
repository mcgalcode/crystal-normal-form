import os
import pytest

from .assertions import *


IS_FAST = int(os.getenv("CNF_FAST_TEST", 0)) == 1

def skip_if_fast(func):
    return pytest.mark.skipif(IS_FAST, reason="Skipped because CNF_FAST_TEST env var was set to 1")(func)

def printif(msg, flag):
    if flag:
        print(msg)