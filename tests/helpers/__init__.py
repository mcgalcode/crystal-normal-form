import os
import pytest

from .assertions import *
from .data import ALL_MP_STRUCTURES, load_pathological_pair

IS_FAST = int(os.getenv("CNF_FAST_TEST", 0)) == 1

STRUCT_SAMPLE_FREQ = int(os.getenv("SSF", 10))

def skip_if_fast(func):
    return pytest.mark.skipif(IS_FAST, reason="Skipped because CNF_FAST_TEST env var was set to 1")(func)

def parameterized_by_mp_structs(func):
    return pytest.mark.parametrize("idx, struct", zip(range(0, len(ALL_MP_STRUCTURES), STRUCT_SAMPLE_FREQ), ALL_MP_STRUCTURES[::STRUCT_SAMPLE_FREQ]))(func)


def printif(msg, flag):
    if flag:
        print(msg)

    