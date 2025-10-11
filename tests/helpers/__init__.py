import os
import pytest

from .assertions import *
from .data import ALL_MP_STRUCTURES, load_pathological_cifs

IS_FAST = int(os.getenv("CNF_FAST_TEST", 0)) == 1

STRUCT_SAMPLE_FREQ = int(os.getenv("SSF", 10))

SPECIFIC_STRUCT_IDX = os.getenv("SSI")
if SPECIFIC_STRUCT_IDX is not None:
    SPECIFIC_STRUCT_IDX = int(SPECIFIC_STRUCT_IDX)

def skip_if_fast(func):
    return pytest.mark.skipif(IS_FAST, reason="Skipped because CNF_FAST_TEST env var was set to 1")(func)

def parameterized_by_mp_structs(func):
    if SPECIFIC_STRUCT_IDX is None:
        return pytest.mark.parametrize("idx, struct", zip(range(0, len(ALL_MP_STRUCTURES), STRUCT_SAMPLE_FREQ), ALL_MP_STRUCTURES[::STRUCT_SAMPLE_FREQ]))(func)
    else:
        return parameterized_by_mp_struct_idxs([SPECIFIC_STRUCT_IDX])(func)

def parameterized_by_mp_struct_idxs(idxs):
    structs = [ALL_MP_STRUCTURES[i] for i in idxs]
    def _wrapper(func):
        return pytest.mark.parametrize("idx, struct", zip(idxs, structs))(func)    
    return _wrapper

def printif(msg, flag):
    if flag:
        print(msg)

    