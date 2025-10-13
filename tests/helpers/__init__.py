import numpy as np
import os
import pytest

from .assertions import *
from .data import ALL_MP_STRUCTURES
from cnf.motif.atomic_motif import FractionalMotif
from cnf.unit_cell import UnitCell

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
    
def are_geo_matches(uc1: UnitCell, uc2: UnitCell, tol=1e-5):
    struct1 = uc1.to_pymatgen_structure()
    struct2 = uc2.to_pymatgen_structure()

    pdd_dist = pdd(struct1, struct2)
    if pdd_dist > tol:
        return False

    # These might 
    if uc1.motif.has_inversion_symmetry():
        return True
    else:
        are_inversions = uc1.motif.find_inverted_match(uc2.motif, atol=tol)
        if are_inversions:
            return False
        else:
            return True

def are_cnfs_mirror_images(cnf1: CrystalNormalForm, cnf2: CrystalNormalForm, atol=1e-6):
    motif1 = FractionalMotif.from_pymatgen_structure(cnf1.reconstruct())
    motif2 = FractionalMotif.from_pymatgen_structure(cnf2.reconstruct())
    return motif1.find_inverted_match(motif2, atol)

