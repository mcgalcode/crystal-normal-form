import numpy as np
import os
import pytest

from .assertions import *
from .data import _ALL_MP_STRUCTURES, load_pathological_cifs, get_data_file_path, save_cnfs_to_dir, load_cnfs
from cnf.motif.atomic_motif import FractionalMotif
from cnf.unit_cell import UnitCell

IS_FAST = int(os.getenv("CNF_FAST_TEST", 0)) == 1

STRUCT_SAMPLE_FREQ = int(os.getenv("SSF", 10))

SPECIFIC_STRUCT_IDX = os.getenv("SSI")
if SPECIFIC_STRUCT_IDX is not None:
    SPECIFIC_STRUCT_IDX = int(SPECIFIC_STRUCT_IDX)

def skip_if_fast(func):
    return pytest.mark.skipif(IS_FAST, reason="Skipped because CNF_FAST_TEST env var was set to 1")(func)

def ALL_MP_STRUCTURES(freq=None):
    if freq is None:
        freq = STRUCT_SAMPLE_FREQ
    return _ALL_MP_STRUCTURES[::freq]

def parameterized_by_mp_structs(func):
    if SPECIFIC_STRUCT_IDX is None:
        return pytest.mark.parametrize("idx, struct", zip(range(0, len(_ALL_MP_STRUCTURES), STRUCT_SAMPLE_FREQ), _ALL_MP_STRUCTURES[::STRUCT_SAMPLE_FREQ]))(func)
    else:
        return parameterized_by_mp_struct_idxs([SPECIFIC_STRUCT_IDX])(func)

def parameterized_by_mp_struct_idxs(idxs):
    structs = [_ALL_MP_STRUCTURES[i] for i in idxs]
    def _wrapper(func):
        return pytest.mark.parametrize("idx, struct", zip(idxs, structs))(func)    
    return _wrapper

def printif(msg, flag):
    if flag:
        print(msg)

def are_cnfs_geo_matches(cnf1: CrystalNormalForm, cnf2: CrystalNormalForm, tol=1e-5):
    return are_geo_matches(UnitCell.from_cnf(cnf1), UnitCell.from_cnf(cnf2), tol=tol)

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
        # s_transform = get_structure_transformation(struct1, struct2, ltol=0.1, stol=0.01, angle_tol=1.0)
        # if s_transform["determinant"] == -1.0:
            # return False
        # else:
        #     return True
        are_inversions = uc1.motif.find_inverted_match(uc2.motif, atol=tol)
        if are_inversions:
            return False
        else:
            return True

def are_cnfs_mirror_images(cnf1: CrystalNormalForm, cnf2: CrystalNormalForm, atol=1e-6):
    motif1 = FractionalMotif.from_pymatgen_structure(cnf1.reconstruct())
    motif2 = FractionalMotif.from_pymatgen_structure(cnf2.reconstruct())
    return motif1.find_inverted_match(motif2, atol)

def get_structure_transformation(struct1, struct2, ltol=0.01, stol=0.01, angle_tol=0.1):
    """
    Get the transformation matrix relating two structures using pymatgen's StructureMatcher.

    Returns a dict with:
        - 'match': bool, whether structures match
        - 'supercell_matrix': 3x3 array, the transformation matrix (if match=True)
        - 'translation': 3-array, translation vector (if match=True)
        - 'mapping': list, atom index mapping (if match=True)
        - 'determinant': float, determinant of transformation matrix (if match=True)

    Args:
        struct1, struct2: pymatgen Structure objects or CNFs
        ltol, stol, angle_tol: tolerances for StructureMatcher
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from cnf import CrystalNormalForm

    # Handle CNF inputs
    if isinstance(struct1, CrystalNormalForm):
        struct1 = struct1.reconstruct()
    if isinstance(struct2, CrystalNormalForm):
        struct2 = struct2.reconstruct()

    matcher = StructureMatcher(
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        primitive_cell=False,
        attempt_supercell=True
    )

    result = {'match': matcher.fit(struct1, struct2)}

    if result['match']:
        try:
            supercell_matrix, translation, mapping = matcher.get_transformation(struct1, struct2)
            result['supercell_matrix'] = supercell_matrix
            result['translation'] = translation
            result['mapping'] = mapping
            result['determinant'] = np.linalg.det(supercell_matrix)
        except Exception as e:
            result['error'] = str(e)

    return result

