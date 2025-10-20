import numpy as np
import os
import pytest

from .assertions import *
from .data import _ALL_MP_STRUCTURES, load_pathological_cifs, get_data_file_path, save_cnfs_to_dir, load_cnfs, save_cifs_to_dir, load_cifs
from cnf.motif.atomic_motif import FractionalMotif
from cnf.unit_cell import UnitCell
from cnf.linalg import MatrixTuple
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher

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

def parameterized_by_structs_with_num_sites_less_than(num):
    matcher = lambda struct: len(struct) < num
    return parameterized_by_matching_mp_structs(matcher)

def parameterized_by_matching_mp_structs(matcher):
    pairs = [(idx, struct) for idx, struct in enumerate(_ALL_MP_STRUCTURES) if matcher(struct)]
    pairs = pairs[::STRUCT_SAMPLE_FREQ]
    def _wrapper(func):
        return pytest.mark.parametrize("idx, struct", pairs)(func)    
    return _wrapper

def parameterized_by_mp_structs(func):
    if SPECIFIC_STRUCT_IDX is None:
        return pytest.mark.parametrize("idx, struct", zip(range(0, len(_ALL_MP_STRUCTURES), STRUCT_SAMPLE_FREQ), _ALL_MP_STRUCTURES[::STRUCT_SAMPLE_FREQ]))(func)
    else:
        return parameterized_by_mp_struct_idxs(idxs=[SPECIFIC_STRUCT_IDX])(func)

def parameterized_by_mp_struct_idxs(idxs=None, every=None):
    if idxs is not None and every is not None:
        raise RuntimeError("Don't supply both idxs and every to parameterized_by_mp_struct_idxs")
    
    if every is not None:
        idxs = list(range(0,len(_ALL_MP_STRUCTURES),every))
        idxs = idxs[::STRUCT_SAMPLE_FREQ]

    structs = [(idx, _ALL_MP_STRUCTURES[idx]) for idx in idxs]

    def _wrapper(func):
        return pytest.mark.parametrize("idx, struct", structs)(func)    
    return _wrapper

def printif(msg, flag):
    if flag:
        print(msg)

def are_cnfs_geo_matches(cnf1: CrystalNormalForm, cnf2: CrystalNormalForm, tol=1e-5):
    return are_unit_cells_geo_matches(UnitCell.from_cnf(cnf1), UnitCell.from_cnf(cnf2), tol=tol)

def are_structs_geo_matches(struct1: Structure, struct2: Structure, tol=1e-5):
    return are_unit_cells_geo_matches(UnitCell.from_pymatgen_structure(struct1), UnitCell.from_pymatgen_structure(struct2), tol=tol)

def _get_spga(struct):
    return SpacegroupAnalyzer(struct, symprec=0.0001, angle_tolerance=0.0001)
def does_struct_have_centrosymmetric_symm(struct):
    spga1 = SpacegroupAnalyzer(struct, symprec=0.0001, angle_tolerance=0.0001)
    return spga1.is_laue()

def is_struct_chiral(struct: Structure):
    spga1 = SpacegroupAnalyzer(struct, symprec=0.0001, angle_tolerance=0.0001)

    # cm1 = spga1.is_laue()
    # print(f"Struct 1 centrosymmetry: {cm1}")

    # point_group1 = spga1.get_point_group_symbol()
    # print(f"Point Group: {point_group1}.")
    # The get_symmetry_operations() method returns a list of symmetry operations.
    # Improper rotations have a rotation matrix determinant of -1. This is the
    # definitive test for chirality.
    symm_ops1 = spga1.get_symmetry_dataset()['rotations']
    has_improper1 = any(np.linalg.det(op) < 0 for op in symm_ops1)
    is_chiral1_general = not has_improper1
    return is_chiral1_general


def are_unit_cells_geo_matches(uc1: UnitCell, uc2: UnitCell, tol=1e-4):
    struct1 = uc1.to_pymatgen_structure()
    struct2 = uc2.to_pymatgen_structure()
    pdd_dist = pdd(struct1, struct2)
    if pdd_dist > tol:
        return False, f"PDD was {pdd_dist}"
    
    matcher = StructureMatcher(ltol=tol, stol=tol, angle_tol=tol, primitive_cell=False, attempt_supercell=False)
    if not matcher.fit(struct1, struct2):
        return False, f"Structure matcher suggests these are different, and pdd was {pdd_dist}"

    is_chiral1 = is_struct_chiral(struct1)
    is_chiral2 = is_struct_chiral(struct2)

    if not is_chiral1 and not is_chiral2:
        return True, f"Both structures are achiral, and pdd was {pdd_dist}"

    if is_chiral1 != is_chiral2:
        return False, f"Different chirality, struct 1: {is_chiral1}, struct 2: {is_chiral2}, and pdd was {pdd_dist}"
    
    # Both structures are chiral

    relation_mat = matcher.get_supercell_matrix(struct1, struct2)
    det = np.linalg.det(relation_mat)
    if det != +1:
        return False, f"Structures were related by mat with det {det}. {MatrixTuple(relation_mat)}, and pdd was {pdd_dist}"

    return True, f"Both structures are chiral, but have {pdd_dist} PDD and a matrix relation with positive det {det}"

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
        attempt_supercell=False
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

