import math
from pymatgen.core import Structure
from cnf import UnitCell, CrystalNormalForm

from typing import Union

Endpoint = Union[CrystalNormalForm, UnitCell, Structure]


def normalize_endpoint(endpoint: Endpoint) -> UnitCell:
    if isinstance(endpoint, CrystalNormalForm):
        return UnitCell.from_cnf(endpoint)
    elif isinstance(endpoint, Structure):
        return UnitCell.from_pymatgen_structure(endpoint)
    else:
        return endpoint

def get_endpoint_unit_cells(pt1: Endpoint, pt2: Endpoint, min_atoms: int = None):
    uc1 = normalize_endpoint(pt1)
    uc2 = normalize_endpoint(pt2)

    n_atoms1 = len(uc1)
    n_atoms2 = len(uc2)

    sc_idx1, sc_idx2 = calculate_supercell_indices(n_atoms1, n_atoms2, min_atoms)
    pt1_scs = uc1.supercells(sc_idx1)
    pt2_scs = uc2.supercells(sc_idx2)
    return pt1_scs, pt2_scs

def are_endpoints_compatible(endpt1: Endpoint, endpt2: Endpoint):
    if isinstance(endpt1, CrystalNormalForm) and isinstance(endpt2, CrystalNormalForm):
        return endpt1.xi == endpt2.xi and \
               endpt1.delta == endpt2.delta
    return True

def get_endpoint_cnfs(pt1: Endpoint, pt2: Endpoint, min_atoms: int = None, xi: float = None, delta: int = None):
    if not are_endpoints_compatible(pt1, pt2) and (xi is None or delta is None):
        raise ValueError("CNF Endpoints are incompatible (mismatch in xi, delta)")
    
    cnf_pts = [pt for pt in [pt1, pt2] if isinstance(pt, CrystalNormalForm)]
    if len(cnf_pts) > 0 and xi is None:
        xi = cnf_pts[0].xi
    
    if len(cnf_pts) > 0 and delta is None:
        delta = cnf_pts[0].delta
    
    if xi is None or delta is None:
        raise ValueError("Must provide either [xi and delta] or at least 1 endpoint in CNF form")
    
    pt1_ucs, pt2_ucs = get_endpoint_unit_cells(pt1, pt2, min_atoms)
    pt1_cnfs = list(set([uc.to_cnf(xi=xi, delta=delta) for uc in pt1_ucs]))
    pt2_cnfs = list(set([uc.to_cnf(xi=xi, delta=delta) for uc in pt2_ucs]))
    return pt1_cnfs, pt2_cnfs

def calculate_supercell_indices(start_atoms: int, end_atoms: int, min_atoms: int = None):
    """Calculate supercell indices needed to match atom counts.

    Returns:
        (start_index, end_index): Supercell indices for start and end structures
    """
    lcm = math.lcm(start_atoms, end_atoms)
    if min_atoms is None or lcm > min_atoms:
        return lcm / start_atoms, lcm / end_atoms
    else:
        lcm_multiple = math.ceil(min_atoms / lcm)
        lcm_above_min_atoms = lcm * lcm_multiple
        return lcm_above_min_atoms / start_atoms, lcm_above_min_atoms / end_atoms