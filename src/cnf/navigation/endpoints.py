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

def get_endpoint_cnfs(pt1: Endpoint, pt2: Endpoint, xi: float = None, delta: int = None, min_atoms: int = None):
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

def get_endpoint_cnfs_with_resolution(
    pt1: Endpoint,
    pt2: Endpoint,
    xi: float,
    delta: int | None = None,
    atom_step_length: float | None = None,
    min_delta: int | None = None,
    min_atoms: int | None = None,
) -> tuple[list, list, int]:
    """Get CNF endpoints with correct delta for the actual cell size.

    This is the correct way to get CNF endpoints when using min_atoms supercells
    and atom_step_length. It ensures delta is computed from the supercell lattice
    parameters, not the primitive cell.

    Why this matters:
        CNF's delta parameter divides fractional coordinates into delta intervals.
        Physical step size = lattice_param / delta. When using min_atoms to create
        supercells, the lattice parameters increase, so the same delta gives a
        coarser physical resolution. This function computes delta from the actual
        supercell size to maintain consistent physical resolution.

    Args:
        pt1: Starting endpoint (CNF, UnitCell, or Structure).
        pt2: Ending endpoint (CNF, UnitCell, or Structure).
        xi: Lattice discretization parameter.
        delta: Explicit motif discretization parameter. If provided, used directly
            (ignoring atom_step_length). Use this when you want exact control.
        atom_step_length: Target physical step size in Angstroms. Used to compute
            delta if delta is not provided. The computed delta ensures the actual
            step size is at most this value for all lattice axes.
        min_delta: Minimum allowed delta. If the computed delta (from atom_step_length)
            is smaller than this, min_delta is used instead. Useful when a previous
            search phase established a minimum resolution requirement.
        min_atoms: Minimum atoms in the working cell. If the LCM of the two
            structures is less than this, supercells will be created.

    Returns:
        (start_cnfs, goal_cnfs, delta) - the CNF endpoints and the delta used.

    Raises:
        ValueError: If neither delta nor atom_step_length is provided.

    Example:
        # Basic usage with atom_step_length
        start_cnfs, goal_cnfs, delta = get_endpoint_cnfs_with_resolution(
            start_uc, end_uc, xi=1.5, atom_step_length=0.1
        )

        # With min_atoms supercells (delta computed from supercell size)
        start_cnfs, goal_cnfs, delta = get_endpoint_cnfs_with_resolution(
            start_uc, end_uc, xi=1.5, atom_step_length=0.1, min_atoms=24
        )

        # With minimum delta from previous search phase
        start_cnfs, goal_cnfs, delta = get_endpoint_cnfs_with_resolution(
            start_uc, end_uc, xi=1.5, atom_step_length=0.1, min_delta=50
        )
    """
    from cnf.navigation.utils import compute_delta_for_endpoints

    if delta is None and atom_step_length is None:
        raise ValueError("Must provide either delta or atom_step_length")

    # Normalize to UnitCells
    uc1 = normalize_endpoint(pt1)
    uc2 = normalize_endpoint(pt2)

    # Compute delta from supercell lattice parameters if not provided
    if delta is None:
        delta = compute_delta_for_endpoints(uc1, uc2, atom_step_length, min_atoms)

    # Apply minimum delta constraint if specified
    if min_delta is not None and delta < min_delta:
        delta = min_delta

    # Get CNFs (this also handles supercells internally)
    pt1_cnfs, pt2_cnfs = get_endpoint_cnfs(pt1, pt2, xi=xi, delta=delta, min_atoms=min_atoms)

    return pt1_cnfs, pt2_cnfs, delta


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