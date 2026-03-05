import numpy as np
from pymatgen.core import Structure
from cnf import CrystalNormalForm, UnitCell
from typing import Union

def compute_pairwise_distances(structure: Union[Structure, CrystalNormalForm, UnitCell]) -> np.ndarray:
    """
    Compute pairwise distances between all atoms in a pymatgen structure
    using periodic boundary conditions.

    For each pair of atoms, checks all 27 periodic images to find the minimum distance.
    The 27 images come from: 3 directions (x,y,z) × 3 offsets per direction (-1,0,+1) = 3³ = 27

    Args:
        structure: A pymatgen Structure object

    Returns:
        A symmetric NxN distance matrix where N is the number of atoms,
        with distances in Angstroms
    """
    if isinstance(structure, CrystalNormalForm):
        structure = structure.reconstruct()
    elif isinstance(structure, UnitCell):
        structure = structure.to_pymatgen_structure()
    
    n_atoms = len(structure)
    distance_matrix = np.zeros((n_atoms, n_atoms))

    # Get lattice matrix for PBC calculations
    lattice_matrix = structure.lattice.matrix

    # Compute distances by checking all 27 periodic images
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            min_dist_sq = float('inf')

            # Check all 27 periodic images (-1, 0, +1 in each direction)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        # Compute offset from lattice vectors
                        offset = dx * lattice_matrix[0] + dy * lattice_matrix[1] + dz * lattice_matrix[2]

                        # Vector from atom i to atom j (with this periodic image)
                        diff = structure.cart_coords[j] + offset - structure.cart_coords[i]

                        # Compute squared distance
                        dist_sq = np.dot(diff, diff)
                        min_dist_sq = min(min_dist_sq, dist_sq)

            # Store the minimum distance
            distance = np.sqrt(min_dist_sq)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix

def no_atoms_closer_than(pt: CrystalNormalForm, min_dist: float):
    distances = compute_pairwise_distances(pt)
    # Get non-diagonal elements (distances between different atoms)
    non_diag_distances = distances[np.triu_indices_from(distances, k=1)]
    # Keep neighbors where all non-diagonal distances are > 1.4
    return (non_diag_distances > min_dist).all()


def compute_delta_for_step_size(
    structure: Union[Structure, CrystalNormalForm, UnitCell],
    target_step: float,
) -> int:
    """Compute delta needed to achieve a target physical step size.

    CNF's delta parameter divides fractional coordinates into delta intervals,
    so physical step size = lattice_param / delta. Different cell sizes with
    the same delta get different physical resolutions.

    This function computes delta = ceil(max_lattice_param / target_step),
    which ensures the physical step is at most target_step for all axes.

    Args:
        structure: A structure (pymatgen Structure, CNF, or UnitCell) to get
            lattice parameters from.
        target_step: Target physical step size in Angstroms. Smaller values
            give finer resolution but larger search spaces.

    Returns:
        Integer delta value that achieves step <= target_step.

    Example:
        >>> # For a cell with max lattice param 9.2 Å and target step 0.3 Å:
        >>> delta = compute_delta_for_step_size(structure, 0.3)
        >>> # delta = ceil(9.2 / 0.3) = 31
        >>> # Actual step = 9.2 / 31 ≈ 0.297 Å
    """
    if isinstance(structure, CrystalNormalForm):
        structure = structure.reconstruct()
    elif isinstance(structure, UnitCell):
        structure = structure.to_pymatgen_structure()

    lattice = structure.lattice
    max_param = max(lattice.a, lattice.b, lattice.c)
    return int(np.ceil(max_param / target_step))


def compute_delta_for_endpoints(
    start_uc: UnitCell,
    end_uc: UnitCell,
    target_step: float,
    min_atoms: int | None = None,
) -> int:
    """Compute delta for pathfinding between two unit cells.

    This is the correct way to compute delta when using min_atoms supercells.
    It first creates the supercells (if needed), then computes delta from the
    supercell lattice parameters to ensure consistent physical resolution.

    Args:
        start_uc: Starting UnitCell.
        end_uc: Ending UnitCell.
        target_step: Target physical step size in Angstroms.
        min_atoms: Minimum atoms (will create supercells if needed).

    Returns:
        Integer delta value that achieves step <= target_step for the actual
        cell sizes used in pathfinding.
    """
    from cnf.navigation.endpoints import get_endpoint_unit_cells

    start_scs, end_scs = get_endpoint_unit_cells(start_uc, end_uc, min_atoms=min_atoms)

    # Compute delta from ALL supercell variants and take the maximum
    all_deltas = []
    for sc in start_scs + end_scs:
        all_deltas.append(compute_delta_for_step_size(sc, target_step))

    return max(all_deltas)


def min_bond_length(structures: list[Union[Structure, CrystalNormalForm, UnitCell]]) -> float:
    """Compute minimum pairwise atomic distance across one or more structures.

    This is useful for determining a safe minimum distance filter that avoids
    unphysical atomic overlaps while still allowing the structures to be reached.

    Args:
        structures: List of structures (pymatgen Structure, CNF, or UnitCell).
            Can also pass a single structure (will be wrapped in a list).

    Returns:
        Minimum interatomic distance in Angstroms across all structures.
    """
    if not isinstance(structures, list):
        structures = [structures]

    min_dist = float('inf')
    for struct in structures:
        distances = compute_pairwise_distances(struct)
        # Get non-diagonal elements (distances between different atoms)
        non_diag = distances[np.triu_indices_from(distances, k=1)]
        if len(non_diag) > 0:
            min_dist = min(min_dist, non_diag.min())

    return min_dist
