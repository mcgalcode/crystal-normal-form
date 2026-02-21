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
