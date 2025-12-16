import numpy as np
from pymatgen.core import Structure
from typing import List, Tuple
from cnf import UnitCell, CrystalNormalForm

def get_endpoints_from_pmg_structs(struct1: Structure, struct2: Structure):
    uc1 = UnitCell.from_pymatgen_structure(struct1)
    uc2 = UnitCell.from_pymatgen_structure(struct2)
    return get_endpoints_from_unit_cells(uc1, uc2)

def get_endpoints_from_unit_cells(cell1: UnitCell, cell2: UnitCell):
    if len(cell1) == len(cell2):
        return [cell1], [cell2]
    if len(cell1) > len(cell2):
        multiplier = len(cell1) / len(cell2)
        other_supercells = cell2.supercells(multiplier)
        return [cell1], other_supercells
    if len(cell2) > len(cell1):
        multiplier = len(cell2) / len(cell1)
        other_supercells = cell1.supercells(multiplier)
        return other_supercells, [cell2]

def compute_pairwise_distance_matrix(cnf: CrystalNormalForm):
    return compute_pairwise_distances(cnf.reconstruct())

def compute_pairwise_distances(structure: Structure) -> np.ndarray:
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
    distances = compute_pairwise_distances(pt.reconstruct())
    # Get non-diagonal elements (distances between different atoms)
    non_diag_distances = distances[np.triu_indices_from(distances, k=1)]
    # Keep neighbors where all non-diagonal distances are > 1.4
    return (non_diag_distances > min_dist).all()
