from pymatgen.core.structure import Structure
from pymatgen.core import Structure
from typing import List, Tuple


def are_atoms_overlapping(
    structure: Structure,
    index1: int,
    index2: int,
    tolerance: float = 1.0
) -> bool:
    """
    Checks if two specific atoms in a pymatgen Structure object are overlapping.

    Overlap is defined as the interatomic distance being less than the sum
    of the covalent radii, scaled by a tolerance factor. This function
    correctly handles periodic boundary conditions.

    Args:
        structure (Structure): The pymatgen Structure object to analyze.
        index1 (int): The index of the first atom in the structure.
        index2 (int): The index of the second atom in the structure.
        tolerance (float): A tolerance factor for the overlap check.
                           Defaults to 1.0. If the distance is less than
                           tolerance * (radius1 + radius2), the atoms
                           are considered to be overlapping. A tolerance
                           < 1.0 is a stricter check, while > 1.0 is looser.

    Returns:
        bool: True if the atoms are overlapping, False otherwise.
        
    Raises:
        IndexError: If either index1 or index2 is out of bounds.
        ValueError: If the covalent radius for one of the atoms is not defined.
    """
    # --- 1. Validate Inputs ---
    num_sites = len(structure)
    if not (0 <= index1 < num_sites and 0 <= index2 < num_sites):
        raise IndexError("Atom index is out of the structure's range.")
    
    if index1 == index2:
        # An atom cannot overlap with itself.
        return False

    # --- 2. Get Atomic Radii and Distance ---
    site1 = structure[index1]
    site2 = structure[index2]

    # Get the distance between the two sites, accounting for periodicity
    distance = structure.get_distance(index1, index2)
    # print(distance)

    # Get the radii of the elements for the two sites
    radius1 = site1.specie.atomic_radius
    radius2 = site2.specie.atomic_radius
    # print(radius1, radius2)
    
    if radius1 is None or radius2 is None:
        raise ValueError(
            f"Radius not defined for element {site1.specie} or "
            f"{site2.specie}. Cannot perform overlap check."
        )

    # --- 3. Perform Overlap Check ---
    # Check if the actual distance is less than the sum of radii (scaled by tolerance)
    return distance < tolerance * (radius1 + radius2)


def find_overlapping_atoms(
    structure: Structure,
    tolerance: float = 1.0
) -> List[Tuple[int, int]]:
    """
    Finds all pairs of overlapping atoms in a pymatgen Structure.

    This function iterates through all unique pairs of atoms and uses the
    are_atoms_overlapping function to check for overlaps.

    Args:
        structure (Structure): The pymatgen Structure object to analyze.
        tolerance (float): A tolerance factor for the overlap check.
                           See are_atoms_overlapping for details. Defaults to 1.0.

    Returns:
        List[Tuple[int, int]]: A list of tuples, where each tuple contains
                               the indices of a pair of overlapping atoms.
                               Returns an empty list if no overlaps are found.
    """
    overlapping_pairs = []
    num_sites = len(structure)
    for i in range(num_sites):
        for j in range(i + 1, num_sites):
            if are_atoms_overlapping(structure, i, j, tolerance):
                overlapping_pairs.append((i, j))
    return overlapping_pairs
