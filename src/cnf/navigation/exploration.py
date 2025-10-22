import json

from ..unit_cell import UnitCell
from pymatgen.core.structure import Structure
from .crystal_map import CrystalMap
from .neighbor_finder import NeighborFinder
from ..crystal_normal_form import CrystalNormalForm
from heapq import heappop, heappush, heapify
from sortedcontainers import SortedSet
import time

from pymatgen.core import Structure, Lattice
from typing import List, Tuple
from cnf.utils.pdd import pdd_for_cnfs

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


def get_endpoints_from_structs(struct1, struct2):
    uc1 = UnitCell.from_pymatgen_structure(struct1)
    uc2 = UnitCell.from_pymatgen_structure(struct2)
    
    if len(uc1) == len(uc2):
        return [uc1], [uc2]
    if len(uc1) > len(uc2):
        multiplier = len(uc1) / len(uc2)
        other_supercells = uc2.supercells(multiplier)
        return [uc1], other_supercells
    if len(uc2) > len(uc1):
        multiplier = len(uc2) / len(uc1)
        other_supercells = uc1.supercells(multiplier)
        return other_supercells, [uc2]

class CrystalExplorer():

    def __init__(self, cmap: CrystalMap, vol_lower_lim: float, vol_upper_lim: float, target_pts: list[CrystalNormalForm]):
        self.map = cmap
        self._is_explored = {}
        self._target_pts = target_pts
        self._target_structs = [pt.reconstruct() for pt in target_pts]
        self.vll = vol_lower_lim
        self.vul = vol_upper_lim
        self.scores = {}
        self._unexplored_pts = set()
        self.unexplored_score_list = SortedSet()
        self._explored_pts = set()
        self.explored_score_list = SortedSet()

        for pt in cmap.all_node_ids():
            self._unexplored_pts.add(pt)
            pt = cmap.get_point_by_id(pt)
            self.score_pt(pt)
    
    def explore_point(self, point_id: int):
        pt = self.map.get_point_by_id(point_id)
        nf = NeighborFinder(pt)
        nb_pts = nf.find_neighbors()
        new_ids = []
        for nb_pt in nb_pts:
            if self.should_add_pt(nb_pt):
                if nb_pt not in self.map:
                        nid = self.map.add_point(nb_pt)
                        new_ids.append(nid)
                        self._unexplored_pts.add(nid)
                        self.score_pt(nb_pt)
                self.map.add_connection(pt, nb_pt)                    
            else:
                pass
                # print("Skipping point outside valid bounds...")

        item = (self.scores[point_id], point_id)
        self._unexplored_pts.remove(point_id)
        self.unexplored_score_list.remove(item)

        self._explored_pts.add(point_id)
        self.explored_score_list.add(item)
        return new_ids
    
    def score_pt(self, pt: CrystalNormalForm):
        score = self.get_goodness(pt)
        id = self.map.get_point_id(pt)
        item = (score, id)
        self.scores[id] = score
        self.unexplored_score_list.add(item)
        return score

    def get_goodness(self, pt: CrystalNormalForm):
        pdd = min([pdd_for_cnfs(pt, target) for target in self._target_pts])
        return pdd
    
    def unexplored_points(self):
        return [i[1] for i in self.unexplored_score_list]

    def best_current_score(self):
        all_scores = self.unexplored_score_list.union(self.explored_score_list)
        if len(all_scores) == 0:
            return None
        else:
            return all_scores[0][0]
    
    def score_for_point(self, pt_id: int):
        return self.scores[pt_id]
    
    def should_add_pt(self, pt: CrystalNormalForm):
        # return True
        struct = pt.reconstruct()
        vol = struct.volume
        if not (vol < self.vul and vol > self.vll):
            return False
        # print(pt.to_discretized_motif())
        overlaps = find_overlapping_atoms(struct, 0.9)
        # print(overlaps)
        if len(overlaps) > 0:
            return False
        return True
    
    def is_point_explored(self, point: CrystalNormalForm):
        pid = self.map.get_point_id(point)
        return self.is_id_explored(pid)
    
    def is_id_explored(self, id: int):
        if id not in self.map.all_node_ids():
            raise ValueError(f"Tried to check if nonexistant node id {id} was explored")
        return self._is_explored[id]
    
    def to_json(self, fname: str):
        d = {
            "crystal_map": self.map.as_dict(),
            "vll": self.vll,
            "vul": self.vul,
            "scores": self.scores,
            "is_explored": list(self._is_explored),
            "target_pts": [pt.coords for pt in self._target_pts]
        }
        with open(fname, 'w+') as f:
            json.dump(d, f)
    