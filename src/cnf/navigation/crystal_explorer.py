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

    def __init__(self, cmap: CrystalMap, vol_lower_lim: float, vol_upper_lim: float, target_pts: list[CrystalNormalForm], skip_scoring=False, preload_scores: dict = None):
        self.map = cmap
        self._target_pts = target_pts
        self._target_structs = [pt.reconstruct() for pt in target_pts]
        self.vll = vol_lower_lim
        self.vul = vol_upper_lim

        # Track all points (scored or not) and their exploration state
        self._unexplored_pts = set()
        self._explored_pts = set()

        # Track scores (only for points that have been scored)
        if preload_scores is None:
            self.scores = {}
        else:
            self.scores = preload_scores

        self.unexplored_score_list = SortedSet()
        self.explored_score_list = SortedSet()

        if not skip_scoring:
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
                        self.score_pt(nb_pt, explored=False)
                self.map.add_connection(pt, nb_pt)                    
            else:
                pass
                # print("Skipping point outside valid bounds...")

        self._set_pt_explored(point_id)
        return new_ids
    
    def score_pt(self, pt: CrystalNormalForm, explored=False):
        score = self.get_goodness(pt)
        pt_id = self.map.get_point_id(pt)
        self._add_scored_pt(pt_id, score, explored)
        return score
    
    def _add_scored_pt(self, pt_id, score, explored):
        if explored:
            self._set_pt_explored(pt_id)
        else:
            self._set_pt_unexplored(pt_id)
        self.set_pt_score(pt_id, score)
    
    def set_pt_score(self, pt_id, score):
        if self.is_id_explored(pt_id):
            if pt_id in self.scores:
                self.explored_score_list.remove(self._get_score_item(pt_id))
            self.scores[pt_id] = score
            self.explored_score_list.add(self._get_score_item(pt_id))
        else:
            if pt_id in self.scores:
                self.unexplored_score_list.remove(self._get_score_item(pt_id))
            self.scores[pt_id] = score
            self.unexplored_score_list.add(self._get_score_item(pt_id))
        
    def _get_score_item(self, pt_id):
        return (self.scores[pt_id], pt_id)

    def _set_pt_explored(self, pt_id):
        if pt_id in self._unexplored_pts:
            self._unexplored_pts.remove(pt_id)
            if pt_id in self.scores:
                self.unexplored_score_list.remove(self._get_score_item(pt_id))

        self._explored_pts.add(pt_id)
        if pt_id in self.scores:
            self.explored_score_list.add(self._get_score_item(pt_id))
    
    def _set_pt_unexplored(self, pt_id):
        if pt_id in self._explored_pts:
            self._explored_pts.remove(pt_id)
            if pt_id in self.scores:
                self.explored_score_list.remove(self._get_score_item(pt_id))

        self._unexplored_pts.add(pt_id)
        if pt_id in self.scores:
            self.unexplored_score_list.add(self._get_score_item(pt_id))

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
        return id in self._explored_pts
    
    @classmethod
    def from_dict(cls, d):
        cmap = CrystalMap.from_dict(d["crystal_map"])
        xi = cmap.xi
        delta = cmap.delta
        elements = cmap.element_list
        target_pts = [CrystalNormalForm.from_tuple(tuple(l), elements, xi, delta) for l in d["target_pts"]]
        scores = { int(nid): score for nid, score in d["scores"].items() }
        e = cls(
            cmap,
            vol_lower_lim=d["vll"],
            vol_upper_lim=d["vul"],
            target_pts=target_pts,
            skip_scoring=True,
            preload_scores=scores
        )
        for nid in cmap.all_node_ids():
            e._set_pt_unexplored(nid)
        print("Finished setting unexplored")
        for nid in d["explored_ids"]:
            # print(f"{nid} in is_explored")
            e._set_pt_explored(nid)
        print("Finished setting explored")

        return e
    
    @classmethod
    def from_json(cls, fname):
        with open(fname, 'r+') as f:
            d = json.load(f)
            return cls.from_dict(d)


    def to_dict(self):
        return {
            "crystal_map": self.map.as_dict(),
            "vll": self.vll,
            "vul": self.vul,
            "scores": self.scores,
            "explored_ids": list(self._explored_pts),
            "target_pts": [pt.coords for pt in self._target_pts]
        }
        
    def to_json(self, fname: str):
        d = self.to_dict()
        with open(fname, 'w+') as f:
            json.dump(d, f)
    