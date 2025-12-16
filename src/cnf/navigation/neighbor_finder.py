from .lattice_neighbor_finder import LatticeNeighborFinder
from .motif_neighbor_finder import MotifNeighborFinder
from ..crystal_normal_form import CrystalNormalForm
from ..utils.config import should_use_rust

import os
import numpy as np

from typing import Union

class NeighborFinder():

    @classmethod
    def from_cnf(cls, pt: CrystalNormalForm):
        return cls(pt.xi, pt.delta, pt.elements)

    def __init__(self, xi: float, delta: int, elements: list[str]):
        self.xi = xi
        self.delta = delta
        self.elements = elements
        self.lattice_neighbor_finder = LatticeNeighborFinder(xi, delta, elements)
        self.motif_neighbor_finder = MotifNeighborFinder(xi, delta, elements)
    
    def _normalize_point(self, point: Union[tuple, CrystalNormalForm]) -> tuple:
        if isinstance(point, CrystalNormalForm):
            return point.coords
        else:
            return point
    
    def _tuples_to_cnfs(self, tuples: list[tuple]) -> list[CrystalNormalForm]:
        return [CrystalNormalForm.from_tuple(t, self.elements, self.xi, self.delta) for t in tuples]
    
    def find_neighbor_tuples(self, point: Union[tuple, CrystalNormalForm]) -> list[tuple]:
        """
        Find all neighbor tuples (lattice + motif).

        Single branching point: Either do everything in pure Rust or everything in Python.
        """
        use_rust = should_use_rust()

        point = self._normalize_point(point)

        if use_rust:
            # Pure Rust path - single call does lattice + motif
            import rust_cnf

            vonorms_i32 = np.array(point[:7], dtype=np.int32)
            coords_i32 = np.array(point[7:], dtype=np.int32)
            n_atoms = len(self.elements)

            tuples = rust_cnf.find_neighbor_tuples_rust(
                vonorms_i32, coords_i32, self.elements, n_atoms, self.xi, self.delta
            )
            return list(set([tuple([*pt[0], *pt[1]]) for pt in tuples]))
        else:
            # Pure Python path - calls lattice + motif finders separately

            lattice_neighbors = self.find_lattice_neighbors(point)
            mnf_neighbors = self.find_motif_neighbors(point)

            return list(set(lattice_neighbors + mnf_neighbors))
        

    def find_neighbors(self, point: Union[tuple, CrystalNormalForm]) -> list[CrystalNormalForm]:
        tuples = self.find_neighbor_tuples(point)
        return self._tuples_to_cnfs(tuples)
    
    def find_lattice_neighbors(self, cnf_tuple: tuple):
        return list(set(self.lattice_neighbor_finder.find_neighbor_tuples(cnf_tuple)))
    
    def find_motif_neighbors(self, cnf_tuple: tuple):
        return list(set(self.motif_neighbor_finder.find_neighbor_tuples(cnf_tuple)))
    
    def find_lattice_neighbor_cnfs(self, point: Union[tuple, CrystalNormalForm]):
        point = self._normalize_point(point)
        return self._tuples_to_cnfs(set(self.lattice_neighbor_finder.find_neighbor_tuples(point)))

    def find_motif_neighbor_cnfs(self, point: Union[tuple, CrystalNormalForm]):
        point = self._normalize_point(point)
        return self._tuples_to_cnfs(set(self.motif_neighbor_finder.find_neighbor_tuples(point)))
