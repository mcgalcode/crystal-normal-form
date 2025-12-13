from .lattice_neighbor_finder import LatticeNeighborFinder
from .motif_neighbor_finder import MotifNeighborFinder
from ..crystal_normal_form import CrystalNormalForm
import os
import numpy as np

class NeighborFinder():

    def __init__(self, point: CrystalNormalForm):
        self.point = point

    def find_neighbor_tuples(self) -> list[tuple]:
        """
        Find all neighbor tuples (lattice + motif).

        Single branching point: Either do everything in pure Rust or everything in Python.
        """
        use_rust = os.getenv("USE_RUST") is not None

        if use_rust:
            # Pure Rust path - single call does lattice + motif
            import rust_cnf

            vonorms_i32 = np.array(self.point.lattice_normal_form.vonorms.tuple, dtype=np.int32)
            coords_i32 = np.array(self.point.motif_normal_form.coord_list, dtype=np.int32)
            elements = [str(el) for el in self.point.motif_normal_form.elements]
            n_atoms = len(elements)
            xi = float(self.point.xi)
            delta = int(self.point.delta)

            return rust_cnf.find_neighbor_tuples_rust(
                vonorms_i32, coords_i32, elements, n_atoms, xi, delta
            )
        else:
            # Pure Python path - calls lattice + motif finders separately
            lnf_neighbor_finder = LatticeNeighborFinder(self.point)
            mnf_neighbor_finder = MotifNeighborFinder(self.point)

            lattice_neighbors = lnf_neighbor_finder.find_neighbor_tuples()
            mnf_neighbors = mnf_neighbor_finder.find_neighbor_tuples()

            return list(set(lattice_neighbors + mnf_neighbors))

    def find_neighbors(self) -> list[CrystalNormalForm]:
        lnf_neighbor_finder = LatticeNeighborFinder(self.point)
        mnf_neighbor_finder = MotifNeighborFinder(self.point)

        lattice_neighbors = lnf_neighbor_finder.find_cnf_neighbors()
        mnf_neighbors = mnf_neighbor_finder.find_motif_neighbors()


        all_neighbor_points = set()
        for lat_neighb in lattice_neighbors:
            all_neighbor_points.add(lat_neighb)

        for mot_neighb in mnf_neighbors.neighbors:
            all_neighbor_points.add(mot_neighb.point)

        return list(all_neighbor_points)
