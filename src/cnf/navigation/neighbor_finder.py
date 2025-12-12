from .lattice_neighbor_finder import LatticeNeighborFinder
from .motif_neighbor_finder import MotifNeighborFinder
from ..crystal_normal_form import CrystalNormalForm

class NeighborFinder():

    def __init__(self, point: CrystalNormalForm):
        self.point = point

    def find_neighbor_tuples(self) -> list[tuple]:
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
        