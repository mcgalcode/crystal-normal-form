from .voronoi.vonorm_list import VonormList

class LatticeNormalForm():

    @classmethod
    def from_coords(cls, coords, lattice_step_size):
        vlist = VonormList(coords)
        return cls(vlist, lattice_step_size)

    def __init__(self, canonical_vonorm_list: VonormList, lattice_step_size: float):
        self.vonorms = canonical_vonorm_list
        self.lattice_step_size = lattice_step_size
    
    def to_superbasis(self):
        return self.vonorms.to_superbasis(self.lattice_step_size)
    
    @property
    def coords(self):
        return tuple([int(vo) for vo in self.vonorms.vonorms])
    
    def __repr__(self):
        return f"LatticeNormalForm(vonorms={self.vonorms.vonorms},step_size={self.lattice_step_size})"

    def __eq__(self, other: 'LatticeNormalForm'):
        return self.coords == other.coords and self.lattice_step_size == other.lattice_step_size
    
    def __hash__(self):
        return self.coords.__hash__()