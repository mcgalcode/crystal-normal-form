import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from .vonorm_list import VonormList
from .permutations import VonormPermutation
from .rounding import DiscretizedVonormComputer
from .utils import selling_reduce
from .superbasis import Superbasis

class LatticeNormalForm():

    @classmethod
    def from_pymatgen_structure(cls, structure: Structure, lattice_step_size: float, return_transforms=True):
        return cls.from_superbasis(Superbasis.from_pymatgen_structure(structure),
                                   lattice_step_size,
                                   return_transforms=return_transforms)

    @classmethod
    def from_pymatgen_lattice(cls, lattice: Lattice, lattice_step_size: float, return_transforms=True):
        return cls.from_superbasis(Superbasis.from_pymatgen_lattice(lattice),
                                   lattice_step_size,
                                   return_transforms=return_transforms)

    @classmethod
    def from_generating_vecs(cls, generating_vecs: np.array, lattice_step_size: float, return_transforms=True):
        return cls.from_superbasis(Superbasis.from_generating_vecs(generating_vecs),
                                   lattice_step_size,
                                   return_transforms=return_transforms)

    @classmethod
    def from_superbasis(cls, superbasis: Superbasis, lattice_step_size: float, return_transforms=True):
        return cls.from_vonorms(superbasis.compute_vonorms(),
                                lattice_step_size,
                                return_transforms=return_transforms)

    @classmethod
    def from_vonorms(cls, vonorms: VonormList, lattice_step_size: float, return_transforms=True):
        vonorms = VonormList(
            DiscretizedVonormComputer(vonorms.vonorms, lattice_step_size).find_closest_valid_vonorms()
        )

        reduced_vonorms, _, selling_transform_mat = selling_reduce(vonorms, return_transform_mat=True)
        conorms = reduced_vonorms.conorms

        permuted_vonorm_lists: list[tuple[VonormList, VonormPermutation]] = []
        for conorm_permutation in conorms.permissible_permutations:
            vonorm_permutation = conorm_permutation.to_vonorm_permutation()
            permuted_vlist = reduced_vonorms.apply_permutation(vonorm_permutation)
            permuted_vonorm_lists.append((permuted_vlist, vonorm_permutation))

        sorted_vlists = sorted(permuted_vonorm_lists, key=lambda pair: pair[0].vonorms, reverse=False)
        canonical_vonorm_list = sorted_vlists[0][0]
        stabilizer_permutations = [pair[1] for pair in permuted_vonorm_lists if pair[0] == canonical_vonorm_list]
        if return_transforms:
            return cls(canonical_vonorm_list, lattice_step_size), selling_transform_mat, stabilizer_permutations
        else:
            return cls(canonical_vonorm_list, lattice_step_size)
        
    def __init__(self, vonorm_list: VonormList, lattice_step_size: float):
        self.vonorms = vonorm_list
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