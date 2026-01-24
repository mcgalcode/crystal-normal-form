from ..crystal_normal_form import CrystalNormalForm
from .utils import no_atoms_closer_than
from abc import ABC, abstractmethod
from pymatgen.core.structure import Structure
from ..calculation.grace import GraceCalculator

import os

class SearchFilter(ABC):

    def __init__(self):
        self.is_batch = False
        self.requires_structs = True

    @abstractmethod
    def should_add_pt(self, pt: CrystalNormalForm, struct: Structure) -> bool:
        pass

    def filter_nbs_cnf_only(self, nbs: list[CrystalNormalForm]):
        structs = [nb.reconstruct() for nb in nbs]
        valid_pts, _ = self.filter_nbs(nbs, structs)
        return valid_pts

    def filter_nbs(self, cnfs: list[CrystalNormalForm], structs: list[Structure]):
        valid_pts = []
        valid_structs = []
        for cnf, struct in zip(cnfs, structs):
            if self.should_add_pt(cnf, struct):
                valid_pts.append(cnf)
                valid_structs.append(struct)
        return valid_pts, valid_structs
    
class FilterSet():

    def __init__(self, filters: list[SearchFilter], use_structs = True):
        self.filters = filters
        self.use_structs = use_structs
        if not self.use_structs and any([f.requires_structs for f in self.filters]):
            raise RuntimeError("Tried to instantiate filterset with use_structs = False and at least one filter that requires structs!")
        
    def filter_cnfs(self, cnfs: list[CrystalNormalForm]):
        if self.use_structs:
            structs = [c.reconstruct() for c in cnfs]
        else:
            structs = [None for c in cnfs]
        return self._filter_cnfs_and_structs(cnfs, structs)
    
    def _filter_cnfs_and_structs(self, cnfs: list[CrystalNormalForm], structs: list[Structure]):
        valid_pts = cnfs
        valid_structs = structs
        for f in self.filters:
            valid_pts, valid_structs = f.filter_nbs(valid_pts, valid_structs)
        return valid_pts, valid_structs

class VolumeLimitFilter(SearchFilter):
    
    @classmethod
    def from_endpoint_structs(cls, endpoint_structs: list[Structure], low_ratio=0.8, high_ratio=1.2):
        volumes = [s.volume for s in endpoint_structs]
        min_vol = min(volumes) * low_ratio
        max_vol = max(volumes) * high_ratio
        return cls(min_vol, max_vol)
    
    @classmethod
    def from_cnf(cls, cnf: CrystalNormalForm, low_ratio=0.8, high_ratio=1.2):
        return cls.from_struct(cnf.reconstruct(), low_ratio, high_ratio)

    @classmethod
    def from_struct(cls, struct: Structure, low_ratio=0.8, high_ratio=1.2):
        min_vol = struct.volume * low_ratio
        max_vol = struct.volume * high_ratio
        return cls(min_vol, max_vol)

    def __init__(self, vol_lower_lim: float, vol_upper_lim: float):
        self.vll = vol_lower_lim
        self.vul = vol_upper_lim
        super().__init__()
    
    def should_add_pt(self, pt: CrystalNormalForm, struct: Structure):
        vol = struct.volume
        return (vol < self.vul and vol > self.vll)
    
class MinDistanceFilter(SearchFilter):

    def __init__(self, min_dist):
        self.dist = min_dist
        self._use_rust = os.getenv('USE_RUST') == '1'
        self.requires_structs = not self._use_rust

    def should_add_pt(self, pt: CrystalNormalForm, struct: Structure):
        return no_atoms_closer_than(pt, self.dist)
    
    def filter_nbs(self, cnfs, structs):
        if self._use_rust:
            import rust_cnf
            pt = cnfs[0]
            n_atoms = len(pt.elements)

            # Convert neighbors to list of lists (concatenated vonorms + coords)
            neighbor_list = [list(n.coords) for n in cnfs]

            # Filter in Rust (returns filtered list)
            filtered = rust_cnf.filter_neighbors_by_min_distance_rust(
                neighbor_list,
                n_atoms,
                pt.xi,
                pt.delta,
                self.dist
            )

            # Convert back to tuples
            return [CrystalNormalForm.from_tuple(n, pt.elements, pt.xi, pt.delta) for n in filtered], []
        else:
            return super().filter_nbs(cnfs, structs)
    
class EnergyFilter(SearchFilter):

    @classmethod
    def from_cnfs(cls, cnfs: list[CrystalNormalForm], tol: float = 3.0):
        calc = GraceCalculator()
        energies = [calc.calculate_energy(cnf) for cnf in cnfs]
        num_atoms = len(cnfs[0].elements)
        e_per_atoms = [e / num_atoms for e in energies]
        limit = (max(e_per_atoms) + tol) * num_atoms
        return cls(limit)


    def __init__(self, max_energy: float):
        self.max_energy = max_energy
        self.calc = GraceCalculator()

    def should_add_pt(self, pt, struct):
        return self.calc.calculate_energy(pt) < self.max_energy