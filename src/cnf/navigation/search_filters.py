from ..crystal_normal_form import CrystalNormalForm
from .utils import find_overlapping_atoms, no_atoms_closer_than
from abc import ABC, abstractmethod
from pymatgen.core.structure import Structure


class SearchFilter(ABC):

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

    def __init__(self, filters: list[SearchFilter]):
        self.filters = filters
    
    def filter_cnfs(self, cnfs: list[CrystalNormalForm]):
        structs = [c.reconstruct() for c in cnfs]
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
    
    def should_add_pt(self, pt: CrystalNormalForm, struct: Structure):
        vol = struct.volume
        return (vol < self.vul and vol > self.vll)
    
class MinDistanceFilter(SearchFilter):

    def __init__(self, min_dist):
        self.dist = min_dist
    
    def should_add_pt(self, pt: CrystalNormalForm, struct: Structure):
        return no_atoms_closer_than(pt, self.dist)