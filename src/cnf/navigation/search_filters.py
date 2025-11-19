from ..crystal_normal_form import CrystalNormalForm
from .utils import find_overlapping_atoms
from abc import ABC, abstractmethod
from pymatgen.core.structure import Structure

class SearchFilter(ABC):

    @abstractmethod
    def should_add_pt(self, pt: CrystalNormalForm, struct: Structure) -> bool:
        pass

class VolumeLimitFilter(SearchFilter):
    
    @classmethod
    def from_endpoint_structs(cls, endpoint_structs: list[Structure], low_ratio=0.8, high_ratio=1.2):
        volumes = [s.volume for s in endpoint_structs]
        min_vol = min(volumes) * low_ratio
        max_vol = max(volumes) * high_ratio
        return cls(min_vol, max_vol)

    def __init__(self, vol_lower_lim: float, vol_upper_lim: float):
        self.vll = vol_lower_lim
        self.vul = vol_upper_lim
    
    def should_add_pt(self, pt: CrystalNormalForm, struct: Structure):
        struct = pt.reconstruct()
        vol = struct.volume
        return (vol < self.vul and vol > self.vll)

class AtomOverlapFilter(SearchFilter):

    def __init__(self, overlap_tol=0.5):
        self.overlap_tol = overlap_tol
    
    def should_add_pt(self, pt: CrystalNormalForm, struct: Structure):
        overlaps = find_overlapping_atoms(struct, self.overlap_tol)
        return len(overlaps) == 0