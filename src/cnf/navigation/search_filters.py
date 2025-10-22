from ..crystal_normal_form import CrystalNormalForm
from .utils import find_overlapping_atoms
from abc import ABC, abstractmethod

class SearchFilter(ABC):

    @abstractmethod
    def should_add_pt(self, pt: CrystalNormalForm) -> bool:
        pass

class SimpleVolumeAndOverlapFilter(SearchFilter):

    def __init__(self, vol_lower_lim: float, vol_upper_lim: float, overlap_tol=0.5):
        self.vll = vol_lower_lim
        self.vul = vol_upper_lim
        self.overlap_tol = overlap_tol
    
    def should_add_pt(self, pt: CrystalNormalForm):
        struct = pt.reconstruct()
        vol = struct.volume
        if not (vol < self.vul and vol > self.vll):
            return False
        overlaps = find_overlapping_atoms(struct, 0.9)
        if len(overlaps) > 0:
            return False
        return True