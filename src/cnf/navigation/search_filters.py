from ..crystal_normal_form import CrystalNormalForm
from .utils import no_atoms_closer_than
from ..utils.pdd import pdd_for_cnfs
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

    def add_filter(self, filter: SearchFilter):
        self.filters.append(filter)

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

    # Default ratio of shortest bond length to use as min_dist filter
    DEFAULT_BOND_RATIO = 0.75

    @classmethod
    def from_structures(cls, structures, ratio=None):
        """Create a MinDistanceFilter based on the shortest bond in the structures.

        Computes the minimum interatomic distance across all provided structures
        and uses a fraction of it as the filter threshold. This ensures the filter
        allows the endpoint structures while rejecting unphysical configurations
        with severe atomic overlap.

        Args:
            structures: List of structures (pymatgen Structure, CNF, or UnitCell).
            ratio: Fraction of shortest bond to use (default: 0.75).
                Lower values are more permissive.

        Returns:
            MinDistanceFilter configured with auto-computed min_dist.
        """
        from .utils import min_bond_length

        if ratio is None:
            ratio = cls.DEFAULT_BOND_RATIO

        shortest_bond = min_bond_length(structures)
        min_dist = shortest_bond * ratio
        return cls(min_dist)

    def __init__(self, min_dist):
        self.dist = min_dist
        self._use_rust = os.getenv('USE_RUST') == '1'
        self.requires_structs = not self._use_rust

    def should_add_pt(self, pt: CrystalNormalForm, struct: Structure):
        return no_atoms_closer_than(pt, self.dist)
    
    def filter_nbs(self, cnfs, structs):

        if self._use_rust:
            if len(cnfs) == 0:
                return cnfs, []
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

            # Convert back to CNFs, with None structs to maintain list length for chaining
            filtered_cnfs = [CrystalNormalForm.from_tuple(n, pt.elements, pt.xi, pt.delta) for n in filtered]
            return filtered_cnfs, [None] * len(filtered_cnfs)
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


    def __init__(self, max_energy: float, calc=None, cache=None):
        super().__init__()
        self.requires_structs = False
        self.max_energy = max_energy
        self.calc = calc or GraceCalculator()
        self._cache = cache if cache is not None else {}

    def should_add_pt(self, pt, struct):
        key = pt.coords
        if key not in self._cache:
            self._cache[key] = self.calc.calculate_energy(pt)
        return self._cache[key] < self.max_energy


class PDDCylinderFilter(SearchFilter):
    """Filter that constrains search to an ellipsoid in PDD space.

    Uses the triangle inequality: for any node N on the path between
    start S and goal G, we require:
        min_d(S, N) + min_d(N, G) <= min_d(S, G) * tolerance

    where min_d means minimum over all start/goal CNFs (since multiple
    supercell orientations may exist for each polymorph).

    When tolerance=1.0, only nodes on the geodesic are allowed.
    Higher tolerance values (e.g., 1.2, 1.5) allow nodes within
    an ellipsoid "buffer" around the direct path.
    """

    def __init__(
        self,
        start_cnfs: list[CrystalNormalForm],
        goal_cnfs: list[CrystalNormalForm],
        tolerance: float = 1.2,
        k: int = 100,
    ):
        super().__init__()
        self.requires_structs = False
        self.start_cnfs = start_cnfs
        self.goal_cnfs = goal_cnfs
        self.tolerance = tolerance
        self.k = k

        # Precompute direct distance: min over all (start, goal) pairs
        self.direct_distance = min(
            pdd_for_cnfs(s, g, k=k)
            for s in start_cnfs
            for g in goal_cnfs
        )
        self.max_path_distance = self.direct_distance * tolerance

        # Cache for PDD distances: key -> (min_dist_to_starts, min_dist_to_goals)
        self._cache: dict[tuple, tuple[float, float]] = {}

        # Pre-cache endpoint distances
        for start in start_cnfs:
            self._cache[start.coords] = (0.0, self._min_dist_to_goals(start))
        for goal in goal_cnfs:
            self._cache[goal.coords] = (self._min_dist_to_starts(goal), 0.0)

    def _min_dist_to_starts(self, cnf: CrystalNormalForm) -> float:
        """Compute minimum PDD distance from cnf to any start CNF."""
        return min(pdd_for_cnfs(cnf, s, k=self.k) for s in self.start_cnfs)

    def _min_dist_to_goals(self, cnf: CrystalNormalForm) -> float:
        """Compute minimum PDD distance from cnf to any goal CNF."""
        return min(pdd_for_cnfs(cnf, g, k=self.k) for g in self.goal_cnfs)

    def _get_distances(self, cnf: CrystalNormalForm) -> tuple[float, float]:
        """Get (min_dist_to_starts, min_dist_to_goals) for a CNF, using cache."""
        key = cnf.coords
        if key not in self._cache:
            dist_to_starts = self._min_dist_to_starts(cnf)
            dist_to_goals = self._min_dist_to_goals(cnf)
            self._cache[key] = (dist_to_starts, dist_to_goals)
        return self._cache[key]

    def should_add_pt(self, pt: CrystalNormalForm, struct: Structure) -> bool:
        dist_to_starts, dist_to_goals = self._get_distances(pt)
        path_distance = dist_to_starts + dist_to_goals
        return path_distance <= self.max_path_distance

    @property
    def cache_size(self) -> int:
        """Return the number of cached PDD distances."""
        return len(self._cache)