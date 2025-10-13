import numpy as np

from ..crystal_normal_form import CrystalNormalForm
from ..cnf_constructor import CNFConstructor, CNFConstructionResult
from ..lattice import LatticeNormalForm
from ..lattice.lnf_constructor import LatticeNormalFormConstructor, LatticeNormalFormConstructionResult
from ..motif.atomic_motif import DiscretizedMotif
from ..lattice.voronoi import VonormList
from ..lattice.permutations import PermutationMatrix

def is_primary_idx(idx):
    return idx >= 0 and idx < 4

def is_secondary_idx(idx):
    return idx >= 4 and idx < 7

class LatticeStep():

    @classmethod
    def all_step_vecs(cls):
        steps = []
        for first_idx in range(7):
            for second_idx in range(first_idx + 1, 7):
                vec = np.zeros(7)
                vec[first_idx] = 1
                if is_primary_idx(first_idx) and is_primary_idx(second_idx):
                    vec[second_idx] = -1

                if is_primary_idx(first_idx) and is_secondary_idx(second_idx):
                    vec[second_idx] = 1

                if is_secondary_idx(first_idx):
                    vec[second_idx] = -1
                
                steps.append([int(v) for v in vec])
                steps.append([-int(v) for v in vec])
        
        return steps

    def __init__(self, vals, prereq_perm: PermutationMatrix = None):
        self.vals = vals
        self.tuple = tuple(vals)
        self.prereq_perm = prereq_perm

        for idx, val in enumerate(vals):
            if np.abs(val) != 1 and val != 0:
                raise ValueError(f"LatticeStep instantiated with invalid element != 1 at pos {idx}: {tuple(vals)}")
    
    def __eq__(self, other: 'LatticeStep'):
        return self.tuple == other.tuple and self.prereq_perm == other.prereq_perm
    
    def __hash__(self):
        return (self.tuple, self.prereq_perm).__hash__()
    
    def __repr__(self):
        return f"LatticeStep<vonorm_adj={self.vals}, perm={self.prereq_perm.vonorm_permutation.perm}>"
    
    def print_details(self):
        print(f"Step adj. vec: {self.vals}")
        print(f"Prerequisite Vo. perm: {self.prereq_perm.vonorm_permutation}")
        print(f"Prerequisite Co. perm: {self.prereq_perm.conorm_permutation}")
        print(f"Prerequisite matrix: {self.prereq_perm.matrix.tuple}")
        print(f"Prerequisite matrix det: {self.prereq_perm.matrix.determinant()}")        

class LatticeStepResult():

    def __init__(self,
                 step: LatticeStep,
                 adjusted_vonorms: VonormList,
                 construction_result: CNFConstructionResult | LatticeNormalFormConstructionResult,
                 result: LatticeNormalForm | CrystalNormalForm,
                 adjusted_motif: DiscretizedMotif = None):
        self.step = step
        self.adjusted_vonorms = adjusted_vonorms
        self.adjusted_motif = adjusted_motif
        self.construction_result = construction_result
        self.result = result
    
    def print_details(self):
        print(f"Applied step:")
        self.step.print_details()
        print()
        print(f"Got new Vonorm List:")
        print(self.adjusted_vonorms)
        print()
        self.construction_result.print_details()
    
class LatticeNeighbor():

    def __init__(self,
                 lnf: LatticeNormalForm | CrystalNormalForm,
                 step_results: list[LatticeStepResult] = None):
        self.point = lnf
        if step_results is not None:
            self.step_results = step_results
        else:
            self.step_results = []
        
    def add_step(self, step_result: LatticeStepResult):
        self.step_results.append(step_result)

    def __eq__(self, other: 'LatticeNeighbor'):
        return self.point == other.point
    
    def __hash__(self):
        return self.point.__hash__()

class LatticeNeighborSet():

    def __init__(self):
        self.neighbors_to_steps: dict[CrystalNormalForm | LatticeNormalForm, list[LatticeStepResult]] = {}
    
    def add_neighbor(self, step_result: LatticeStepResult):
        nb_point = step_result.result
        if nb_point in self.neighbors_to_steps:
            self.neighbors_to_steps[nb_point].append(step_result)
        else:
            self.neighbors_to_steps[nb_point] = [step_result]
    
    @property
    def neighbors(self) -> list[LatticeNeighbor | CrystalNormalForm]:
        nbs = []
        for nb, step_results in self.neighbors_to_steps.items():
            nbs.append(LatticeNeighbor(nb, step_results))
        return nbs
    
    def steps_for_neighbor(self, neighbor: LatticeNormalForm | CrystalNormalForm):
        no_steps = []
        return self.neighbors_to_steps.get(neighbor, [])
    
    def get_neighbor(self, nb_tuple: tuple):
        matching = [n for n in self.neighbors if n.coords == nb_tuple]
        if len(matching) == 1:
            return matching[0]
        else:
            return None
    
    def __contains__(self, item: LatticeNormalForm | CrystalNormalForm):
        if not (isinstance(item, LatticeNormalForm) or isinstance(item, CrystalNormalForm)):
            raise ValueError(f"Can't tell if type {type(item)} is in LatticeNeighborSet")

        return item in [n.point for n in self.neighbors]
    
    def __len__(self):
        return len(self.neighbors_to_steps)

class LatticeNeighborFinder():

    def __init__(self, cnf_point: CrystalNormalForm, verbose_logging=False):
        self.verbose_logging = verbose_logging
        self.point = cnf_point

        if self._is_cnf_neighbor_finder():
            self.discretized_motif = cnf_point.basis_normal_form.to_discretized_motif()
            self.fractional_motif = cnf_point.basis_normal_form.to_motif()

    def _is_cnf_neighbor_finder(self):
        return isinstance(self.point, CrystalNormalForm)
    
    def _lnf(self):
        if self._is_cnf_neighbor_finder():
            return self.point.lattice_normal_form
        else:
            return self.point

    def _log(self, msg):
        if self.verbose_logging:
            print(msg)

    def possible_steps(self):
        vonorms = self._lnf().vonorms
        grouped_vnorm_perms = vonorms.conorms.form.grouped_vonorm_permutations()
        steps: list[LatticeStep] = []
        for _, perms in grouped_vnorm_perms.items():
            # Choose a representative perm - any of these will work
            # because the all_step_vecs chi vectors from David's thesis
            # are already designed to cover S4. So we just need any representative
            # that gets us to each of Kurlin's basis possibilities.
            repr_perm = perms[0]
            for step_vec in LatticeStep.all_step_vecs():
                steps.append(LatticeStep(step_vec, repr_perm))
        return steps

    def get_vonorm_neighbor(self, step: LatticeStep):
        lnf_point = self._lnf()
        vonorms = lnf_point.vonorms
        self._log(f"Original vonorms: {vonorms}")

        permuted_vonorms = vonorms.apply_permutation(step.prereq_perm.vonorm_permutation)
        self._log(f"Permuted vonorms: {permuted_vonorms} (after applying {step.prereq_perm.vonorm_permutation})")
        old_vonorms = np.array(permuted_vonorms.vonorms)
        
        new_vonorms = VonormList(tuple([int(v) for v in old_vonorms + np.array(step.vals)]))
        self._log(f"Computed neighbor vonorms before canonicalization: {new_vonorms}")

        if new_vonorms.is_obtuse() and new_vonorms.is_superbasis():
            return new_vonorms
        else:
            if self.verbose_logging:
                if not new_vonorms.is_obtuse():
                    self._log(f"Neighbor was not obtuse.")

                if not new_vonorms.is_superbasis():
                    self._log(f"Neighbor was not a superbasis.")
            return None
    
    def get_basis_neighbor(self, step: LatticeStep):
        matrix = step.prereq_perm.matrix
        return self.discretized_motif.apply_unimodular(matrix)

    def find_lnf_neighbor(self, step: LatticeStep):
        new_vonorms = self.get_vonorm_neighbor(step)
        if new_vonorms is None:
            return None
        
        lnf_constructor = LatticeNormalFormConstructor(self._lnf().lattice_step_size)
        construction_result = lnf_constructor.build_lnf_from_discretized_vonorms(new_vonorms, skip_reduction=True)
        neighbor_lnf = construction_result.lnf
            
        return LatticeStepResult(step, new_vonorms, construction_result, neighbor_lnf)
        

    def find_lnf_neighbors(self) -> LatticeNeighborSet:
        neighbors = LatticeNeighborSet()
        for step in self.possible_steps():
            result = self.find_lnf_neighbor(step)
            if result is not None:
                neighbors.add_neighbor(result)
        return neighbors
    
    def find_cnf_neighbor(self, step: LatticeStep) -> LatticeStepResult | None:
        neighbor_vonorms = self.get_vonorm_neighbor(step)
        if neighbor_vonorms is None:
            return None
        
        
        cnf_results: list[CrystalNormalForm] = []
        for mat in step.prereq_perm.all_matrices:
            neighbor_motif = self.discretized_motif.apply_unimodular(mat)
            cnf_constructor = CNFConstructor(
                self.point.xi,
                self.point.delta,
                verbose_logging=self.verbose_logging,
            )

            cnf_results.append(cnf_constructor.from_discretized_vonorms_and_motif(neighbor_vonorms, neighbor_motif).cnf)
        
        cnf_construction_result = sorted(cnf_results, key=lambda cnf: cnf.coords)[0]
    
        return LatticeStepResult(
            step,
            neighbor_vonorms,
            None,
            cnf_construction_result,
        )

    def find_cnf_neighbors(self) -> LatticeNeighborSet:
        neighbors = LatticeNeighborSet()
        self._log(f"Considering {len(self.possible_steps())} possible steps...")
        for step in self.possible_steps():
            self._log("")
            self._log(f"Step: {step.vals}, {step.prereq_perm.perm.perm}")
            result = self.find_cnf_neighbor(step)
            if result is not None:
                neighbors.add_neighbor(result)
        return neighbors