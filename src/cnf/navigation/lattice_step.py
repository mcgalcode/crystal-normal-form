import numpy as np

from ..lattice.permutations import PermutationMatrix
from ..crystal_normal_form import CrystalNormalForm
from ..cnf_constructor import CNFConstructionResult
from ..lattice import LatticeNormalForm
from ..lattice.lnf_constructor import LatticeNormalFormConstructionResult
from ..motif.atomic_motif import DiscretizedMotif
from ..lattice.voronoi import VonormList

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