import numpy as np

from ..lattice.permutations import PermutationMatrix
from ..crystal_normal_form import CrystalNormalForm
from ..cnf_constructor import CNFConstructionResult
from ..lattice import LatticeNormalForm
from ..lattice.lnf_constructor import LatticeNormalFormConstructionResult
from ..motif.atomic_motif import DiscretizedMotif
from ..lattice.voronoi import VonormList
from ..linalg import MatrixTuple

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

    def __init__(self, vals, vonorms: VonormList, transformed_motif: DiscretizedMotif, matrix: MatrixTuple):
        self.vals = vals
        self.tuple = tuple(vals)
        self.vonorms = vonorms
        self.matrix = matrix
        self.transformed_motif = transformed_motif

        for idx, val in enumerate(vals):
            if np.abs(val) != 1 and val != 0:
                raise ValueError(f"LatticeStep instantiated with invalid element != 1 at pos {idx}: {tuple(vals)}")
            
    def _as_tuple(self):
        return (self.tuple, tuple(self.vonorms.vonorms), self.transformed_motif.to_bnf_list(), self.matrix.tuple)
    
    def __eq__(self, other: 'LatticeStep'):
        return self._as_tuple() == other._as_tuple()
    
    def __hash__(self):
        return self._as_tuple().__hash__()
    
    def __repr__(self):
        return f"LatticeStep<vonorm_adj={self.vals}, perm={self.vonorms}>"
    
    def print_details(self):
        print(f"Step adj. vec: {self.vals}")
        print(f"Vonorms: {self.vonorms}")
        print(f"Prerequisite matrix: {self.matrix.tuple}")

class LatticeStepResult():

    def __init__(self,
                 step: LatticeStep,
                 adjusted_vonorms: VonormList,
                 construction_result: CNFConstructionResult | LatticeNormalFormConstructionResult,
                 result: LatticeNormalForm | CrystalNormalForm,
                 motif_stabilizers: list[MatrixTuple],
                 stabilized_motif: DiscretizedMotif = None):
        self.step = step
        self.adjusted_vonorms = adjusted_vonorms
        self.motif_stabilizers = motif_stabilizers
        self.stabilized_motif = stabilized_motif
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