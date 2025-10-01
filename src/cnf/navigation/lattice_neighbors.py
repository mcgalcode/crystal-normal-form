import numpy as np

from ..crystal_normal_form import CrystalNormalForm
from ..lattice import LatticeNormalForm
from ..lattice.lnf_constructor import LatticeNormalFormConstructor
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
                
                steps.append(cls([int(v) for v in vec]))
                steps.append(cls([-int(v) for v in vec]))
        
        return steps


    def __init__(self, vals):
        self.vals = vals
        self.tuple = tuple(vals)

        for idx, val in enumerate(vals):
            if np.abs(val) != 1 and val != 0:
                raise ValueError(f"LatticeStep instantiated with invalid element != 1 at pos {idx}: {tuple(vals)}")
        
        primary_sum = np.sum(vals[:4])
        secondary_sum = np.sum(vals[4:])
        if primary_sum != secondary_sum:
            raise ValueError(f"LatticeStep instantiated with imbalanced primary (sum: {primary_sum}) and secondary (sum: {secondary_sum}) values: {tuple(vals)}")
        

class LatticeNeighborFinder():

    def __init__(self):
        pass

    def find_lnf_neighbors(self, lnf_point: LatticeNormalForm):
        steps = LatticeStep.all_step_vecs()
        neighbors = set()
        lnf_constructor = LatticeNormalFormConstructor(lnf_point.lattice_step_size)
        for step in steps:
            old_vonorms = np.array(lnf_point.vonorms.vonorms)
            new_vonorms = tuple([int(v) for v in old_vonorms + np.array(step.vals)])
            construction_res = lnf_constructor.build_lnf_from_vonorms(new_vonorms)
            neighbors.add(construction_res.lnf)
        return list(neighbors)
    
    def find_cnf_neighbors(self, cnf_point: CrystalNormalForm) -> list[CrystalNormalForm]:
        steps = LatticeStep.all_step_vecs()
        neighbors = set()
        for step in steps:
            old_vonorms = np.array(cnf_point.lattice_normal_form.vonorms.vonorms)
            new_vonorms = VonormList(tuple([int(v) for v in old_vonorms + np.array(step.vals)]))

            if new_vonorms.is_obtuse() and np.all(np.array(new_vonorms.vonorms) > 0):
                new_point = CrystalNormalForm.from_discretized_vonorms_and_bnf(
                    new_vonorms,
                    cnf_point.basis_normal_form,
                    cnf_point.xi
                )
                neighbors.add(new_point)
        return list(neighbors)        
