import numpy as np

from ..lattice import LatticeNormalForm


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
                print(first_idx, second_idx)
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
