import numpy as np
from .vonorm_list import VonormList

PRIMARY_VONORM_IDXS = [0, 1, 2, 3]
SECONDARY_VONORM_IDXS = [4,5,6]

INCREMENT = 1
DECREMENT = -1

class DiscretizedVonormComputer():

    @staticmethod
    def discretize_vonorm_list(true_vonorm_list: VonormList, lattice_step_size: float):
        return DiscretizedVonormComputer(lattice_step_size).find_closest_valid_vonorms(true_vonorm_list)

    def __init__(self, lattice_step_size, verbose_log=False):
        self.lattice_step_size = lattice_step_size
        self._verbose_log = verbose_log

    def compute_error_change_at_idx(self, true_vonorms, rounded_vnorms, idx, adjustment):
        old_error = np.abs(rounded_vnorms[idx] * self.lattice_step_size - true_vonorms[idx])
        new_error = np.abs((rounded_vnorms[idx] + adjustment) * self.lattice_step_size - true_vonorms[idx])
        return new_error - old_error

    def compute_norm_error_change(self, true_vonorms, rounded_vnorms, idx, adjustment):
        old_error = np.linalg.norm(rounded_vnorms * self.lattice_step_size - true_vonorms)
        adjusted_rounded_vnorms = np.copy(rounded_vnorms)
        adjusted_rounded_vnorms[idx] = rounded_vnorms[idx] + adjustment
        new_error = np.linalg.norm(adjusted_rounded_vnorms * self.lattice_step_size - true_vonorms)
        return new_error - old_error
    
    def unrounded_discretized_vonorms(self, true_vonorms: VonormList):
        return VonormList(np.array(true_vonorms.vonorms) / self.lattice_step_size)
    
    def uncorrected_discretized_vonorms(self, true_vonorms: VonormList):
        return VonormList([int(v) for v in np.round(np.array(true_vonorms.vonorms) / self.lattice_step_size).astype(np.int64)])
    
    def find_closest_valid_vonorms(self, true_vonorms: VonormList):

        # The discretized version of the vonorm list has to satisfy
        # the following equality:
        #
        # v0^2 + v1^2 + v2^2 + v3^2 = (v0 + v1)^2 + (v0 + v2)^2 + (v0 + v3)^2
        #
        # The vonorm list comes in the following order:
        #
        # (v0^2, v1^2, v2^2, v3^2, (v0 + v1)^2, (v0 + v2)^2, (v0 + v3)^2)
        true_vonorms = VonormList(np.copy(true_vonorms.vonorms))
        if self._verbose_log:
            print(f"Unrounded, discretized: {self.unrounded_discretized_vonorms(true_vonorms)}")
            print(f"Rounded, discretized: {self.uncorrected_discretized_vonorms(true_vonorms)}")

        rounded_vonorms = self.uncorrected_discretized_vonorms(true_vonorms).vonorms
        # rounded_vonorms = np.round(np.array(true_vonorms) / self.lattice_step_size).astype(np.int64)
        primary_sum = np.sum(rounded_vonorms[:4])
        secondary_sum = np.sum(rounded_vonorms[4:])
        while primary_sum != secondary_sum:
            # In this loop, we assess the situation to see which adjustments
            # will get us closer to satisfying the inequality. We then simulate
            # each change and see which increases the distance from the true
            # vonorms the least
            possible_changes = []

            for idx in PRIMARY_VONORM_IDXS + SECONDARY_VONORM_IDXS:
                if primary_sum > secondary_sum:
                    # In this case, we have to either reduce a primary vonorm
                    # or increase a secondary vonorm
                    adjustment = DECREMENT if idx in PRIMARY_VONORM_IDXS else INCREMENT
                else:
                    # In this case, we have to either INCREASE a PRIMARY vonorm
                    # or DECREASE a SECONDARY vonorm
                    adjustment = INCREMENT if idx in PRIMARY_VONORM_IDXS else DECREMENT
                error_change = self.compute_error_change_at_idx(rounded_vonorms, true_vonorms, idx, adjustment)
                possible_changes.append((error_change, idx, adjustment))

            if self._verbose_log:
                print(f"Primary: {primary_sum}, Secondary: {secondary_sum}")
                print("Possible corrections")
                print("====================")
                for pc in possible_changes:
                    err, idx_to_adjust, adjustment = pc
                    print(f"idx: {idx_to_adjust}, change: {adjustment}, diff: {round(err, 3)}")
            least_damaging_change = sorted(possible_changes)[0]
            err, idx_to_adjust, adjustment = least_damaging_change
            rounded_vonorms[idx_to_adjust] = rounded_vonorms[idx_to_adjust] + adjustment
            if self._verbose_log:
                print(f"Selected Correction (idx: {idx_to_adjust}, change: {adjustment}, diff: {round(err, 3)}): {rounded_vonorms}...")
            primary_sum = np.sum(rounded_vonorms[:4])
            secondary_sum = np.sum(rounded_vonorms[4:])
        
        return VonormList([int(vo) for vo in rounded_vonorms])

        
