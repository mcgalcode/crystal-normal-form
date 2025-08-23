import numpy as np

PRIMARY_VONORM_IDXS = [0, 1, 2, 3]
SECONDARY_VONORM_IDXS = [4,5,6]

class DiscretizedVonormComputer():

    def __init__(self, true_vonorms, lattice_step_size):
        self.true_vonorms = true_vonorms
        self.lattice_step_size = lattice_step_size

    def compute_error_change(self, rounded_vnorms, idx, adjustment):
        old_error = np.abs(rounded_vnorms[idx] * self.lattice_step_size - self.true_vonorms[idx])
        new_error = np.abs((rounded_vnorms[idx] + adjustment) * self.lattice_step_size - self.true_vonorms[idx])
        return new_error - old_error

    def find_closest_valid_vonorms(self):

        # The discretized version of the vonorm list has to satisfy
        # the following equality:
        #
        # v0^2 + v1^2 + v2^2 + v3^2 = (v0 + v1)^2 + (v0 + v2)^2 + (v0 + v3)^2
        #
        # The vonorm list comes in the following order:
        #
        # (v0^2, v1^2, v2^2, v3^2, (v0 + v1)^2, (v0 + v2)^2, (v0 + v3)^2)

        rounded_vonorms = np.round(np.array(self.true_vonorms) / self.lattice_step_size).astype(np.int64)
        primary_sum = np.sum(rounded_vonorms[:4])
        secondary_sum = np.sum(rounded_vonorms[4:])
        while primary_sum != secondary_sum:
            # In this loop, we assess the situation to see which adjustments
            # will get us closer to satisfying the inequality. We then simulate
            # each change and see which increases the distance from the true
            # vonorms the least
            possible_changes = []

            if primary_sum > secondary_sum:
                # In this case, we have to either reduce a primary vonorm
                # or increase a secondary vonorm
                for idx in PRIMARY_VONORM_IDXS:
                    error_change = self.compute_error_change(rounded_vonorms, idx, -1)
                    possible_changes.append((idx, -1, error_change))

                for idx in SECONDARY_VONORM_IDXS:
                    error_change = self.compute_error_change(rounded_vonorms, idx, +1)
                    possible_changes.append((idx, +1, error_change))
                    
            else:
                # In this case, we have to either INCREASE a PRIMARY vonorm
                # or DECREASE a SECONDARY vonorm
                for idx in PRIMARY_VONORM_IDXS:
                    error_change = self.compute_error_change(rounded_vonorms, idx, +1)
                    possible_changes.append((idx, +1, error_change))

                for idx in SECONDARY_VONORM_IDXS:
                    error_change = self.compute_error_change(rounded_vonorms, idx, -1)
                    possible_changes.append((idx, -1, error_change))
            
            least_damaging_change = sorted(possible_changes, key=lambda c: c[2])[0]
            idx_to_adjust, adjustment, error = least_damaging_change
            rounded_vonorms[idx_to_adjust] = rounded_vonorms[idx_to_adjust] + adjustment
            primary_sum = np.sum(rounded_vonorms[:4])
            secondary_sum = np.sum(rounded_vonorms[4:])
        
        return rounded_vonorms

        
