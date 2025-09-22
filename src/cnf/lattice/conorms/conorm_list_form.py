from .constants import CONORM_INDICES_TO_CONORMS, CONORM_INDICES_TO_PAIRS

class ConormListForm():

    def __init__(self, zero_indices):
        self.zero_indices = zero_indices
        self.voronoi_class = self._compute_voronoi_class()

    def zero_conorms(self):
        return [CONORM_INDICES_TO_CONORMS[idx] for idx in self.zero_indices]

    def _compute_voronoi_class(self):
        if len(self.zero_indices) == 3:
            return 5
        
        if len(self.zero_indices) == 1:
            return 2
        
        if len(self.zero_indices) == 0:
            return 1
        
        if len(self.zero_indices) == 2:
            pair_1 = self.zero_indices[0]
            pair_2 = self.zero_indices[1]
            if len(CONORM_INDICES_TO_PAIRS[pair_1].intersection(CONORM_INDICES_TO_PAIRS[pair_2])) == 0:
                return 3
            else:
                return 4

    def __repr__(self):
        return f"ConormListForm({self.zero_conorms})"
    
    def __len__(self):
        return len(self.zero_indices)