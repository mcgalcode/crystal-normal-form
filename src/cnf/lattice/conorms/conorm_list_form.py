from .constants import CONORM_INDICES_TO_CONORMS

class ConormListForm():

    def __init__(self, zero_indices):
        self.zero_indices = zero_indices

    def zero_conorms(self):
        return [CONORM_INDICES_TO_CONORMS[idx] for idx in self.zero_indices]

    def __repr__(self):
        return f"ConormListForm({self.zero_conorms})"