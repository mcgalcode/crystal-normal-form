import numpy as np

class VectorTuple():

    @classmethod
    def from_tuple(cls, vec_tuple):
        return cls(np.array(vec_tuple))

    def __init__(self, vector: np.array):
        self.vector = np.copy(vector)
        self.tuple = tuple([int(e) for e in self.vector])
    
    def __hash__(self):
        return self.tuple.__hash__()
    
    def __eq__(self, other: 'VectorTuple'):
        return self.tuple == other.tuple
    
    def __repr__(self):
        return self.tuple.__repr__()