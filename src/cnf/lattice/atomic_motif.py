import numpy as np

class AtomicMotif():

    def __init__(self, relative_coords: list[list[int | float]]):
        self.relative_coords = np.array(relative_coords)
    
    def transform(self, transform: np.array):
        self.relative_coords = self.relative_coords @ transform
