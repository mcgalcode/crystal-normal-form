from .selling_transform_matrix import SellingTransformMatrix
from ..unimodular import combine_unimodular_matrices

class SellingReductionResult():

    def __init__(self,
                 reduced_object,
                 transform_matrices: list[SellingTransformMatrix],
                 num_steps: int,
                 tol: float):
        self.reduced_object = reduced_object
        self.all_transform_matrices = transform_matrices
        self.num_steps = num_steps
        self.tol = tol
    
    @property
    def transform_matrix(self):
        return combine_unimodular_matrices(self.all_transform_matrices)
