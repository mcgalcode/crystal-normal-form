from .selling_transform_matrix import SellingTransformMatrix

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
        combined_mat = self.all_transform_matrices[0]
        for mat in self.all_transform_matrices[1:]:
            combined_mat = combined_mat @ mat
        return combined_mat
