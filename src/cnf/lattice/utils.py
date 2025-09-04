import numpy as np
from .superbasis import Superbasis
from .vonorm_list import VonormList
from .selling import SELLING_TRANSFORM_INVERSE_MATRICES

ERROR_THRESHHOLD = 500

def selling_reduce(object: VonormList | Superbasis, tol=0, return_transform_mat = False):
    num_steps = 0
    transform_matrix = np.eye(3)
    while not object.is_obtuse(tol=tol):
        object, swap = object.selling_transform()
        if return_transform_mat:
            transform_matrix = SELLING_TRANSFORM_INVERSE_MATRICES[swap] @ transform_matrix
        num_steps += 1
        if num_steps > ERROR_THRESHHOLD:
            raise RuntimeError(f"Selling reduction failed to converge after {ERROR_THRESHHOLD} steps")
        
    if return_transform_mat:
        return object, num_steps, transform_matrix
    else:
        return object, num_steps