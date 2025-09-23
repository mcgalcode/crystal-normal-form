import numpy as np
from .selling_reduction_result import SellingReductionResult
from .selling_transform_matrix import SellingTransformMatrix
from ..superbasis import Superbasis
from ..voronoi.vonorm_list import VonormList

from abc import ABC, abstractmethod

class SellingReducer(ABC):

    def __init__(self,
                 tol = 1e-7,
                 verbose_logging = False,
                 max_steps = 500,
                 sorting_decimal_places=4):
        self.tol = tol
        self.max_steps = max_steps
        self._verbose_logging = verbose_logging
        self.sorting_decimal_places = sorting_decimal_places

    def select_pair_for_reduction(self, obj):
        pairs = []
        for i in range(4):
            for j in range(i + 1, 4):
                dot = self.get_dot_product_for_pair(obj, (i, j))
                if dot > self.tol: # Use a tolerance
                    pairs.append((np.round(dot, self.sorting_decimal_places), (i, j)))
        
        if len(pairs) == 0:
            return (None, None)
        else:
            pairs = sorted(pairs, reverse=True)
        return pairs[0]
    
    def apply_selling_transform(self, obj):
        dot_value, selected_pair = self.select_pair_for_reduction(obj)
        if selected_pair is None:
            return obj, None
        
        return self.get_transformed_object(obj, selected_pair), selected_pair
        
    @abstractmethod
    def get_dot_product_for_pair(self, obj, pair):
        pass

    @abstractmethod
    def get_transformed_object(self, obj, pair):
        pass

    def _logging_repr(self, obj):
        return f"{obj}"

    def reduce(self, object: VonormList | Superbasis):
        num_steps = 0
        transform_matrices = [SellingTransformMatrix(np.eye(3))]
        while not object.is_obtuse(tol=self.tol):
            if not object.is_superbasis():
                raise RuntimeError(f"Selling transformation converted object: {object} to non-superbasis form!")

            object, acute_pair = self.apply_selling_transform(object)
            if self._verbose_logging:
                print(f"Selling transform {acute_pair}: {self._logging_repr(object)}")

            transform_matrices.append(SellingTransformMatrix.from_pair(acute_pair))
            num_steps += 1
            if num_steps > self.max_steps:
                raise RuntimeError(f"Selling reduction failed to converge after {self.max_steps} steps")
            
        return SellingReductionResult(object,
                                      transform_matrices=transform_matrices,
                                      num_steps=num_steps,
                                      tol=self.tol)