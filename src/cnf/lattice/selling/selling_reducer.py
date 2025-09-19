import numpy as np
from .selling_reduction_result import SellingReductionResult
from .selling_transform_matrix import SellingTransformMatrix
from ..superbasis import Superbasis
from ..vonorm_list import VonormList

from abc import ABC, abstractmethod

class SellingReducer(ABC):

    def __init__(self, tol = 0, verbose_logging = False, max_steps = 500):
        self.tol = tol
        self.max_steps = max_steps
        self._verbose_logging = verbose_logging

    @abstractmethod
    def apply_selling_transform(self, obj: Superbasis):
        pass

    def reduce(self, object: VonormList | Superbasis):
        num_steps = 0
        transform_matrices = [np.eye(3)]
        while not object.is_obtuse(tol=self.tol):
            object, acute_pair = self.apply_selling_transform(object)
            if self._verbose_logging:
                print(f"Selling transform {acute_pair}: {object}")

            transform_matrices.append(SellingTransformMatrix.from_pair(acute_pair))
            num_steps += 1
            if num_steps > self.max_steps:
                raise RuntimeError(f"Selling reduction failed to converge after {self.max_steps} steps")
            
        return SellingReductionResult(object,
                                      transform_matrices=transform_matrices,
                                      num_steps=num_steps,
                                      tol=self.tol)