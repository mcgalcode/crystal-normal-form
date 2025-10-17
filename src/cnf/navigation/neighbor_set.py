import numpy as np

from ..crystal_normal_form import CrystalNormalForm
from ..lattice import LatticeNormalForm
from .lattice_step import LatticeStepResult
from .neighbor import Neighbor


class NeighborSet():

    def __init__(self):
        self.neighbors_to_steps: dict[CrystalNormalForm | LatticeNormalForm, list[LatticeStepResult]] = {}
    
    def add_neighbor(self, step_result: LatticeStepResult):
        nb_point = step_result.result
        if nb_point in self.neighbors_to_steps:
            self.neighbors_to_steps[nb_point].append(step_result)
        else:
            self.neighbors_to_steps[nb_point] = [step_result]

    @property
    def neighbors(self) -> list[Neighbor]:
        nbs = []
        for nb, step_results in self.neighbors_to_steps.items():
            nbs.append(Neighbor(nb, step_results))
        return nbs
    
    def steps_for_neighbor(self, neighbor: LatticeNormalForm | CrystalNormalForm):
        no_steps = []
        return self.neighbors_to_steps.get(neighbor, [])
    
    def get_neighbor(self, nb_tuple: tuple):
        matching = [n for n in self.neighbors if n.coords == nb_tuple]
        if len(matching) == 1:
            return matching[0]
        else:
            return None
    
    def __contains__(self, item: LatticeNormalForm | CrystalNormalForm):
        if not (isinstance(item, LatticeNormalForm) or isinstance(item, CrystalNormalForm)):
            raise ValueError(f"Can't tell if type {type(item)} is in NeighborSet")

        return item in [n.point for n in self.neighbors]
    
    def __len__(self):
        return len(self.neighbors_to_steps)