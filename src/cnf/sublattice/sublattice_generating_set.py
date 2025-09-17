from .utils import valid_denominator_sets
from itertools import permutations
from .vector import Vector, ModFractionVector
from .fraction import Fraction
from .cyclic_group import CyclicGroup
import math

from typing import Iterable


class SublatticeGeneratingSet():

    @classmethod
    def from_sublattice_index(cls, N: int):
        denominator_sets = valid_denominator_sets(N)
        cyclic_groups: set[CyclicGroup] = set()
        for denominator_set in denominator_sets:
            denom_orderings = permutations(denominator_set)
            for denoms in denom_orderings:
                for m_1 in range(0, denoms[0]):
                    for m_2 in range(0, denoms[1]):
                        for m_3 in range(0, denoms[2]):
                            vec = ModFractionVector([
                                Fraction(m_1, denoms[0]),
                                Fraction(m_2, denoms[1]),
                                Fraction(m_3, denoms[2])
                            ])
                            reduced_denoms = [f.denom for f in vec.coords]
                            if math.lcm(*reduced_denoms) == N:
                                cyclic_groups.add(CyclicGroup.from_generator(vec, N))

        return cls(cyclic_groups)

    def __init__(self, cgs: Iterable[CyclicGroup]):
        Ns = set([cg.N for cg in cgs])
        if not len(Ns) == 1:
            raise ValueError("SublatticeGeneratingSet instantiated with cyclic groups of varying index!")
        
        self._cyclic_groups = frozenset(cgs)
    
    @property
    def representatives(self) -> list[Vector]:
        reps = [cg.representative for cg in self._cyclic_groups]
        return sorted(reps, key=lambda r: r.sortable_string())
    
    def __eq__(self, other: 'SublatticeGeneratingSet'):
        return self._cyclic_groups == other._cyclic_groups