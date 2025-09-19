from .fraction_vector import FractionVector, ModFractionVector
from .utils import is_valid_denominator_set

class CyclicGroup():

    @classmethod
    def from_generator(cls, generator: FractionVector, N: int):
        generator = ModFractionVector.from_vec(generator)
        group = []
        for scalar in range(1,N):
            scaled = generator.scale(scalar)
            denoms = [f.denom for f in scaled.coords]
            if is_valid_denominator_set(denoms, N):
                group.append(scaled)
        return cls(generator, group, N)

    def __init__(self, generator, members: list[FractionVector], N: int):
        self.N = N
        self.generator = generator
        self.members = frozenset(members)

        self.representative = sorted(list(members), key=lambda vec: vec.sortable_string())[0]
    
    def __eq__(self, other: 'CyclicGroup'):
        return self.members == other.members
    
    def __hash__(self):
        return self.members.__hash__()
    
    def __contains__(self, item):
        if not isinstance(item, FractionVector):
            raise ValueError(f"Tested for containment in CyclicGroup item of wrong type {type(item)}")
        
        return item in self.members