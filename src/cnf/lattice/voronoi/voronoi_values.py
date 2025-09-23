from .vector_pair import VoronoiVectorPair

class VoronoiValue():
    pass


CONORM_PAIRS_TO_IDXS = {
    (0,1): 0,
    (0,2): 1,
    (0,3): 2,
    (1,2): 3,
    (1,3): 4,
    (2,3): 5
}
class Conorm(VoronoiValue):

    def __init__(self, pair: VoronoiVectorPair):
        if not isinstance(pair, VoronoiVectorPair): 
            pair = VoronoiVectorPair(*pair)        
        self.pair = pair

    @property
    def i(self):
        return self.pair[0]
    
    @property
    def j(self):
        return self.pair[1]
    
    def idx(self):
        return CONORM_PAIRS_TO_IDXS[self.pair]
    
    def __repr__(self):
        return f"P({self.i},{self.j})"
    
    def __eq__(self, other: 'Conorm'):
        if not isinstance(other, Conorm):
            return False
        return self.pair == other.pair
    
    def __hash__(self):
        return (self.__class__.__name__, self.pair).__hash__()

class PrimaryVonorm(VoronoiValue):
    
    def __init__(self, vector_idx: int):
        if vector_idx not in range(0, 4):
            raise ValueError(f"PrimaryVonorm got invalid index: {vector_idx}")
        self.idx = vector_idx
    
    def __eq__(self, other: 'PrimaryVonorm'):
        if not isinstance(other, PrimaryVonorm):
            return False
        return self.idx == other.idx

    def __repr__(self):
        return f"(V_{self.idx})^2"
    
    def __hash__(self):
        return (self.__class__.__name__, self.idx).__hash__()
    
class SecondaryVonorm(VoronoiValue):

    def __init__(self, pair: VoronoiVectorPair):
        if not isinstance(pair, VoronoiVectorPair):
            pair = VoronoiVectorPair(*pair)
        self.pair = pair
    
    @property
    def complement(self):
        all_idxs = set(range(0,4))
        complement = all_idxs - set(self.pair)
        return SecondaryVonorm(VoronoiVectorPair(*complement))
    
    @property
    def is_canonical(self):
        return 0 in self.pair
    
    def __repr__(self):
        if self.is_canonical():
            return f"(V_{self.pair[0]} + V_{self.pair[1]})^2"
        else:
            return f"(-V_{self.pair[0]} - V_{self.pair[1]})^2"

    def __eq__(self, other: 'SecondaryVonorm'):
        if not isinstance(other, SecondaryVonorm):
            return False
        return self.pair == other.pair
    
    def __hash__(self):
        return (self.__class__.__name__, self.pair).__hash__()

class VoronoiVector(VoronoiValue):

    @classmethod
    def V1(cls):
        return cls(1)

    @classmethod
    def V2(cls):
        return cls(2)

    @classmethod
    def V3(cls):
        return cls(3)

    @classmethod
    def V0(cls):
        return cls(0) 

    def __init__(self, vector_idx: int):
        if vector_idx not in range(0, 4):
            raise ValueError(f"VoronoiVector got invalid index: {vector_idx}")
        self.idx = vector_idx

    def dot(self, other: 'VoronoiVector'):
        if self != other:
            return Conorm(VoronoiVectorPair(self.idx, other.idx))
        else:
            return PrimaryVonorm(self.idx)
    
    def __eq__(self, other: 'VoronoiVector'):
        return self.idx == other.idx
    
    def __repr__(self):
        return f"V_{self.idx}"
    
    def __hash__(self):
        return (self.__class__.__name__, self.idx).__hash__()
