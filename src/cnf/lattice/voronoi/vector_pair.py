from ...utils.sorted_tuple import SortedTuple

class VoronoiVectorPair(SortedTuple):

    def __new__(cls, *vals):
        instantiated = super().__new__(cls, *vals)
        if len(instantiated) != 2:
            raise ValueError(f"VoronoiVectorPair must be initialized with exactly two values, but got {len(vals)}.")
    
        # Validate against the allowed pairs.
        if instantiated not in cls.CANONICAL_PAIRS:
            raise ValueError(f"Tried to instantiate VoronoiVectorPair with non-viable pair: {instantiated}")
        return instantiated

    CANONICAL_PAIRS = [
        (0,1),
        (0,2),
        (0,3),
        (1,2),
        (1,3),
        (2,3),
    ]