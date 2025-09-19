class SellingPair(tuple):

    def __new__(cls, *vals):
        """
        Creates a new instance of the SellingPair. This method is called before
        __init__. It's the correct place to control the creation of immutable
        objects.
        """
        if len(vals) != 2:
            raise ValueError(f"SellingPair must be initialized with exactly two values, but got {len(vals)}.")

        # Sort the values to create the canonical representation.
        sorted_pair = tuple(sorted(vals))

        # Validate against the allowed pairs.
        if sorted_pair not in cls.CANONICAL_PAIRS:
            raise ValueError(f"Tried to instantiate SellingPair with non-viable pair: {vals}")

        # If valid, call the parent tuple's __new__ method to create the object.
        # We pass `cls` to ensure it creates an instance of SellingPair, not just tuple.
        return super().__new__(cls, sorted_pair)

    CANONICAL_PAIRS = [
        (0,1),
        (0,2),
        (0,3),
        (1,2),
        (1,3),
        (2,3),
    ]