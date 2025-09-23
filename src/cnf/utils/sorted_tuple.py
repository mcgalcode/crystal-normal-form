class SortedTuple(tuple):

    def __new__(cls, *vals):
        sorted_vals = tuple(sorted(vals))
        return super().__new__(cls, sorted_vals)