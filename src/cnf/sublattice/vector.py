from .fraction import Fraction

class Vector():

    def __init__(self, coords: list[Fraction]):
        self.coords = coords

    def __len__(self):
        return len(self.coords)

    def __repr__(self):
        return f"{self.coords}"
    
    def __eq__(self, other: 'Vector'):
        return all([self_coord == other_coord for self_coord, other_coord in zip(self.coords, other.coords)])

    def __hash__(self):
        return tuple([c.simplify().as_tuple() for c in self.coords]).__hash__()
    
    def in_same_cyclic_group(self, other: 'Vector', N: int):
        other_mod_one = other.mod_one()
        for scalar in range(2,N):
            scaled_self = self.scale(Fraction.whole_number(scalar))
            scaled_self_mod_one = scaled_self.mod_one().simplify()
            # print(f"Scaled by {scalar}: {scaled_self_mod_one}")
            if other_mod_one == scaled_self_mod_one:
                return True
        return False
    
    def scale(self, scale: Fraction):
        return Vector([c.multiply(scale) for c in self.coords])
    
    def mod_one(self):
        return Vector([f.mod_one() for f in self.coords])
    
    def simplify(self):
        return Vector([f.simplify() for f in self.coords])
    
    def is_multiple_of(self, other: 'Vector'):
        multiple_test_results = [this.is_multiple_of(that) for this, that in zip(self.coords, other.coords)]
        # print(multiple_test_results)
        if False in multiple_test_results:
            return False
        else:
            numeric_results = [m for m in multiple_test_results if m is not True]
            return len(set(numeric_results)) == 1
    
    def sortable_string(self):
        vals = []
        for c in self.coords:
            vals.append(c.num / c.denom)
        return tuple(vals)

class ModFractionVector(Vector):

    def __init__(self, coords: list[Fraction]):
        self.coords = [c.simplify().mod_one() for c in coords]
        
    # def to_upper_triangular(self):
    #     pass
        # implement algorithm on page 35 of David's thesis
