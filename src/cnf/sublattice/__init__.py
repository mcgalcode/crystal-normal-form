import math
from itertools import combinations_with_replacement, permutations
from .utils import are_coprime

class Fraction():

    @classmethod
    def whole_number(cls, num):
        return cls(num, 1)
    
    @classmethod
    def zero(cls):
        return cls.whole_number(0)

    def __init__(self, num: int, denom: int):
        if not isinstance(num, int):
            raise ValueError(f"Fraction instantiated with non-integer numerator: {num}")

        if not isinstance(denom, int):
            raise ValueError(f"Fraction instantiated with non-integer denominator: {denom}")

        self.num = num
        self.denom = denom
    
    def simplify(self):
        to_divide = math.gcd(self.num, self.denom)
        return Fraction(int(self.num / to_divide), int(self.denom / to_divide))
    
    def mod_one(self):
        new_num = int(self.num % self.denom)
        return Fraction(new_num, self.denom)
    
    def can_be_simplified(self):
        return math.gcd(self.num, self.denom) > 1
    
    def as_tuple(self):
        return (self.num, self.denom)

    def scale(self, scale: 'Fraction'):
        return Fraction(self.num * scale.num, self.denom * scale.denom)
        
    def is_multiple_of(self, other: 'Fraction'):
        if self.num == 0 and other.num == 0:
            return True
        elif self.num == 0 and other.num != 0:
            return False
        elif self.num != 0 and other.num == 0:
            return False
        
        common_multiple = math.lcm(self.denom, other.denom)
        # print(common_multiple)
        self_converted_num = int(self.num * common_multiple / self.denom)
        other_converted_num = int(other.num * common_multiple / other.denom)
        # print(self_converted_num, other_converted_num)
        if self_converted_num % other_converted_num == 0:
            return self_converted_num / other_converted_num
        else:
            return False

    def __eq__(self, other: 'Fraction'):
        simplified_self = self.simplify()
        simplified_other = other.simplify()

        return simplified_self.num == simplified_other.num and simplified_self.denom == simplified_other.denom
    
    def __repr__(self):
        if self.num == 0:
            return "0"
        else:
            return f"{self.num}/{self.denom}"
    
    def __hash__(self):
        return self.as_tuple().__hash__()
    
def get_divisors(N: int):
    return [i for i in range(1, N+1) if N%i == 0]

def valid_denominator_sets(N: int):
    divisors = get_divisors(N)
    denominator_sets = combinations_with_replacement(divisors, 3)
    filtered_sets = [s for s in denominator_sets if math.lcm(*s) == N]
    return filtered_sets

def generate_rational_coordinates(N: int):
    denominator_sets = valid_denominator_sets(N)
    vecs: list[Vector] = []
    for denominator_set in denominator_sets:
        denom_orderings = permutations(denominator_set)
        for denoms in denom_orderings:
            for m_1 in range(0, denoms[0]):
                for m_2 in range(0, denoms[1]):
                    for m_3 in range(0, denoms[2]):
                        coords = [
                            Fraction(m_1, denoms[0]).simplify(),
                            Fraction(m_2, denoms[1]).simplify(),
                            Fraction(m_3, denoms[2]).simplify()
                        ]
                        reduced_denoms = [f.denom for f in coords]
                        if math.lcm(*reduced_denoms) == N:
                            vec = Vector(coords)
                            vecs.append(vec)

    deduplicated_vecs = sorted(list(set(vecs)), key=lambda v: v.sortable_string())
    filtered_vecs: list[Vector] = []
    for candidate_vec in deduplicated_vecs:
        already_added = False
        for chosen_vec in filtered_vecs:
            if candidate_vec.in_same_cyclic_group(chosen_vec, N) or chosen_vec.in_same_cyclic_group(candidate_vec, N):
               already_added = True
               print(f"Excluding: {candidate_vec} because {chosen_vec} is already there")
               break
        if not already_added:
            filtered_vecs.append(candidate_vec) 

    return filtered_vecs

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
        return Vector([c.scale(scale) for c in self.coords])
    
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

class CyclicGroup():

    @classmethod
    def from_generator(cls, generator: Vector, N: int):
        generator = generator.simplify().mod_one()
        group = []
        for scalar in range(1,N):
            group.append(generator.scale(Fraction.whole_number(scalar)).mod_one().simplify())
        return cls(generator, group, N)

    def __init__(self, generator, members: list[Vector], N: int):
        self.N = N
        self.generator = generator
        self.members = frozenset(members)
    
    def __eq__(self, other: 'CyclicGroup'):
        return self.members == other.members
    
    def __hash__(self):
        return self.members.__hash__()
    
    def __contains__(self, item):
        if not isinstance(item, Vector):
            raise ValueError(f"Tested for containment in CyclicGroup item of wrong type {type(item)}")
        
        return item in self.members
        



class ModFractionVector():

    def __init__(self, coords: list[Fraction]):
        self.coords = [c.simplify().mod_one() for c in coords]
        
    def to_upper_triangular(self):
        pass
        # implement algorithm on page 35 of David's thesis

class UniqueSublatticeGeneratingVectorSet():

    def __init__(self, N: int, vectors: list[Vector]):
        self.vectors = []
        self._cyclic_groups = []
    
    def add_vector(self, vector: Vector):
        
        self.vectors.append(vector)