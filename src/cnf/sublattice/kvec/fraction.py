import math

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
    
    def copy(self):
        return Fraction(self.num, self.denom)

    def to_float(self):
        return self.num / self.denom
    
    def is_int(self):
        return self.num % self.denom == 0
    
    def is_zero(self):
        return self == Fraction.zero()
    
    def to_int(self):
        if not self.is_int():
            raise RuntimeError(f"Tried to convert non-int fraction {self} to int")
        return int(self.num / self.denom)
    
    def add(self, other: 'Fraction'):
        common_denom = self.common_denominator(other)
        compatible_self = self.convert_denominator(common_denom)
        compatible_other = other.convert_denominator(common_denom)
        assert compatible_other.denom == compatible_self.denom
        new_num = compatible_self.num + compatible_other.num
        return Fraction(new_num, common_denom).simplify()

    def convert_denominator(self, new_denominator):
        if new_denominator >= self.denom:
            if not new_denominator % self.denom == 0:
                raise ValueError(f"New denominator ({new_denominator}) is not an integer multiple of old {self.denom}")

            multiple = int(new_denominator / self.denom)
            return Fraction(self.num * multiple, new_denominator)
        else:
            raise ValueError(f"Converting to lower denom (old: {self.denom}, new: {new_denominator}) not supported :()")


    def common_denominator(self, other: 'Fraction'):
        return math.lcm(self.denom, other.denom)
    
    def as_tuple(self):
        return (self.num, self.denom)

    def multiply(self, scale):
        if isinstance(scale, int):
            scale = Fraction.whole_number(scale)

        return Fraction(self.num * scale.num, self.denom * scale.denom)
        
    def is_multiple_of(self, other: 'Fraction'):
        if self.num == 0 and other.num == 0:
            return True
        elif self.num == 0 and other.num != 0:
            return False
        elif self.num != 0 and other.num == 0:
            return False
        
        common_multiple = self.common_denominator(other)
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