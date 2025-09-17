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