import enum

class Sign(enum.Enum):

    POSITIVE = 1
    NEGATIVE = -1

class VoronoiValue(enum.Enum):
    pass

class Conorm(VoronoiValue):

    P_01 = 0
    P_02 = 1
    P_03 = 2
    P_12 = 3
    P_13 = 4
    P_23 = 5

class Vonorm(VoronoiValue):
    V_0 = 0
    V_1 = 1
    V_2 = 2
    V_3 = 3
    V_01 = 4
    V_02 = 5
    V_03 = 6

    V_12 = 7
    V_13 = 8
    V_23 = 9


class SignedVoronoiValue(tuple):

    def __new__(cls, *vals):

        def _raise_validation_err():
            raise ValueError("A SignedVoronoiValue should be instantiated with a three-tuple (Sign, int, VoronoiValue)")

        if len(vals) != 3:
            _raise_validation_err()
        
        if not isinstance(vals[0], Sign):
            _raise_validation_err()
        
        if not isinstance(vals[1], VoronoiValue):
            _raise_validation_err()

        return super().__new__(cls, vals)
    
    @property
    def sign(self):
        return self[0]
    
    @property
    def count(self):
        return self[1]
    
    @property
    def value(self):
        return self[2]
