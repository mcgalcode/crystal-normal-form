import enum
import copy
import numpy as np

from itertools import permutations
from .voronoi_values import VoronoiValue, VoronoiVector, PrimaryVonorm, Conorm
from ...linalg import MatrixTuple
from ..permutations import ConormPermutation
from ..superbasis import get_v0_from_generating_vecs

class Sign(enum.Enum):

    POSITIVE = 1
    NEGATIVE = -1

    def multiply(self, other: 'Sign') -> 'Sign':
        if self == other:
            return Sign.POSITIVE
        else:
            return Sign.NEGATIVE

    @classmethod
    def from_int(cls, sign_int):
        if sign_int == -1:
            return Sign.NEGATIVE
        elif sign_int == 1:
            return Sign.POSITIVE
        else:
            raise ValueError(f"Cannot convert {sign_int} to Sign")
    
    def __repr__(self):
        if self == Sign.NEGATIVE:
            return "-"
        else:
            return "+"

class SignedVoronoiValue(tuple):

    @classmethod
    def one_from_int(cls, sign_int: int, val: VoronoiValue):
        sign = Sign.from_int(sign_int)
        return cls([sign, 1, val])

    @classmethod
    def negative_one(cls, v: VoronoiValue):
        return cls([Sign.NEGATIVE, 1, v])
    
    @classmethod
    def positive_one(cls, v: VoronoiValue):
        return cls([Sign.POSITIVE, 1, v])
    
    @classmethod
    def from_signed_count(cls, count: int, val: VoronoiValue):
        if count < 0:
            return cls([Sign.NEGATIVE, abs(count), val])
        elif count > 0:
            return cls([Sign.POSITIVE, abs(count), val])
        else:
            raise ValueError(f"Tried to instantiate SignedVoronoiValue w value zero!")        

    def __new__(cls, vals):

        if len(vals) != 3:
            raise ValueError(f"SignedVoronoiValue requires three-tuple, got: {vals}")
        
        if not isinstance(vals[0], Sign):
            raise ValueError(f"SignedVoronoiValue requires Sign in position 0, got: {vals[0]}")
        
        if not (isinstance(vals[1], int) or isinstance(vals[1], np.integer)):
            raise ValueError(f"SignedVoronoiValue requires positive int in position 1, got: {vals[1]}")
        
        if not vals[1] > 0:
            raise ValueError(f"SignedVoronoiValue requires positive int in position 1, got: {vals[1]}")
        
        if not (isinstance(vals[2], VoronoiValue) or isinstance(vals[2], VoronoiVector)):
            raise ValueError(f"SignedVoronoiValue requires VoronoiValue in position 2, got: {vals[2]}")

        return super().__new__(cls, vals)
    
    @property
    def sign(self) -> Sign:
        return self[0]
    
    @property
    def count(self) -> int:
        return self[1]

    @property
    def signed_count(self) -> int:
        if self.sign == Sign.POSITIVE:
            return self.count
        else:
            return -self.count

    def negative(self):
        return self.multiply_sign(Sign.NEGATIVE)
    
    @property
    def value(self) -> VoronoiValue:
        return self[2]
    
    def multiply_sign(self, sign: Sign):
        return SignedVoronoiValue([
            self.sign.multiply(sign),
            self.count,
            self.value
        ])
    
    def multiply_count(self, count: int):
        return SignedVoronoiValue([
            self.sign,
            self.count * count,
            self.value
        ])

    def __eq__(self, other: 'SignedVoronoiValue'):
        return self.sign == other.sign and \
               self.count == other.count and \
               self.value == other.value

class SignedValueSet():

    def __init__(self, vals: list[SignedVoronoiValue]):
        self.vals = {}
        for v in vals:
            self.add_val(v)
    
    def contained_type():
        return SignedVoronoiValue
    
    def add_val(self, signed_val: SignedVoronoiValue):
        obj = signed_val.value
        if obj in self.vals:
            self.vals[obj] = self.vals[obj] + signed_val.signed_count
        else:
            self.vals[obj] = signed_val.signed_count

        if self.vals[obj] == 0:
            del self.vals[obj]
        
    def get_val(self, val: VoronoiValue) -> SignedVoronoiValue:
        ct = self.get_count(val)
        if ct == 0:
            return None
        else:
            return SignedVector.from_signed_count(ct, val)
    
    def to_list(self) -> list[SignedVoronoiValue]:
        return [self.get_val(v) for v in self.vals.keys()]
    
    def to_object_list(self):
        return [v.value for v in self.to_list()]
    
    def __contains__(self, obj):
        if isinstance(obj, SignedVoronoiValue):
            return obj in self.to_list()
        else:
            return obj in self.to_object_list()
    
    def __len__(self):
        return len(self.to_list())
    
    def __repr__(self):
        return f"SignedValueSet({self.to_list()})"
    
    def get_count(self, val: VoronoiValue):
        return self.vals.get(val, 0)
    
    def distribute(self, sign: Sign):
        return SignedValueSet([v.multiply_sign(sign) for v in self.to_list()])


class SignedVector(SignedVoronoiValue):

    @classmethod
    def positive_v1(cls):
        return cls.positive_one(VoronoiVector.V1())
    
    @classmethod
    def negative_v1(cls):
        return cls.negative_one(VoronoiVector.V1())
    
    @classmethod
    def positive_v2(cls):
        return cls.positive_one(VoronoiVector.V2())
    
    @classmethod
    def negative_v2(cls):
        return cls.negative_one(VoronoiVector.V2())
    
    @classmethod
    def positive_v3(cls):
        return cls.positive_one(VoronoiVector.V3())
    
    @classmethod
    def negative_v3(cls):
        return cls.negative_one(VoronoiVector.V3())
    
    @classmethod
    def positive_v0(cls):
        return cls.positive_one(VoronoiVector.V0())
    
    @classmethod
    def negative_v0(cls):
        return cls.negative_one(VoronoiVector.V0())

    def dot(self, other: 'SignedVector') :
        sign = self.sign.multiply(other.sign)
        count = self.count * other.count
        val = self.value.dot(other.value)
        return SignedVoronoiValue([sign, count, val])
    
class SignedVectorSet(SignedValueSet):

    def contained_type():
        return SignedVector

    def multiply(self, other: 'SignedVectorSet'):
        value_set = SignedValueSet([])
        for v_1 in self.to_list():
            for v_2 in other.to_list():
                value_set.add_val(v_1.dot(v_2))
        return value_set
    
class Transformation():

    def __init__(self, mat: MatrixTuple):
        self.mat = mat
    
    def get_col(self, idx):
        if idx not in range(0,4):
            raise ValueError(f"Can't get basis column with idx {idx}")
        if idx == 0:
            return self.v0()
        else:
            return self.mat.to_cols()[idx - 1]
    
    def v0(self):
        v0 = get_v0_from_generating_vecs([col.vector for col in self.mat.to_cols()])
        if all([isinstance(entry, int) for entry in self.mat.tuple]):
            v0 = tuple([int(i) for i in v0])
        return v0


class ConormCalculator():

    @staticmethod
    def vonorms_to_conorms(value_set: SignedValueSet):
        new_set = SignedValueSet([])
        for val in value_set.to_list():
            # print(val)
            if isinstance(val.value, PrimaryVonorm):
                new_vals = ConormCalculator.primary_vonorm_to_conorms(val.sign, val.value)
                # print(f"Converted: {new_vals}")
                for nv in new_vals.to_list():
                    scaled = nv.multiply_count(val.count)
                    new_set.add_val(scaled)
            else:
                new_set.add_val(val)
        return new_set

    @staticmethod
    def primary_vonorm_to_conorms(sign: Sign, pv: PrimaryVonorm):
        other_idxs = set(range(4)) - {pv.idx}
        signed_conorms = [SignedVoronoiValue.negative_one(Conorm((pv.idx, i))) for i in other_idxs]
        svs = SignedValueSet(signed_conorms)
        return svs.distribute(sign)

    @staticmethod
    def col_to_vector_set(col):
        vector_set = SignedVectorSet([])

        if all([v == -1 for v in col]):
            vector_set.add_val(SignedVector.positive_v0())
            return vector_set
        
        for idx, col_val in enumerate(col):
            vector_idx = idx + 1
            if col_val != 0:
                signed_vec = SignedVector.from_signed_count(col_val, VoronoiVector(vector_idx))
                vector_set.add_val(signed_vec)
        return vector_set
    
    @staticmethod
    def remove_zeros(conorm_set: SignedValueSet, zero_conorms: set[Conorm]):
        new_set = SignedValueSet([])
        for signed_value in conorm_set.to_list():
            if signed_value.value not in zero_conorms:
                new_set.add_val(signed_value)
        return new_set

    @staticmethod
    def validate_conorm_set(conorm_set: SignedValueSet):
        if len(conorm_set) > 1:
            raise ValueError(f"Computed ConormSet had too many vals: {conorm_set}")
        
        cnorm_signed = conorm_set.to_list()[0]
        if cnorm_signed.sign == Sign.NEGATIVE:
            raise ValueError(f"Computed ConormSet had negative coeff: {conorm_set}")
        if cnorm_signed.count != 1:
            raise ValueError(f"Computed ConormSet had too multiple conorms: {conorm_set}")
        if not isinstance(cnorm_signed.value, Conorm):
            raise ValueError(f"Computed ConormSet did not contain conorm: {conorm_set}")
        
        return cnorm_signed.value
    
    def __init__(self, t: Transformation, zero_conorms = None):
        if zero_conorms is None:
            self.zero_conorms = set()
        else:
            self.zero_conorms = set(zero_conorms)
        self.transformation = t

    def get_conorm(self, c: Conorm, log=False):
        v1_idx = c.i
        v2_idx = c.j

        col1 = self.transformation.get_col(v1_idx)
        col2 = self.transformation.get_col(v2_idx)
        
        if log:
            print(f"Finding dot product between: {col1} and {col2}")
        col1_vectors = ConormCalculator.col_to_vector_set(col1)
        col2_vectors = ConormCalculator.col_to_vector_set(col2)
        if log:
            print(f"Converted {col1} to {col1_vectors}")
        if log:
            print(f"Converted {col2} to {col2_vectors}")
        values = col1_vectors.multiply(col2_vectors)
        if log:
            print(f"Dot product is {values}")
        reduced = ConormCalculator.vonorms_to_conorms(values)
        if log:
            print(f"After reduction {reduced}")
        filtered = ConormCalculator.remove_zeros(reduced, self.zero_conorms)
        if log:
            print(f"After removing zero values ({self.zero_conorms}): {filtered}")
        return filtered
    
    
    def get_permutations(self):
        template = [None, None, None, None, None, None, None]
        
        for c in Conorm.all_conorms():
            # Compute the set of old conorms that constitute this new one
            old_conorm_combination = self.get_conorm(c)
            if len(old_conorm_combination) > 0:
                computed_conorm = ConormCalculator.validate_conorm_set(old_conorm_combination)
                if not computed_conorm:
                    raise ValueError("Conorm mask and transform did not yield valid permutation")
                template[c.idx] = computed_conorm.idx

        # print(f"Template: {template}")
        pinned_indices = [idx for idx, val in enumerate(template) if val is not None]
        zeros = [c.idx for c in self.zero_conorms] + [6]
        # print(f"Pinned indices: {pinned_indices}")
        # print(f"Known zeros: {zeros}")
        fillable_indices = list(set(range(7)) - set(pinned_indices))

        filled_permutations = []
        for perm in permutations(zeros):
            # print(perm)
            filled_perm = copy.copy(template)
            for item in zip(fillable_indices, perm):
                fillable_idx, permutation_idx = item
                filled_perm[fillable_idx] = permutation_idx
            filled_permutations.append(filled_perm)

        return [ConormPermutation(p) for p in filled_permutations]


