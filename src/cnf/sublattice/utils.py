import math
from itertools import combinations_with_replacement

def get_divisors(N: int):
    return [i for i in range(1, N+1) if N%i == 0]

def is_valid_denominator_set(denoms, N: int):
    return math.lcm(*denoms) == N

def valid_denominator_sets(N: int):
    divisors = get_divisors(N)
    denominator_sets = combinations_with_replacement(divisors, 3)
    filtered_sets = [s for s in denominator_sets if is_valid_denominator_set(s, N)]
    return filtered_sets
