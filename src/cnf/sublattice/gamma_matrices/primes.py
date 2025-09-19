import math
import copy

from itertools import permutations, combinations

def get_prime_factors(n):
    factors = []
    i = 2
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n = n / i
        i = i + 1

    if n != 1:
        factors.append(int(n))
    return factors

def get_prime_parcels(prime_factors: list, parcel_size=3):
    padded_factors = copy.copy(prime_factors)
    parcels = set()

    while len(padded_factors) < parcel_size:
        padded_factors.append(1)

    if len(prime_factors) > 1:
        pairs_to_combine = set(combinations(prime_factors, 2))
        for pair in pairs_to_combine:
            new_factors = copy.copy(prime_factors)
            new_factors.remove(pair[0])
            new_factors.remove(pair[1])
            new_factors.append(pair[0] * pair[1])
            parcels = parcels.union(get_prime_parcels(new_factors, parcel_size))

    if len(padded_factors) == parcel_size:
        parcels = parcels.union(set(permutations(padded_factors)))

    return parcels