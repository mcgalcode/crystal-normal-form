import pytest

from cnf.sublattice.gamma_matrices.primes import get_prime_factors, get_prime_parcels

# def test_get_prime_factors():
#     print(get_prime_factors(123))

def test_get_prime_parcels():
    print(get_prime_parcels(get_prime_factors(6)))