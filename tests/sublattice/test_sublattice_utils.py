import pytest

from cnf.sublattice import get_divisors, valid_denominator_sets, generate_rational_coordinates, Fraction, Vector, CyclicGroup

def test_divisors():
    assert set(get_divisors(4)) == {1, 2, 4}
    assert set(get_divisors(12)) == {1, 2, 3, 4, 6, 12}

def test_denominator_sets():
    print(valid_denominator_sets(4))

def test_generate_rational_coords():
    f34 = Fraction(3, 4)
    f14 = Fraction(1, 4)
    f12 = Fraction(1, 2)
    f0 = Fraction.zero()

    expected_vectors = [
        Vector([f14, f0, f0]),
        Vector([f0, f14, f0]),
        Vector([f14, f14, f0]),
        Vector([f12, f14, f0]),
        Vector([f34, f14, f0]),
        Vector([f14, f12, f0]),
        Vector([f0, f0, f14]),
        ##
        Vector([f14, f0, f14]),
        Vector([f12, f0, f14]),
        Vector([f34, f0, f14]),
        Vector([f0, f14, f14]),
        Vector([f14, f14, f14]),
        Vector([f12, f14, f14]),
        Vector([f34, f14, f14]),
        ##
        Vector([f0, f12, f14]),
        Vector([f14, f12, f14]),
        Vector([f12, f12, f14]),
        Vector([f34, f12, f14]),
        Vector([f0, f34, f14]),
        Vector([f14, f34, f14]),
        Vector([f12, f34, f14]),
        ##
        Vector([f34, f34, f14]),
        Vector([f14, f0, f12]),
        Vector([f0, f14, f12]),
        Vector([f14, f14, f12]),
        Vector([f12, f14, f12]),
        Vector([f34, f14, f12]),
        Vector([f14, f12, f12]),
    ]

    chosen_evecs = set()
    for evec in expected_vectors:
        if evec in chosen_evecs:
            print(f"Found duplicate: {evec}")
            raise RuntimeError("Max you wrote this test wrong!")
        else:
            chosen_evecs.add(evec)

    expected_cyclic_groups = set([CyclicGroup.from_generator(g, 4) for g in expected_vectors])
    assert len(expected_cyclic_groups) == len(expected_vectors)
    assert len(chosen_evecs) == 28

    generated_vecs = generate_rational_coordinates(4)
    generated_cyclic_groups = set([CyclicGroup.from_generator(g, 4) for g in generated_vecs])
    print("These vectors are NOT represented but should be")
    excess_expected = expected_cyclic_groups - generated_cyclic_groups
    for v in excess_expected:
        print(v.generator)

    print("These vectors ARE represented but should NOT be")
    excess_generated = generated_cyclic_groups - expected_cyclic_groups
    for v in excess_generated:
        print(v.generator)    

def test_set_of_vectors():
    f1 = Fraction(1, 2)
    f2 = Fraction(1, 3)
    f3 = Fraction(1, 4)
    vec = Vector([f1, f2, f3])
    vec2 = Vector([f1, f2, f3])
    assert vec == vec2

    assert len(set([vec, vec2])) == 1   

def test_set_of_fractions():
    f1 = Fraction(1, 2)
    f2 = Fraction(1, 2)
    assert len(set([f1, f2])) == 1
