import pytest

from cnf.sublattice import Fraction, Vector, CyclicGroup, SublatticeGeneratingSet

@pytest.fixture
def expected_n4_vecs():
    f34 = Fraction(3, 4)
    f14 = Fraction(1, 4)
    f12 = Fraction(1, 2)
    f0 = Fraction.zero()

    expected_n4_vecs = [
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
    for evec in expected_n4_vecs:
        if evec in chosen_evecs:
            print(f"Found duplicate: {evec}")
            raise RuntimeError("Max you wrote this test wrong!")
        else:
            chosen_evecs.add(evec)
    return chosen_evecs

def test_correctly_generates_sublattice_generating_set(expected_n4_vecs):
    expected_cyclic_groups = set([CyclicGroup.from_generator(g, 4) for g in expected_n4_vecs])
    assert len(expected_cyclic_groups) == len(expected_n4_vecs)
    assert len(expected_n4_vecs) == 28

    generating_set = SublatticeGeneratingSet.from_sublattice_index(4)
    generated_cyclic_groups = generating_set._cyclic_groups
    excess_expected: set[CyclicGroup] = expected_cyclic_groups - generated_cyclic_groups
    for v in excess_expected:
        print(v.generator)
    print("These vectors are NOT represented but should be") if len(excess_expected) > 0 else None
    assert len(excess_expected) == 0
    

    excess_generated = generated_cyclic_groups - expected_cyclic_groups
    print("These vectors ARE represented but should NOT be") if len(excess_generated) > 0 else None
    assert len(excess_generated) == 0
    for v in excess_generated:
        print(v.generator)    
    
    # for r in generating_set.representatives:
    #     print(r)

def test_can_compare_sublattice_generators(expected_n4_vecs):
    cgs = [CyclicGroup.from_generator(g, 4) for g in expected_n4_vecs]
    assert SublatticeGeneratingSet.from_sublattice_index(4) == SublatticeGeneratingSet(cgs)
