import pytest
import numpy as np

from cnf.sublattice.kvec import Fraction, FractionVector

@pytest.fixture(scope='module')
def n_equals_4_generators_with_kvecs():
    return [
        # Row 1
        (FractionVector([Fraction(1,4), Fraction.zero(), Fraction.zero()]),
         np.array([
            [4, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (FractionVector([Fraction.zero(), Fraction(1,4), Fraction.zero()]),
         np.array([
            [1, 0, 0],
            [0, 4, 0],
            [0, 0, 1],
        ])),
        (FractionVector([Fraction(1,4), Fraction(1,4), Fraction.zero()]),
         np.array([
            [4, 3, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (FractionVector([Fraction(1,2), Fraction(1,4), Fraction.zero()]),
         np.array([
            [2, 1, 0],
            [0, 2, 0],
            [0, 0, 1],
        ])),
        (FractionVector([Fraction(3,4), Fraction(1,4), Fraction.zero()]),
         np.array([
            [4, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (FractionVector([Fraction(1,4), Fraction(1,2), Fraction.zero()]),
         np.array([
            [4, 2, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (FractionVector([Fraction(0,4), Fraction(0,4), Fraction(1,4)]),
         np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 4],
        ])),

        # # Row 2
        (FractionVector([Fraction(1,4), Fraction(0,4), Fraction(1,4)]),
         np.array([
            [4, 0, 3],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (FractionVector([Fraction(1,2), Fraction(0,4), Fraction(1,4)]),
         np.array([
            [2, 0, 1],
            [0, 1, 0],
            [0, 0, 2],
        ])),
        (FractionVector([Fraction(1,4), Fraction(1,4), Fraction.zero()]),
         np.array([
            [4, 3, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (FractionVector([Fraction(1,4), Fraction(1,4), Fraction.zero()]),
         np.array([
            [4, 3, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (FractionVector([Fraction(1,4), Fraction(1,4), Fraction(1,4)]),
         np.array([
            [4, 3, 3],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (FractionVector([Fraction(1,2), Fraction(1,4), Fraction(1,4)]),
         np.array([
            [2, 1, 0],
            [0, 2, 3],
            [0, 0, 1],
        ])),
        (FractionVector([Fraction(3,4), Fraction(1,4), Fraction(1,4)]),
         np.array([
            [4, 1, 1],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        # Row 3
        (FractionVector([Fraction(0,4), Fraction(1,2), Fraction(1,4)]),
         np.array([
            [1, 0, 0],
            [0, 2, 1],
            [0, 0, 2],
        ])),
        (FractionVector([Fraction(1,4), Fraction(1,2), Fraction(1,4)]),
         np.array([
            [4, 2, 3],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (FractionVector([Fraction(1,2), Fraction(1,2), Fraction(1,4)]),
         np.array([
            [2, 1, 1],
            [0, 1, 0],
            [0, 0, 2],
        ])),
        (FractionVector([Fraction(3,4), Fraction(1,2), Fraction(1,4)]),
         np.array([
            [4, 2, 1],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (FractionVector([Fraction(0,4), Fraction(3,4), Fraction(1,4)]),
         np.array([
            [1, 0, 0],
            [0, 4, 1],
            [0, 0, 1],
        ])),
        (FractionVector([Fraction(1,4), Fraction(3,4), Fraction(1,4)]),
         np.array([
            [4, 1, 3],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (FractionVector([Fraction(1,2), Fraction(3,4), Fraction(1,4)]),
         np.array([
            [2, 1, 0],
            [0, 2, 1],
            [0, 0, 1],
        ])),
        # Row 4
        (FractionVector([Fraction(3,4), Fraction(3,4), Fraction(1,4)]),
         np.array([
            [4, 3, 1],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (FractionVector([Fraction(1,4), Fraction(0,4), Fraction(1,2)]),
         np.array([
            [4, 0, 2],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (FractionVector([Fraction(0,4), Fraction(1,4), Fraction(1,2)]),
         np.array([
            [1, 0, 0],
            [0, 4, 2],
            [0, 0, 1],
        ])),
        (FractionVector([Fraction(1,4), Fraction(1,4), Fraction(1,2)]),
         np.array([
            [4, 3, 2],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (FractionVector([Fraction(1,2), Fraction(1,4), Fraction(1,2)]),
         np.array([
            [2, 1, 0],
            [0, 2, 2],
            [0, 0, 1],
        ])),
        (FractionVector([Fraction(3,4), Fraction(1,4), Fraction(1,2)]),
         np.array([
            [4, 1, 2],
            [0, 1, 0],
            [0, 0, 1],
        ])),
        (FractionVector([Fraction(1,4), Fraction(1,2), Fraction(1,2)]),
         np.array([
            [4, 2, 2],
            [0, 1, 0],
            [0, 0, 1],
        ])),
    ]

@pytest.fixture(scope='module')
def n_equals_4_generators_from_kvecs(n_equals_4_generators_with_kvecs):
    return [pair[1] for pair in n_equals_4_generators_with_kvecs]

@pytest.fixture(scope='module')
def n_equals_4_generators_not_from_kvecs():
    return [
        np.array([
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 1],
        ]),
        np.array([
            [2, 0, 1],
            [0, 2, 0],
            [0, 0, 1],
        ]),
        np.array([
            [2, 0, 0],
            [0, 2, 1],
            [0, 0, 1],
        ]),
        np.array([
            [2, 0, 1],
            [0, 2, 1],
            [0, 0, 1],
        ]),
        np.array([
            [2, 0, 0],
            [0, 1, 0],
            [0, 0, 2],
        ]),
        np.array([
            [2, 1, 0],
            [0, 1, 0],
            [0, 0, 2],
        ]),
        np.array([
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
        ]),
    ]

@pytest.fixture(scope='module')
def all_n_equals_4_generators(n_equals_4_generators_from_kvecs, n_equals_4_generators_not_from_kvecs):
    return n_equals_4_generators_not_from_kvecs + n_equals_4_generators_from_kvecs
