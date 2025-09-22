import enum

class Conorms(enum.Enum):

    P_01 = 0
    P_02 = 1
    P_03 = 2
    P_12 = 3
    P_13 = 4
    P_23 = 5


conorm_swaps = {
    (0, 1): [(Conorms.P_02, Conorms.P_12), (Conorms.P_03, Conorms.P_13)],
    (0, 2): [(Conorms.P_01, Conorms.P_12), (Conorms.P_03, Conorms.P_23)],
    (0, 3): [(Conorms.P_02, Conorms.P_23), (Conorms.P_01, Conorms.P_13)],
    (1, 2): [(Conorms.P_01, Conorms.P_02), (Conorms.P_13, Conorms.P_23)],
    (1, 3): [(Conorms.P_01, Conorms.P_03), (Conorms.P_12, Conorms.P_23)],
    (2, 3): [(Conorms.P_02, Conorms.P_03), (Conorms.P_12, Conorms.P_13)],
}

CONORM_INDICES_TO_CONORMS = {
    0: Conorms.P_01,
    1: Conorms.P_02,
    2: Conorms.P_03,
    3: Conorms.P_12,
    4: Conorms.P_13,
    5: Conorms.P_23,
}

CONORM_INDICES_TO_PAIRS = {
    0: {0,1},
    1: {0,2},
    2: {0,3},
    3: {1,2},
    4: {1,3},
    5: {2,3}
}
