import enum


CONORM_PAIRS_TO_IDXS = {
    (0,1): 0,
    (0,2): 1,
    (0,3): 2,
    (1,2): 3,
    (1,3): 4,
    (2,3): 5
}

CONORM_IDX_TO_PAIR = { v: k for k, v in CONORM_PAIRS_TO_IDXS.items() }



CONORM_INDICES_TO_PAIRS = {
    0: {0,1},
    1: {0,2},
    2: {0,3},
    3: {1,2},
    4: {1,3},
    5: {2,3}
}
