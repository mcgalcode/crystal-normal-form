import numpy as np

VONORM_TO_CONORM_TRANSFORM_NSO_SUPERBASIS = np.array([
    [-1, -1, 1, 1, 1, -1, -1],
    [-1, 1, -1, 1, -1, 1, -1],
    [-1, 1, 1, -1, -1, -1, 1],
    [1, -1, -1, 1, -1, -1, 1],
    [1, -1, 1, -1, -1, 1, -1],
    [1, 1, -1, -1, 1, -1, -1],
    [-1, -1, -1, -1, 1, 1, 1],
])

CONORM_TO_VONORM_TRANSFORM_NSO_SUPERBASIS = np.linalg.inv(VONORM_TO_CONORM_TRANSFORM_NSO_SUPERBASIS)