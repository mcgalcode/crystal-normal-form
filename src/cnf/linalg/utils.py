import numpy as np

def is_unimodular(matrix: np.ndarray) -> bool:
    """
    Checks if a given matrix is unimodular.

    A matrix is unimodular if it is a square integer matrix
    with a determinant of +1 or -1.

    Args:
        matrix: A NumPy array representing the matrix.

    Returns:
        True if the matrix is unimodular, False otherwise.
    """
    # Ensure the input is a NumPy array
    m = np.asarray(matrix)

    # 1. Check if the matrix is square.
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False

    # 2. Check if all elements are integers.
    # We check if the array's data type is a kind of integer.
    if not np.issubdtype(m.dtype, np.integer):
        # If not, check if they are all numerically equal to integers.
        if not np.all(np.equal(np.mod(m, 1), 0)):
            return False

    # 3. Calculate the determinant and check if it's +1 or -1.
    # np.linalg.det returns a float, so we use np.isclose for robust comparison.
    det = np.linalg.det(m)
    return np.isclose(det, 1) or np.isclose(det, -1)    