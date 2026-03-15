//! Linear algebra utilities for 3x3 integer matrices.
//!
//! This module provides optimized operations for the unimodular matrices
//! used in lattice transformations (det = ±1).

// =============================================================================
// Matrix Multiplication
// =============================================================================

/// Multiply two 3x3 integer matrices.
#[inline]
pub fn mat_mul(a: &[[i32; 3]; 3], b: &[[i32; 3]; 3]) -> [[i32; 3]; 3] {
    let mut result = [[0i32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    result
}

// =============================================================================
// Matrix Inversion
// =============================================================================

/// Invert a 3x3 unimodular matrix (det = ±1).
///
/// Returns exact integer result since det = ±1 means inverse = adjugate * det.
/// Panics in debug mode if the matrix is not unimodular.
#[inline]
pub fn mat_inv(m: &[[i32; 3]; 3]) -> [[i32; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    debug_assert!(det == 1 || det == -1, "Matrix is not unimodular (det = {})", det);

    // Adjugate matrix
    let adj = [
        [
            m[1][1] * m[2][2] - m[1][2] * m[2][1],
            m[0][2] * m[2][1] - m[0][1] * m[2][2],
            m[0][1] * m[1][2] - m[0][2] * m[1][1],
        ],
        [
            m[1][2] * m[2][0] - m[1][0] * m[2][2],
            m[0][0] * m[2][2] - m[0][2] * m[2][0],
            m[0][2] * m[1][0] - m[0][0] * m[1][2],
        ],
        [
            m[1][0] * m[2][1] - m[1][1] * m[2][0],
            m[0][1] * m[2][0] - m[0][0] * m[2][1],
            m[0][0] * m[1][1] - m[0][1] * m[1][0],
        ],
    ];

    // inv = adj * det (exact for det = ±1)
    let mut inv = [[0i32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            inv[i][j] = adj[i][j] * det;
        }
    }
    inv
}

/// Invert a 3x3 integer matrix, returning f64 result.
///
/// Use this only when you need floating-point precision (e.g., coordinate
/// transformations with modulo operations). For unimodular matrices,
/// prefer `mat_inv` for exact integer results.
pub fn mat_inv_f64(mat: &[[i32; 3]; 3]) -> [[f64; 3]; 3] {
    let m: [[f64; 3]; 3] = [
        [mat[0][0] as f64, mat[0][1] as f64, mat[0][2] as f64],
        [mat[1][0] as f64, mat[1][1] as f64, mat[1][2] as f64],
        [mat[2][0] as f64, mat[2][1] as f64, mat[2][2] as f64],
    ];

    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    assert!(det.abs() > 1e-10, "Matrix is singular and cannot be inverted");

    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det,
        ],
    ]
}

// =============================================================================
// Determinant
// =============================================================================

/// Calculate determinant of a 3x3 integer matrix.
#[inline]
pub fn mat_det(m: &[[i32; 3]; 3]) -> i32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

// =============================================================================
// Constants
// =============================================================================

/// 3x3 identity matrix
pub const IDENTITY_3X3: [[i32; 3]; 3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

// =============================================================================
// Array Conversion Helpers
// =============================================================================

/// Convert a flat 9-element slice to a 3x3 matrix
#[inline]
pub fn flat9_to_mat3x3(flat: &[i32]) -> [[i32; 3]; 3] {
    debug_assert!(flat.len() >= 9, "flat array must have at least 9 elements");
    [
        [flat[0], flat[1], flat[2]],
        [flat[3], flat[4], flat[5]],
        [flat[6], flat[7], flat[8]],
    ]
}

/// Convert a slice of f64 to a fixed-size 7-element array (for vonorms)
#[inline]
pub fn slice_to_vonorms(slice: &[f64]) -> [f64; 7] {
    debug_assert!(slice.len() >= 7, "slice must have at least 7 elements");
    [slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6]]
}

/// Parse a flat i32 slice into a Vec of 3x3 matrices
pub fn parse_flat_to_matrices(flat: &[i32]) -> Vec<[[i32; 3]; 3]> {
    let n = flat.len() / 9;
    let mut matrices = Vec::with_capacity(n);
    for i in 0..n {
        let o = i * 9;
        matrices.push([
            [flat[o], flat[o + 1], flat[o + 2]],
            [flat[o + 3], flat[o + 4], flat[o + 5]],
            [flat[o + 6], flat[o + 7], flat[o + 8]],
        ]);
    }
    matrices
}

/// Convert flat stabilizer matrices to 3D nested Vec for Python
pub fn flat_matrices_to_nested(flat: &[i32]) -> Vec<Vec<Vec<i32>>> {
    let n_matrices = flat.len() / 9;
    let mut matrices = Vec::with_capacity(n_matrices);
    for i in 0..n_matrices {
        let start = i * 9;
        matrices.push(vec![
            flat[start..start+3].to_vec(),
            flat[start+3..start+6].to_vec(),
            flat[start+6..start+9].to_vec(),
        ]);
    }
    matrices
}

// =============================================================================
// Matrix Conversion
// =============================================================================

/// Flatten a 3x3 matrix to a 9-element array (for use as HashMap key).
#[inline]
pub fn mat_to_flat(m: &[[i32; 3]; 3]) -> [i32; 9] {
    [
        m[0][0], m[0][1], m[0][2],
        m[1][0], m[1][1], m[1][2],
        m[2][0], m[2][1], m[2][2],
    ]
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat_mul_identity() {
        let identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        let b = [[2, 3, 4], [5, 6, 7], [8, 9, 10]];
        assert_eq!(mat_mul(&identity, &b), b);
        assert_eq!(mat_mul(&b, &identity), b);
    }

    #[test]
    fn test_mat_mul_associative() {
        let a = [[1, 2, 0], [0, 1, 1], [1, 0, 1]];
        let b = [[0, 1, 0], [1, 0, 0], [0, 0, 1]];
        let c = [[1, 0, 1], [0, 1, 0], [1, 1, 0]];

        let ab_c = mat_mul(&mat_mul(&a, &b), &c);
        let a_bc = mat_mul(&a, &mat_mul(&b, &c));
        assert_eq!(ab_c, a_bc);
    }

    #[test]
    fn test_mat_inv_permutation() {
        // Simple permutation matrix (det = -1)
        let m = [[0, 1, 0], [1, 0, 0], [0, 0, -1]];
        let inv = mat_inv(&m);
        let product = mat_mul(&m, &inv);
        assert_eq!(product, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    }

    #[test]
    fn test_mat_inv_unimodular() {
        // Upper triangular unimodular (det = 1)
        let m = [[1, 2, 3], [0, 1, 4], [0, 0, 1]];
        let inv = mat_inv(&m);
        let product = mat_mul(&m, &inv);
        assert_eq!(product, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    }

    #[test]
    fn test_mat_inv_f64_matches_integer() {
        let m = [[0, 1, 0], [1, 0, 0], [0, 0, -1]];
        let inv_i32 = mat_inv(&m);
        let inv_f64 = mat_inv_f64(&m);

        for i in 0..3 {
            for j in 0..3 {
                assert!((inv_i32[i][j] as f64 - inv_f64[i][j]).abs() < 1e-10);
            }
        }
    }
}
