/// Selling reduction algorithm for lattice vectors
///
/// This module implements the Selling reduction algorithm which transforms
/// a superbasis into its obtuse (reduced) form where all pairwise dot products
/// are non-positive.

use crate::linalg::{mat_det, mat_mul, IDENTITY_3X3};
use crate::permutations::compute_conorms;

/// Mapping from conorm index to vector pair
pub const CONORM_IDX_TO_VECTOR_PAIRS: [(usize, usize); 6] = [
    (0, 1), // conorm 0
    (0, 2), // conorm 1
    (0, 3), // conorm 2
    (1, 2), // conorm 3
    (1, 3), // conorm 4
    (2, 3), // conorm 5
];

/// Mapping from vector pair to secondary vonorm index
#[inline]
pub fn get_secondary_vonorm_idx(i: usize, j: usize) -> usize {
    let (min, max) = if i < j { (i, j) } else { (j, i) };
    match (min, max) {
        (0, 1) | (2, 3) => 4,
        (0, 2) | (1, 3) => 5,
        (0, 3) | (1, 2) => 6,
        _ => panic!("Invalid vector pair: ({}, {})", i, j),
    }
}

/// Round a float to specified number of decimal places
#[inline]
pub fn round_to_decimal_places(value: f64, decimal_places: u32) -> f64 {
    let multiplier = 10f64.powi(decimal_places as i32);
    (value * multiplier).round() / multiplier
}

/// Find the other two indices from {0,1,2,3} given i and j
#[inline]
fn get_other_indices(i: usize, j: usize) -> (usize, usize) {
    match (i, j) {
        (0, 1) | (1, 0) => (2, 3),
        (0, 2) | (2, 0) => (1, 3),
        (0, 3) | (3, 0) => (1, 2),
        (1, 2) | (2, 1) => (0, 3),
        (1, 3) | (3, 1) => (0, 2),
        (2, 3) | (3, 2) => (0, 1),
        _ => panic!("Invalid indices: ({}, {})", i, j),
    }
}

/// Apply one Selling transformation step for the given vector pair (i, j)
fn apply_selling_step(vonorms: &[f64; 7], conorms: &[f64; 6], conorm_idx: usize) -> [f64; 7] {
    let (i, j) = CONORM_IDX_TO_VECTOR_PAIRS[conorm_idx];
    let (k, l) = get_other_indices(i, j);

    let mut new_vonorms = [0.0; 7];

    // Two vonorms remain the same
    new_vonorms[i] = vonorms[i];
    new_vonorms[j] = vonorms[j];

    // Two vonorm pairs swap
    // pair 1: u_k = v_ik, u_ik = v_k
    let ik_idx = get_secondary_vonorm_idx(i, k);
    new_vonorms[k] = vonorms[ik_idx];
    new_vonorms[ik_idx] = vonorms[k];

    // pair 2: u_l = v_il, u_il = v_l
    let il_idx = get_secondary_vonorm_idx(i, l);
    new_vonorms[l] = vonorms[il_idx];
    new_vonorms[il_idx] = vonorms[l];

    // The i,j vonorm is reduced by 4 x v_i dot v_j
    let ij_idx = get_secondary_vonorm_idx(i, j);
    let conorm_v_i_dot_v_j = conorms[conorm_idx];
    new_vonorms[ij_idx] = vonorms[ij_idx] - 4.0 * conorm_v_i_dot_v_j;

    new_vonorms
}

/// Get the column vector for a given label (0, 1, 2, or 3)
fn get_label_column(label: usize) -> [i32; 3] {
    match label {
        0 => [1, 0, 0],
        1 => [0, 1, 0],
        2 => [0, 0, 1],
        3 => [-1, -1, -1],
        _ => panic!("Invalid label: {}", label),
    }
}

/// Add two column vectors
fn add_columns(a: [i32; 3], b: [i32; 3]) -> [i32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Negate a column vector
fn negate_column(a: [i32; 3]) -> [i32; 3] {
    [-a[0], -a[1], -a[2]]
}

/// Get the transformation matrix for a Selling step
/// Returns the 3x3 unimodular matrix corresponding to the Selling transformation
/// Based on Lemma A.1 (Kurlin et al.) and implemented in selling_transform_matrix.py
///
/// Note: i and j are indices in range [0, 1, 2, 3] for the 4 superbasis vectors.
/// The actual lattice has 3 basis vectors (indices 0, 1, 2).
/// Vector 3 represents the diagonal vector [-1, -1, -1].
fn get_selling_step_matrix(i: usize, j: usize) -> [[i32; 3]; 3] {
    // Build matrix column by column
    // For each lattice vector (0, 1, 2), determine its transformation
    let mut columns = Vec::new();

    for lattice_vec_label in 0..3 {
        let col = if lattice_vec_label == i {
            // If this lattice vector is i, negate it
            negate_column(get_label_column(lattice_vec_label))
        } else if lattice_vec_label == j {
            // If this lattice vector is j, keep it as is
            get_label_column(lattice_vec_label)
        } else {
            // Otherwise, add the i-th column to it
            add_columns(get_label_column(lattice_vec_label), get_label_column(i))
        };
        columns.push(col);
    }

    // Transpose to get row-major matrix
    let mut mat = [[0i32; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            mat[row][col] = columns[col][row];
        }
    }

    mat
}

/// Perform Selling reduction with transformation tracking
/// Returns (reduced_vonorms, cumulative_transformation_matrix)
pub fn selling_reduce_with_transform(vonorms: &[f64; 7], tol: f64, max_steps: usize) -> ([f64; 7], [[i32; 3]; 3]) {
    let mut current = *vonorms;
    let mut transform = IDENTITY_3X3;
    let mut steps = 0;

    // Python uses sorting_decimal_places=4 when rounding conorms before selecting
    let sorting_decimal_places = 4;

    loop {
        // Compute conorms
        let conorms = compute_conorms(&current);

        // Check if obtuse (all conorms <= tol)
        let is_obtuse = conorms.iter().all(|&c| c <= tol);
        if is_obtuse {
            return (current, transform);
        }

        // Find the most positive conorm, matching Python's selection logic:
        // 1. Round conorms to sorting_decimal_places
        // 2. Sort by (rounded_conorm, (i, j)) in descending order
        // 3. Pick the first one
        // This matches Python's behavior in selling_reducer.py:select_pair_for_reduction
        let mut pairs: Vec<(i64, usize, usize, usize)> = Vec::new();

        for (idx, &conorm) in conorms.iter().enumerate() {
            if conorm > tol {
                let rounded_conorm = round_to_decimal_places(conorm, sorting_decimal_places);
                // Convert to integer for exact comparison (multiply by 10^4 and round)
                let rounded_int = (rounded_conorm * 10000.0).round() as i64;
                let (i, j) = CONORM_IDX_TO_VECTOR_PAIRS[idx];
                pairs.push((rounded_int, i, j, idx));
            }
        }

        if pairs.is_empty() {
            return (current, transform);
        }

        // Sort by (rounded_conorm, i, j) in descending order
        pairs.sort_by(|a, b| {
            // First compare by rounded conorm (descending)
            match b.0.cmp(&a.0) {
                std::cmp::Ordering::Equal => {
                    // If tied, compare by (i, j) tuple (descending)
                    match b.1.cmp(&a.1) {
                        std::cmp::Ordering::Equal => b.2.cmp(&a.2),
                        other => other,
                    }
                }
                other => other,
            }
        });

        let selected_idx = pairs[0].3;
        let (i, j) = CONORM_IDX_TO_VECTOR_PAIRS[selected_idx];

        // Apply the Selling step to vonorms using the selected conorm index
        current = apply_selling_step(&current, &conorms, selected_idx);

        // Update transformation matrix
        let mut step_matrix = get_selling_step_matrix(i, j);

        // Check determinant - if -1, flip signs (matching Python behavior)
        let det = mat_det(&step_matrix);
        if det == -1 {
            // Flip signs of the entire matrix
            for row in 0..3 {
                for col in 0..3 {
                    step_matrix[row][col] = -step_matrix[row][col];
                }
            }
        }

        // Multiply left-to-right to match Python: transform @ step_matrix
        transform = mat_mul(&transform, &step_matrix);

        steps += 1;
        if steps >= max_steps {
            panic!("Selling reduction failed to converge after {} steps", max_steps);
        }
    }
}

/// Check if vonorms represent an obtuse superbasis
/// Matches Python's VonormList.is_obtuse()
pub fn is_obtuse(vonorms: &[f64; 7]) -> bool {
    let conorms = compute_conorms(vonorms);
    conorms.iter().all(|&c| c <= 0.0)
}

/// Check if vonorms represent a valid superbasis
/// Matches Python's VonormList.is_superbasis_exact()
pub fn is_superbasis_exact(vonorms: &[f64; 7]) -> bool {
    // Sum of primary vonorms (indices 0-3)
    let primary_sum: f64 = vonorms[0..4].iter().sum();

    // Sum of secondary vonorms (indices 4-6)
    let secondary_sum: f64 = vonorms[4..7].iter().sum();

    // Check if they're exactly equal (for integer vonorms, this should be exact)
    primary_sum == secondary_sum
}
