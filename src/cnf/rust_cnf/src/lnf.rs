use std::collections::HashSet;

use crate::linalg::{mat_det, mat_mul, mat_to_flat, parse_flat_to_matrices, IDENTITY_3X3};
use crate::permutations::{compute_conorms, find_zero_indices_exact, find_zero_indices_tol, apply_vonorm_perm, PERMUTATIONS};

/// Mapping from conorm index to vector pair
const CONORM_IDX_TO_VECTOR_PAIRS: [(usize, usize); 6] = [
    (0, 1), // conorm 0
    (0, 2), // conorm 1
    (0, 3), // conorm 2
    (1, 2), // conorm 3
    (1, 3), // conorm 4
    (2, 3), // conorm 5
];

/// Mapping from vector pair to secondary vonorm index
#[inline]
fn get_secondary_vonorm_idx(i: usize, j: usize) -> usize {
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
fn round_to_decimal_places(value: f64, decimal_places: u32) -> f64 {
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
fn selling_reduce_with_transform(vonorms: &[f64; 7], tol: f64, max_steps: usize) -> ([f64; 7], [[i32; 3]; 3]) {
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

/// Internal LNF construction with optional tolerance
///
/// If tol is None, uses exact equality (for discretized vonorms).
/// If tol is Some(t), uses tolerance-based comparisons with rounding.
fn build_lnf_raw_internal(
    vonorms: &[f64; 7],
    tol: Option<f64>,
) -> (Vec<f64>, Vec<usize>, Option<Vec<i32>>, Vec<Vec<i32>>) {
    let tolerance = tol.unwrap_or(0.0);
    let use_rounding = tol.is_some();

    // Step 1: Compute conorms and check if already obtuse
    let mut current_vonorms = *vonorms;
    let conorms = compute_conorms(&current_vonorms);
    let is_obtuse = conorms.iter().all(|&c| c <= tolerance);

    // Step 2: If not obtuse, apply Selling reduction
    let selling_transform = if !is_obtuse {
        let (reduced, transform) = selling_reduce_with_transform(&current_vonorms, tolerance, 500);
        current_vonorms = reduced;
        Some(transform)
    } else {
        None
    };

    // Step 3: Find zero conorm indices
    let final_conorms = compute_conorms(&current_vonorms);
    let zero_idxs = if use_rounding {
        find_zero_indices_tol(&final_conorms, tolerance)
    } else {
        find_zero_indices_exact(&final_conorms)
    };

    // Step 4: Get permissible permutations
    let perm_to_mats = PERMUTATIONS.zero_to_perm_to_mats.get(&zero_idxs)
        .unwrap_or_else(|| panic!("Invalid zero conorm set: {:?}", zero_idxs));

    // Step 5: Apply all permutations
    // Store (sort_key, actual_vonorms, matrices)
    let sorting_decimal_places = 5;
    let mut candidates: Vec<(Vec<f64>, Vec<f64>, Vec<Vec<i32>>)> = Vec::new();

    for (conorm_perm, mat_list) in perm_to_mats.iter() {
        let vonorm_perm = PERMUTATIONS.conorm_to_vonorm_perm.get(conorm_perm)
            .unwrap_or_else(|| panic!("No vonorm perm for conorm perm: {:?}", conorm_perm));

        let permuted = apply_vonorm_perm(&current_vonorms, vonorm_perm);
        let permuted_vec = permuted.to_vec();

        let sort_key = if use_rounding {
            permuted.iter().map(|&v| round_to_decimal_places(v, sorting_decimal_places)).collect()
        } else {
            permuted_vec.clone()
        };

        let matrices: Vec<Vec<i32>> = mat_list.iter()
            .map(|mat| mat.iter().flat_map(|row| row.iter().copied()).collect())
            .collect();

        candidates.push((sort_key, permuted_vec, matrices));
    }

    // Step 6: Sort to find canonical (by sort_key)
    candidates.sort_by(|a, b| {
        a.0.iter().zip(b.0.iter())
            .find_map(|(x, y)| match x.partial_cmp(y) {
                Some(std::cmp::Ordering::Equal) => None,
                ord => ord,
            })
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let canonical = candidates[0].1.clone();
    let canonical_key = &candidates[0].0;

    // Collect all equivalent matrices
    let all_equivalent_mats: Vec<Vec<i32>> = candidates.iter()
        .filter(|(key, _, _)| key == canonical_key)
        .flat_map(|(_, _, mats)| mats.clone())
        .collect();

    // Convert selling transform to flat array
    let selling_flat = selling_transform.map(|mat| {
        mat.iter().flat_map(|row| row.iter().copied()).collect()
    });

    (canonical, zero_idxs, selling_flat, all_equivalent_mats)
}

/// Fast LNF construction for discretized vonorms (exact equality)
pub fn build_lnf_raw_discretized(vonorms: &[f64; 7]) -> (Vec<f64>, Vec<usize>, Option<Vec<i32>>, Vec<Vec<i32>>) {
    build_lnf_raw_internal(vonorms, None)
}

/// Fast LNF construction for float vonorms (tolerance-based comparisons)
pub fn build_lnf_raw_float(vonorms: &[f64; 7], tol: f64) -> (Vec<f64>, Vec<usize>, Option<Vec<i32>>, Vec<Vec<i32>>) {
    build_lnf_raw_internal(vonorms, Some(tol))
}

/// Internal helper for finding stabilizers with custom equality check
fn find_stabilizers_internal<F>(
    vonorms: &[f64; 7],
    zero_idxs: Vec<usize>,
    equals: F,
) -> Vec<i32>
where
    F: Fn(&[f64; 7], &[f64; 7]) -> bool,
{
    let perm_to_mats = match PERMUTATIONS.zero_to_perm_to_mats.get(&zero_idxs) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let mut stabilizers = Vec::new();

    for (conorm_perm, mat_list) in perm_to_mats.iter() {
        let vonorm_perm = match PERMUTATIONS.conorm_to_vonorm_perm.get(conorm_perm) {
            Some(p) => p,
            None => continue,
        };

        let permuted = apply_vonorm_perm(vonorms, vonorm_perm);

        if equals(&permuted, vonorms) {
            for mat in mat_list {
                for row in mat {
                    stabilizers.extend_from_slice(row);
                }
            }
        }
    }

    stabilizers
}

/// Find stabilizers for discretized vonorms (exact equality)
pub fn find_stabilizers_raw(vonorms: &[f64; 7]) -> Vec<i32> {
    let raw_conorms = compute_conorms(vonorms);
    let zero_idxs = find_zero_indices_exact(&raw_conorms);

    find_stabilizers_internal(vonorms, zero_idxs, |a, b| a == b)
}

/// Find stabilizers for float vonorms (tolerance-based equality)
/// Rounds to 8 decimals before comparing, matching Python's behavior
pub fn find_stabilizers_raw_float(vonorms: &[f64; 7], tol: f64) -> Vec<i32> {
    let raw_conorms = compute_conorms(vonorms);
    let zero_idxs = find_zero_indices_tol(&raw_conorms, tol);

    // Round to 8 decimals for comparison (matches Python behavior)
    let comparison_decimal_places = 8;
    find_stabilizers_internal(vonorms, zero_idxs, |a, b| {
        let rounded_a: Vec<f64> = a.iter()
            .map(|&v| round_to_decimal_places(v, comparison_decimal_places))
            .collect();
        let rounded_b: Vec<f64> = b.iter()
            .map(|&v| round_to_decimal_places(v, comparison_decimal_places))
            .collect();
        rounded_a == rounded_b
    })
}

/// Combine and deduplicate stabilizer matrices
///
/// Computes all combinations s1[i] @ middle @ s2[j] and deduplicates.
/// Returns a flat Vec of unique matrices (9 elements per matrix in row-major order)
pub fn combine_stabilizers(
    s1_flat: &[i32],
    s2_flat: &[i32],
    middle: &[[i32; 3]; 3],
) -> Vec<i32> {
    let s1_matrices = parse_flat_to_matrices(s1_flat);
    let s2_matrices = parse_flat_to_matrices(s2_flat);

    let mut unique: HashSet<[i32; 9]> = HashSet::new();

    for s1 in &s1_matrices {
        let temp = mat_mul(s1, middle);
        for s2 in &s2_matrices {
            unique.insert(mat_to_flat(&mat_mul(&temp, s2)));
        }
    }

    unique.into_iter().flat_map(|m| m).collect()
}

/// Combine middle transformation with s2 stabilizers only (optimized version)
///
/// Computes middle @ s2 for each s2, skipping the s1 orbit.
/// ~5x faster than full combine_stabilizers for CNF construction.
pub fn combine_middle_and_s2_stabilizers(
    s2_flat: &[i32],
    middle: &[[i32; 3]; 3],
) -> Vec<i32> {
    let s2_matrices = parse_flat_to_matrices(s2_flat);

    let unique: HashSet<[i32; 9]> = s2_matrices.iter()
        .map(|s2| mat_to_flat(&mat_mul(middle, s2)))
        .collect();

    unique.into_iter().flat_map(|m| m).collect()
}

/// Check if vonorms have valid conorms
pub fn has_valid_conorms_exact(vonorms: &[f64; 7]) -> bool {
    let conorms = compute_conorms(vonorms);
    let zero_indices = find_zero_indices_exact(&conorms);

    // Check if zero_indices is in the valid sets (matching Python's ZERO_CONORM_SETS_TO_PERMUTATIONS_TO_UNIMOD_MATS)
    PERMUTATIONS.zero_to_s4_groups.contains_key(&zero_indices)
}

/// Check if vonorms represent an obtuse superbasis
/// Matches Python's VonormList.is_obtuse()
/// Python: raw_conorms = (1/2) * VONORM_TO_DOT_PRODUCTS @ vonorms[:6]
///         return all(c <= 0 for c in raw_conorms)
pub fn is_obtuse(vonorms: &[f64; 7]) -> bool {
    let conorms = compute_conorms(vonorms);
    conorms.iter().all(|&c| c <= 0.0)
}

/// Check if vonorms represent a valid superbasis
/// Matches Python's VonormList.is_superbasis_exact()
/// Python implementation: return self.primary_sum() == self.secondary_sum()
pub fn is_superbasis_exact(vonorms: &[f64; 7]) -> bool {
    // Sum of primary vonorms (indices 0-3)
    let primary_sum: f64 = vonorms[0..4].iter().sum();

    // Sum of secondary vonorms (indices 4-6)
    let secondary_sum: f64 = vonorms[4..7].iter().sum();

    // Check if they're exactly equal (for integer vonorms, this should be exact)
    primary_sum == secondary_sum
}
