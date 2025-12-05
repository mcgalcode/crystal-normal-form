use crate::permutations::{compute_conorms, find_zero_indices_exact, find_zero_indices_tol, PERMUTATIONS};

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
fn get_secondary_vonorm_idx(i: usize, j: usize) -> usize {
    let (min, max) = if i < j { (i, j) } else { (j, i) };
    match (min, max) {
        (0, 1) | (2, 3) => 4,
        (0, 2) | (1, 3) => 5,
        (0, 3) | (1, 2) => 6,
        _ => panic!("Invalid vector pair: ({}, {})", i, j),
    }
}

/// Apply one Selling transformation step for the given vector pair (i, j)
fn apply_selling_step(vonorms: &[f64; 7], conorms: &[f64; 6], conorm_idx: usize) -> [f64; 7] {
    let (i, j) = CONORM_IDX_TO_VECTOR_PAIRS[conorm_idx];

    // Find k and l (the other two indices from {0, 1, 2, 3})
    let all_indices: Vec<usize> = vec![0, 1, 2, 3];
    let other_indices: Vec<usize> = all_indices
        .iter()
        .filter(|&&idx| idx != i && idx != j)
        .copied()
        .collect();
    let (k, l) = (other_indices[0], other_indices[1]);

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
    let mut transform = [[0i32; 3]; 3];
    transform[0][0] = 1;
    transform[1][1] = 1;
    transform[2][2] = 1;  // Start with identity
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
        let det = matrix_determinant_3x3(&step_matrix);
        if det == -1 {
            // Flip signs of the entire matrix
            for row in 0..3 {
                for col in 0..3 {
                    step_matrix[row][col] = -step_matrix[row][col];
                }
            }
        }

        // Multiply left-to-right to match Python: transform @ step_matrix
        transform = matrix_multiply_3x3(&transform, &step_matrix);

        steps += 1;
        if steps >= max_steps {
            panic!("Selling reduction failed to converge after {} steps", max_steps);
        }
    }
}

/// Perform Selling reduction to make all conorms <= 0
fn selling_reduce(vonorms: &[f64; 7], tol: f64, max_steps: usize) -> [f64; 7] {
    let (reduced, _) = selling_reduce_with_transform(vonorms, tol, max_steps);
    reduced
}

/// Fast LNF construction for discretized vonorms (exact equality)
///
/// Returns:
/// - canonical vonorms
/// - zero conorm indices
/// - selling transformation matrix (None if no reduction needed)
/// - sorting permutation matrices (matrices that give the canonical form)
pub fn build_lnf_raw_discretized(vonorms: &[f64; 7]) -> (Vec<f64>, Vec<usize>, Option<Vec<i32>>, Vec<Vec<i32>>) {
    // Step 1: Compute conorms
    let mut current_vonorms = *vonorms;
    let conorms = compute_conorms(&current_vonorms);

    // Step 2: Check if already obtuse (all conorms <= 0)
    let is_obtuse = conorms.iter().all(|&c| c <= 0.0);

    // Step 2b: If not obtuse, apply Selling reduction and track transformation
    let selling_transform = if !is_obtuse {
        let (reduced_vonorms, transform) = selling_reduce_with_transform(&current_vonorms, 0.0, 500);
        current_vonorms = reduced_vonorms;
        Some(transform)
    } else {
        None
    };

    // Step 3: Find zero conorm indices (exact equality for discretized)
    let zero_idxs = find_zero_indices_exact(&compute_conorms(&current_vonorms));

    // Step 4: Get permissible permutations
    let perm_to_mats = match PERMUTATIONS.zero_to_perm_to_mats.get(&zero_idxs) {
        Some(p) => p,
        None => panic!("Invalid zero conorm set: {:?}", zero_idxs),
    };

    // Step 5: Apply all permutations and find lexicographically smallest
    let mut permuted_vonorms: Vec<(Vec<f64>, Vec<Vec<i32>>)> = Vec::new();

    for (conorm_perm, mat_list) in perm_to_mats.iter() {
        // Convert conorm perm to vonorm perm using pre-loaded mapping
        let vonorm_perm = PERMUTATIONS.conorm_to_vonorm_perm.get(conorm_perm)
            .unwrap_or_else(|| panic!("No vonorm permutation found for conorm permutation: {:?}", conorm_perm));

        // Apply permutation
        let mut permuted = [0.0; 7];
        for (i, &idx) in vonorm_perm.iter().enumerate() {
            if idx < 7 && i < 7 {
                permuted[i] = current_vonorms[idx];
            }
        }

        // Clone the matrix list for this permutation
        let matrices: Vec<Vec<i32>> = mat_list.iter().map(|mat| {
            mat.iter().flat_map(|row| row.iter().copied()).collect()
        }).collect();

        permuted_vonorms.push((permuted.to_vec(), matrices));
    }

    // Step 6: Sort to find canonical
    permuted_vonorms.sort_by(|a, b| {
        for (x, y) in a.0.iter().zip(b.0.iter()) {
            match x.partial_cmp(y) {
                Some(std::cmp::Ordering::Equal) => continue,
                Some(ord) => return ord,
                None => continue,
            }
        }
        std::cmp::Ordering::Equal
    });

    let canonical = permuted_vonorms[0].0.clone();

    // Find all equivalent transformations (those that give the canonical form)
    let mut all_equivalent_mats: Vec<Vec<i32>> = Vec::new();
    for (perm_vonorms, matrices) in &permuted_vonorms {
        if perm_vonorms == &canonical {
            all_equivalent_mats.extend(matrices.clone());
        }
    }

    // Convert selling transform to flat array if present
    let selling_flat = selling_transform.map(|mat| {
        mat.iter().flat_map(|row| row.iter().copied()).collect()
    });

    (canonical, zero_idxs, selling_flat, all_equivalent_mats)
}

/// Round a float to specified number of decimal places
fn round_to_decimal_places(value: f64, decimal_places: u32) -> f64 {
    let multiplier = 10f64.powi(decimal_places as i32);
    (value * multiplier).round() / multiplier
}

/// Fast LNF construction for float vonorms (tolerance-based comparisons)
///
/// Returns:
/// - canonical vonorms
/// - zero conorm indices
/// - selling transformation matrix (None if no reduction needed)
/// - sorting permutation matrices (matrices that give the canonical form)
pub fn build_lnf_raw_float(vonorms: &[f64; 7], tol: f64) -> (Vec<f64>, Vec<usize>, Option<Vec<i32>>, Vec<Vec<i32>>) {
    // Step 1: Compute conorms
    let mut current_vonorms = *vonorms;
    let conorms = compute_conorms(&current_vonorms);

    // Step 2: Check if already obtuse (all conorms <= tol)
    let is_obtuse = conorms.iter().all(|&c| c <= tol);

    // Step 2b: If not obtuse, apply Selling reduction and track transformation
    let selling_transform = if !is_obtuse {
        let (reduced_vonorms, transform) = selling_reduce_with_transform(&current_vonorms, tol, 500);
        current_vonorms = reduced_vonorms;
        Some(transform)
    } else {
        None
    };

    // Step 3: Find zero conorm indices (tolerance-based for floats)
    let zero_idxs = find_zero_indices_tol(&compute_conorms(&current_vonorms), tol);

    // Step 4: Get permissible permutations
    let perm_to_mats = match PERMUTATIONS.zero_to_perm_to_mats.get(&zero_idxs) {
        Some(p) => p,
        None => panic!("Invalid zero conorm set: {:?}", zero_idxs),
    };

    // Step 5: Apply all permutations and round for sorting (to handle float precision)
    let sorting_decimal_places = 5; // Match Python's sorting_dec_places
    let mut permuted_vonorms: Vec<(Vec<f64>, Vec<f64>, Vec<Vec<i32>>)> = Vec::new();

    for (conorm_perm, mat_list) in perm_to_mats.iter() {
        // Convert conorm perm to vonorm perm using pre-loaded mapping
        let vonorm_perm = PERMUTATIONS.conorm_to_vonorm_perm.get(conorm_perm)
            .unwrap_or_else(|| panic!("No vonorm permutation found for conorm permutation: {:?}", conorm_perm));

        // Apply permutation
        let mut permuted = [0.0; 7];
        for (i, &idx) in vonorm_perm.iter().enumerate() {
            if idx < 7 && i < 7 {
                permuted[i] = current_vonorms[idx];
            }
        }

        // Round for sorting to avoid float precision issues
        let rounded: Vec<f64> = permuted.iter()
            .map(|&v| round_to_decimal_places(v, sorting_decimal_places))
            .collect();

        // Clone the matrix list for this permutation
        let matrices: Vec<Vec<i32>> = mat_list.iter().map(|mat| {
            mat.iter().flat_map(|row| row.iter().copied()).collect()
        }).collect();

        permuted_vonorms.push((rounded, permuted.to_vec(), matrices));
    }

    // Step 6: Sort by rounded values to find canonical
    permuted_vonorms.sort_by(|a, b| {
        for (x, y) in a.0.iter().zip(b.0.iter()) {
            match x.partial_cmp(y) {
                Some(std::cmp::Ordering::Equal) => continue,
                Some(ord) => return ord,
                None => continue,
            }
        }
        std::cmp::Ordering::Equal
    });

    // Use the unrounded canonical vonorms
    let canonical = permuted_vonorms[0].1.clone();
    let canonical_rounded = &permuted_vonorms[0].0;

    // Find all equivalent transformations (compare rounded values)
    let mut all_equivalent_mats: Vec<Vec<i32>> = Vec::new();
    for (rounded, _unrounded, matrices) in &permuted_vonorms {
        if rounded == canonical_rounded {
            all_equivalent_mats.extend(matrices.clone());
        }
    }

    // Convert selling transform to flat array if present
    let selling_flat = selling_transform.map(|mat| {
        mat.iter().flat_map(|row| row.iter().copied()).collect()
    });

    (canonical, zero_idxs, selling_flat, all_equivalent_mats)
}

/// Fast stabilizer computation for discretized vonorms
///
/// Returns a flat Vec of matrices (each matrix is 9 elements in row-major order)
/// to avoid Python object creation overhead.
/// Internal helper for finding stabilizers with custom equality check
fn find_stabilizers_internal<F>(
    vonorms: &[f64; 7],
    zero_idxs: Vec<usize>,
    equals: F,
) -> Vec<i32>
where
    F: Fn(&[f64; 7], &[f64; 7]) -> bool,
{
    // Get permutations from pre-computed mapping
    let perm_to_mats = match PERMUTATIONS.zero_to_perm_to_mats.get(&zero_idxs) {
        Some(p) => p,
        None => return Vec::new(), // No valid permutations
    };

    let mut stabilizers = Vec::new();

    // Iterate over conorm permutations and check which preserve vonorms
    for (conorm_perm, mat_list) in perm_to_mats.iter() {
        // Get vonorm permutation
        let vonorm_perm = match PERMUTATIONS.conorm_to_vonorm_perm.get(conorm_perm) {
            Some(p) => p,
            None => continue,
        };

        // Apply permutation
        let mut permuted = [0.0; 7];
        for (i, &idx) in vonorm_perm.iter().enumerate() {
            if idx < 7 && i < 7 {
                permuted[i] = vonorms[idx];
            }
        }

        // Check if permutation preserves vonorms using custom equality
        let matches = equals(&permuted, vonorms);
        if matches {
            // Add all matrices for this permutation (flattened to row-major)
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

/// Calculate determinant of a 3x3 i32 matrix
#[inline]
fn matrix_determinant_3x3(m: &[[i32; 3]; 3]) -> i32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// 3x3 matrix multiplication for i32 matrices
#[inline]
fn matrix_multiply_3x3(a: &[[i32; 3]; 3], b: &[[i32; 3]; 3]) -> [[i32; 3]; 3] {
    let mut result = [[0i32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

/// Combine and deduplicate stabilizer matrices
///
/// Computes all combinations s1[i] @ middle @ s2[j] and deduplicates.
/// This replaces the Python einsum + deduplication code.
///
/// Returns a flat Vec of unique matrices (each matrix is 9 elements in row-major order)
pub fn combine_stabilizers(
    s1_flat: &[i32],
    s2_flat: &[i32],
    middle: &[[i32; 3]; 3],
) -> Vec<i32> {
    use std::collections::HashSet;

    // Convert flat arrays to matrix arrays
    let n1 = s1_flat.len() / 9;
    let n2 = s2_flat.len() / 9;

    let mut s1_matrices = Vec::with_capacity(n1);
    for i in 0..n1 {
        let offset = i * 9;
        let mat = [
            [s1_flat[offset], s1_flat[offset + 1], s1_flat[offset + 2]],
            [s1_flat[offset + 3], s1_flat[offset + 4], s1_flat[offset + 5]],
            [s1_flat[offset + 6], s1_flat[offset + 7], s1_flat[offset + 8]],
        ];
        s1_matrices.push(mat);
    }

    let mut s2_matrices = Vec::with_capacity(n2);
    for i in 0..n2 {
        let offset = i * 9;
        let mat = [
            [s2_flat[offset], s2_flat[offset + 1], s2_flat[offset + 2]],
            [s2_flat[offset + 3], s2_flat[offset + 4], s2_flat[offset + 5]],
            [s2_flat[offset + 6], s2_flat[offset + 7], s2_flat[offset + 8]],
        ];
        s2_matrices.push(mat);
    }

    // Use HashSet for automatic deduplication
    // We'll use the flattened matrix as the key
    let mut unique_matrices: HashSet<[i32; 9]> = HashSet::new();

    // Compute all combinations
    for s1 in &s1_matrices {
        // Pre-compute s1 @ middle
        let temp = matrix_multiply_3x3(s1, middle);

        for s2 in &s2_matrices {
            // Compute (s1 @ middle) @ s2
            let result = matrix_multiply_3x3(&temp, s2);

            // Flatten and add to set
            let flat = [
                result[0][0], result[0][1], result[0][2],
                result[1][0], result[1][1], result[1][2],
                result[2][0], result[2][1], result[2][2],
            ];
            unique_matrices.insert(flat);
        }
    }

    // Convert HashSet to flat Vec
    let mut result = Vec::with_capacity(unique_matrices.len() * 9);
    for mat in unique_matrices {
        result.extend_from_slice(&mat);
    }

    result
}
