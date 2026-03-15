/// Stabilizer finding and combination for CNF operations
///
/// Stabilizers are the unimodular matrices that leave the canonical vonorms invariant.
/// They form a symmetry group used in both LNF and MNF construction.

use std::collections::HashSet;

use crate::linalg::{mat_mul, mat_to_flat, parse_flat_to_matrices};
use crate::permutations::{compute_conorms, find_zero_indices_exact, find_zero_indices_tol, apply_vonorm_perm, PERMUTATIONS};
use crate::selling::round_to_decimal_places;

// =============================================================================
// Stabilizer Finding
// =============================================================================

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

// =============================================================================
// Stabilizer Combination
// =============================================================================

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

// =============================================================================
// Validation
// =============================================================================

/// Check if vonorms have valid conorms (valid zero pattern)
pub fn has_valid_conorms_exact(vonorms: &[f64; 7]) -> bool {
    let conorms = compute_conorms(vonorms);
    let zero_indices = find_zero_indices_exact(&conorms);

    // Check if zero_indices is in the valid sets
    PERMUTATIONS.zero_to_s4_groups.contains_key(&zero_indices)
}
