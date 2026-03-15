/// LNF (Lattice Normal Form) construction
///
/// This module builds the canonical LNF representation of a lattice
/// using Selling reduction and lexicographic minimization.

use crate::permutations::{compute_conorms, find_zero_indices_exact, find_zero_indices_tol, apply_vonorm_perm, PERMUTATIONS};
use crate::selling::{selling_reduce_with_transform, round_to_decimal_places};

// Re-export stabilizer functions for backward compatibility
pub use crate::stabilizers::{
    find_stabilizers_raw,
    find_stabilizers_raw_float,
    combine_stabilizers,
    combine_middle_and_s2_stabilizers,
    has_valid_conorms_exact,
};

// Re-export validation functions from selling
pub use crate::selling::{is_obtuse, is_superbasis_exact};

// =============================================================================
// LNF Construction
// =============================================================================

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
    let already_obtuse = conorms.iter().all(|&c| c <= tolerance);

    // Step 2: If not obtuse, apply Selling reduction
    let selling_transform = if !already_obtuse {
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
