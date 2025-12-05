use lazy_static::lazy_static;
use std::collections::HashMap;
use serde::Deserialize;

/// Structure matching the JSON data format
#[derive(Deserialize, Clone)]
struct PermutationEntry {
    zeros: Vec<usize>,
    matrix: Vec<Vec<i32>>,
    permutations: Vec<Vec<usize>>,
}

/// Data for one S4 equivalence class group
#[derive(Clone)]
pub struct S4Group {
    /// Vonorm permutations in this group
    pub vonorm_perms: Vec<Vec<usize>>,
    /// Conorm permutations corresponding to each vonorm perm
    pub conorm_perms: Vec<Vec<usize>>,
    /// Transition matrices for each permutation
    pub transition_mats: Vec<Vec<Vec<Vec<i32>>>>,
}

/// Pre-loaded permutation mappings
pub struct PermutationMaps {
    /// Maps (zero_indices, conorm_perm) -> list of unimodular matrices
    pub zero_to_perm_to_mats: HashMap<Vec<usize>, HashMap<Vec<usize>, Vec<Vec<Vec<i32>>>>>,
    /// Maps conorm_perm -> vonorm_perm (pre-computed)
    pub conorm_to_vonorm_perm: HashMap<Vec<usize>, Vec<usize>>,
    /// Maps zero_indices -> S4 groups (precomputed equivalence classes)
    pub zero_to_s4_groups: HashMap<Vec<usize>, HashMap<Vec<usize>, S4Group>>,
}

lazy_static! {
    pub static ref PERMUTATIONS: PermutationMaps = load_permutations();

    /// VONORM_TO_DOT_PRODUCTS matrix (6x6)
    pub static ref VONORM_TO_DOT_PRODUCTS: [[f64; 6]; 6] = [
        [-1.0, -1.0, 0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, -1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0, -1.0, -1.0],
        [1.0, 0.0, 0.0, 1.0, -1.0, -1.0],
        [0.0, -1.0, 0.0, -1.0, 0.0, 1.0],
        [0.0, 0.0, -1.0, -1.0, 1.0, 0.0],
    ];
}

fn load_permutations() -> PermutationMaps {
    // Load unimodular matrices JSON
    let json_str = include_str!("../../lattice/data/unimodular_mats_max_6_det_1_to_perms.json");
    let entries: Vec<serde_json::Value> = serde_json::from_str(json_str)
        .expect("Failed to parse permutations JSON");

    let mut zero_to_perm_to_mats: HashMap<Vec<usize>, HashMap<Vec<usize>, Vec<Vec<Vec<i32>>>>> = HashMap::new();

    // Parse entries
    for entry in entries {
        let zeros: Vec<usize> = serde_json::from_value(entry[0].clone())
            .expect("Failed to parse zeros");
        // Matrix is stored as flat array of 9 elements (3x3 in row-major order)
        let matrix_flat: Vec<i32> = serde_json::from_value(entry[1].clone())
            .expect("Failed to parse matrix");
        // Reshape into 3x3 matrix
        let matrix: Vec<Vec<i32>> = vec![
            matrix_flat[0..3].to_vec(),
            matrix_flat[3..6].to_vec(),
            matrix_flat[6..9].to_vec(),
        ];
        let perms: Vec<Vec<usize>> = serde_json::from_value(entry[2].clone())
            .expect("Failed to parse permutations");

        for perm in perms {
            zero_to_perm_to_mats
                .entry(zeros.clone())
                .or_insert_with(HashMap::new)
                .entry(perm.clone())
                .or_insert_with(Vec::new)
                .push(matrix.clone());
        }
    }

    // Load matching permutations JSON (vonorm_perm, conorm_perm pairs)
    let matching_json = include_str!("../../lattice/data/matching_perms.json");
    let matching_entries: Vec<[Vec<usize>; 2]> = serde_json::from_str(matching_json)
        .expect("Failed to parse matching_perms JSON");

    let mut conorm_to_vonorm_perm: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
    for [vonorm_perm, conorm_perm] in matching_entries {
        // Map conorm_perm -> vonorm_perm (reverse of the file order)
        conorm_to_vonorm_perm.insert(conorm_perm, vonorm_perm);
    }

    // Precompute S4 groupings for each zero pattern
    let mut zero_to_s4_groups: HashMap<Vec<usize>, HashMap<Vec<usize>, S4Group>> = HashMap::new();

    for (zero_idxs, perm_to_mats) in &zero_to_perm_to_mats {
        let mut s4_groups: HashMap<Vec<usize>, S4Group> = HashMap::new();

        // Group conorm permutations by their corresponding vonorm perm's S4 indices
        for (conorm_perm, matrices) in perm_to_mats {
            // Get the corresponding vonorm permutation
            if let Some(vonorm_perm) = conorm_to_vonorm_perm.get(conorm_perm) {
                // Extract S4 indices (first 4 elements, sorted)
                let s4_key: Vec<usize> = {
                    let mut s4 = vonorm_perm[0..4].to_vec();
                    s4.sort();
                    s4
                };

                // Add to the appropriate S4 group
                let group = s4_groups.entry(s4_key).or_insert_with(|| S4Group {
                    vonorm_perms: Vec::new(),
                    conorm_perms: Vec::new(),
                    transition_mats: Vec::new(),
                });

                group.vonorm_perms.push(vonorm_perm.clone());
                group.conorm_perms.push(conorm_perm.clone());
                group.transition_mats.push(matrices.clone());
            }
        }

        zero_to_s4_groups.insert(zero_idxs.clone(), s4_groups);
    }

    PermutationMaps {
        zero_to_perm_to_mats,
        conorm_to_vonorm_perm,
        zero_to_s4_groups,
    }
}

/// Compute conorms from vonorms (first 6 values)
pub fn compute_conorms(vonorms: &[f64; 7]) -> [f64; 6] {
    let mut conorms = [0.0; 6];
    for i in 0..6 {
        for j in 0..6 {
            conorms[i] += 0.5 * VONORM_TO_DOT_PRODUCTS[i][j] * vonorms[j];
        }
    }
    conorms
}

/// Find zero conorm indices with exact equality (for discretized vonorms)
pub fn find_zero_indices_exact(conorms: &[f64; 6]) -> Vec<usize> {
    conorms
        .iter()
        .enumerate()
        .filter_map(|(idx, &cn)| if cn == 0.0 { Some(idx) } else { None })
        .collect()
}

/// Apply a vonorm permutation to get permuted vonorms
fn apply_vonorm_permutation(vonorms: &[f64; 7], perm: &[usize]) -> [f64; 7] {
    [
        vonorms[perm[0]],
        vonorms[perm[1]],
        vonorms[perm[2]],
        vonorms[perm[3]],
        vonorms[perm[4]],
        vonorms[perm[5]],
        vonorms[perm[6]],
    ]
}

/// Result for one S4 equivalence class representative
pub struct S4Representative {
    pub s4_key: Vec<usize>,
    pub permuted_vonorms: [f64; 7],
    pub transition_mats: Vec<Vec<Vec<i32>>>,
}

/// Get one representative from each S4 equivalence class
///
/// Takes the first permutation from each S4 group (no need to find maximal)
pub fn get_s4_representatives(
    vonorms: &[f64; 7],
    zero_indices: &[usize],
) -> Vec<S4Representative> {
    let mut results = Vec::new();

    // Get the S4 groups for this zero pattern
    if let Some(s4_groups) = PERMUTATIONS.zero_to_s4_groups.get(zero_indices) {
        for (s4_key, group) in s4_groups {
            // Just take the first permutation from the group
            if let Some(first_perm) = group.vonorm_perms.first() {
                let permuted = apply_vonorm_permutation(vonorms, first_perm);
                let mats = group.transition_mats[0].clone();

                results.push(S4Representative {
                    s4_key: s4_key.clone(),
                    permuted_vonorms: permuted,
                    transition_mats: mats,
                });
            }
        }
    }

    results
}

/// Find zero conorm indices with tolerance (for float vonorms)
pub fn find_zero_indices_tol(conorms: &[f64; 6], tol: f64) -> Vec<usize> {
    conorms
        .iter()
        .enumerate()
        .filter_map(|(idx, &cn)| if cn.abs() <= tol { Some(idx) } else { None })
        .collect()
}
