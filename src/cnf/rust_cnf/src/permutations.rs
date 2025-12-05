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

/// Pre-loaded permutation mappings
pub struct PermutationMaps {
    /// Maps (zero_indices, conorm_perm) -> list of unimodular matrices
    pub zero_to_perm_to_mats: HashMap<Vec<usize>, HashMap<Vec<usize>, Vec<Vec<Vec<i32>>>>>,
    /// Maps conorm_perm -> vonorm_perm (pre-computed)
    pub conorm_to_vonorm_perm: HashMap<Vec<usize>, Vec<usize>>,
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

    PermutationMaps {
        zero_to_perm_to_mats,
        conorm_to_vonorm_perm,
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

/// Find zero conorm indices with tolerance (for float vonorms)
pub fn find_zero_indices_tol(conorms: &[f64; 6], tol: f64) -> Vec<usize> {
    conorms
        .iter()
        .enumerate()
        .filter_map(|(idx, &cn)| if cn.abs() <= tol { Some(idx) } else { None })
        .collect()
}
