/// MNF (Motif Normal Form) construction in Rust
///
/// This module implements the vectorized MNF construction algorithm,
/// which finds the lexicographically smallest representation of an atomic motif.

use std::cmp::Ordering;

use crate::linalg::{mat_inv_f64, parse_flat_to_matrices};

// =============================================================================
// Helper Functions (pub(crate) for use by neighbors module)
// =============================================================================

/// Multiply a 3x3 matrix by a 3xN coordinate matrix
pub(crate) fn matrix_mult_coords(mat: &[[f64; 3]; 3], coords: &[f64], n_atoms: usize) -> Vec<f64> {
    let mut result = vec![0.0; 3 * n_atoms];

    for i in 0..n_atoms {
        let offset = i * 3;
        for row in 0..3 {
            result[offset + row] = 0.0;
            for k in 0..3 {
                result[offset + row] += mat[row][k] * coords[offset + k];
            }
        }
    }

    result
}

/// Move coordinates into bounds [0, mod) using modulo
pub(crate) fn move_into_bounds(coords: &mut [f64], mod_val: f64) {
    for c in coords.iter_mut() {
        // Round to 6 decimal places first (matching Python)
        *c = (*c * 1e6).round() / 1e6;
        // Apply modulo
        *c = c.rem_euclid(mod_val);
    }
}

/// Sort a coordinate matrix lexicographically by (element_label, x, y, z)
pub(crate) fn sort_coords(coords: &[f64], atom_labels: &[usize], n_atoms: usize) -> Vec<f64> {
    // Create index array
    let mut indices: Vec<usize> = (0..n_atoms).collect();

    // Sort indices by (atom_label, x, y, z) lexicographically
    indices.sort_by(|&a, &b| {
        let a_offset = a * 3;
        let b_offset = b * 3;

        // Compare element label first
        match atom_labels[a].cmp(&atom_labels[b]) {
            Ordering::Equal => {
                // Compare x
                let cmp_x = coords[a_offset].partial_cmp(&coords[b_offset]).unwrap_or(Ordering::Equal);
                match cmp_x {
                    Ordering::Equal => {
                        // Compare y
                        let cmp_y = coords[a_offset + 1].partial_cmp(&coords[b_offset + 1]).unwrap_or(Ordering::Equal);
                        match cmp_y {
                            Ordering::Equal => {
                                // Compare z
                                coords[a_offset + 2].partial_cmp(&coords[b_offset + 2]).unwrap_or(Ordering::Equal)
                            }
                            other => other,
                        }
                    }
                    other => other,
                }
            }
            other => other,
        }
    });

    // Build sorted coordinate array
    let mut sorted = Vec::with_capacity(n_atoms * 3);
    for &idx in &indices {
        let offset = idx * 3;
        sorted.push(coords[offset]);
        sorted.push(coords[offset + 1]);
        sorted.push(coords[offset + 2]);
    }

    sorted
}

// =============================================================================
// Internal MNF Helpers
// =============================================================================

/// Apply all stabilizer matrices to motif coordinates
/// Returns a flat array of all stabilized coordinate matrices
fn apply_stabilizers(
    stabilizers: &[[[i32; 3]; 3]],
    coords: &[f64],
    n_atoms: usize,
    mod_val: f64,
) -> Vec<Vec<f64>> {
    let mut results = Vec::with_capacity(stabilizers.len());

    for stab in stabilizers {
        // Invert the stabilizer
        let inv = mat_inv_f64(stab);

        // Apply to coordinates
        let mut transformed = matrix_mult_coords(&inv, coords, n_atoms);

        // Move into bounds
        move_into_bounds(&mut transformed, mod_val);

        results.push(transformed);
    }

    results
}

/// Generate all origin shifts for a single coordinate matrix
/// Shifts the origin to each position of the first element type
fn generate_all_shifts(
    coords: &[f64],
    n_atoms: usize,
    num_origin_atoms: usize,
    mod_val: f64,
) -> Vec<Vec<f64>> {
    let mut shifted_mats = Vec::with_capacity(num_origin_atoms);

    // For each of the first num_origin_atoms positions
    for origin_idx in 0..num_origin_atoms {
        let offset = origin_idx * 3;

        // Shift vector is negative of this atom's position
        let shift = [
            -coords[offset],
            -coords[offset + 1],
            -coords[offset + 2],
        ];

        // Apply shift to all atoms
        let mut shifted = Vec::with_capacity(n_atoms * 3);
        for i in 0..n_atoms {
            let atom_offset = i * 3;
            shifted.push(coords[atom_offset] + shift[0]);
            shifted.push(coords[atom_offset + 1] + shift[1]);
            shifted.push(coords[atom_offset + 2] + shift[2]);
        }

        // Move into bounds
        move_into_bounds(&mut shifted, mod_val);
        shifted_mats.push(shifted);
    }

    shifted_mats
}

/// Extract MNF string from sorted coordinates (skip first atom, which is at origin)
#[inline]
fn extract_mnf(coords: &[f64], n_atoms: usize) -> Vec<f64> {
    coords[3..n_atoms * 3].to_vec()
}

/// Core MNF algorithm: compute canonical MNF for a single coordinate set
///
/// Takes pre-parsed stabilizers to allow sharing across batch calls.
fn compute_canonical_mnf(
    coords: &[f64],
    stabilizers: &[[[i32; 3]; 3]],
    n_atoms: usize,
    atom_labels: &[usize],
    num_origin_atoms: usize,
    mod_val: f64,
) -> Vec<f64> {
    // Step 1: Apply all stabilizers
    let stabilized_mats = apply_stabilizers(stabilizers, coords, n_atoms, mod_val);

    // Step 2: Generate all shifts for each stabilized motif
    let mut all_shifted = Vec::new();
    for stab_coords in stabilized_mats {
        let shifted = generate_all_shifts(&stab_coords, n_atoms, num_origin_atoms, mod_val);
        all_shifted.extend(shifted);
    }

    // Step 3: Sort each shifted motif
    let sorted_mats: Vec<Vec<f64>> = all_shifted
        .into_iter()
        .map(|c| sort_coords(&c, atom_labels, n_atoms))
        .collect();

    // Step 4: Extract MNF strings and find lexicographically smallest
    let mut mnf_strings: Vec<Vec<f64>> = sorted_mats
        .iter()
        .map(|c| extract_mnf(c, n_atoms))
        .collect();

    mnf_strings.sort_by(|a, b| {
        a.iter().zip(b.iter())
            .find_map(|(x, y)| match x.partial_cmp(y) {
                Some(Ordering::Equal) => None,
                ord => ord,
            })
            .unwrap_or(Ordering::Equal)
    });

    mnf_strings.swap_remove(0)
}

// =============================================================================
// Public API
// =============================================================================

/// Build MNF using vectorized algorithm
///
/// Returns the lexicographically smallest MNF coordinates
pub fn build_mnf_vectorized(
    coords: &[f64],
    n_atoms: usize,
    atom_labels: &[usize],
    num_origin_atoms: usize,
    stabilizers_flat: &[i32],
    mod_val: f64,
) -> Vec<f64> {
    if n_atoms == 1 {
        return vec![];
    }
    let stabilizers = parse_flat_to_matrices(stabilizers_flat);
    compute_canonical_mnf(coords, &stabilizers, n_atoms, atom_labels, num_origin_atoms, mod_val)
}

/// Build MNFs for many coordinate sets in batch
///
/// Parses stabilizers once and reuses across all coordinate sets.
pub fn build_mnf_batch(
    coords_batch: &[Vec<f64>],
    n_atoms: usize,
    atom_labels: &[usize],
    num_origin_atoms: usize,
    stabilizers_flat: &[i32],
    mod_val: f64,
) -> Vec<Vec<f64>> {
    if n_atoms == 1 {
        return vec![vec![]; coords_batch.len()];
    }
    let stabilizers = parse_flat_to_matrices(stabilizers_flat);
    coords_batch
        .iter()
        .map(|coords| compute_canonical_mnf(coords, &stabilizers, n_atoms, atom_labels, num_origin_atoms, mod_val))
        .collect()
}

/// Compute atom labels from atom symbols.
///
/// Consecutive identical atoms get the same label (0, 0, 0, 1, 1, 2, ...).
pub fn compute_atom_labels(atoms: &[String]) -> Vec<usize> {
    let mut labels = Vec::with_capacity(atoms.len());
    if atoms.is_empty() {
        return labels;
    }

    let mut current_label = 0;
    let mut prev_atom = &atoms[0];

    for atom in atoms {
        if atom != prev_atom {
            current_label += 1;
            prev_atom = atom;
        }
        labels.push(current_label);
    }

    labels
}
