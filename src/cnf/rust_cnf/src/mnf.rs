/// MNF (Motif Normal Form) construction in Rust
///
/// This module implements the vectorized MNF construction algorithm,
/// which finds the lexicographically smallest representation of an atomic motif.

use std::cmp::Ordering;

/// Invert a 3x3 matrix (for i32, returns f64)
fn invert_matrix_3x3(mat: &[[i32; 3]; 3]) -> [[f64; 3]; 3] {
    let m = mat;

    // Convert to f64 for inversion
    let a = [
        [m[0][0] as f64, m[0][1] as f64, m[0][2] as f64],
        [m[1][0] as f64, m[1][1] as f64, m[1][2] as f64],
        [m[2][0] as f64, m[2][1] as f64, m[2][2] as f64],
    ];

    // Calculate determinant
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

    if det.abs() < 1e-10 {
        panic!("Matrix is singular, cannot invert");
    }

    let inv_det = 1.0 / det;

    // Calculate inverse using cofactor method
    let mut inv = [[0.0; 3]; 3];

    inv[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv_det;
    inv[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv_det;
    inv[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_det;

    inv[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * inv_det;
    inv[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_det;
    inv[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv_det;

    inv[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv_det;
    inv[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv_det;
    inv[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_det;

    inv
}

/// Multiply a 3x3 matrix by a 3xN coordinate matrix
fn matrix_mult_coords(mat: &[[f64; 3]; 3], coords: &[f64], n_atoms: usize) -> Vec<f64> {
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
fn move_into_bounds(coords: &mut [f64], mod_val: f64) {
    for c in coords.iter_mut() {
        // Round to 6 decimal places first (matching Python)
        *c = (*c * 1e6).round() / 1e6;
        // Apply modulo
        *c = c.rem_euclid(mod_val);
    }
}

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
        let inv = invert_matrix_3x3(stab);

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

/// Sort a coordinate matrix lexicographically by (element_label, x, y, z)
fn sort_coords(coords: &[f64], atom_labels: &[usize], n_atoms: usize) -> Vec<f64> {
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

/// Extract MNF string from sorted coordinates (skip first atom, which is at origin)
fn extract_mnf(coords: &[f64], n_atoms: usize) -> Vec<f64> {
    // Skip first 3 elements (the origin atom at 0,0,0)
    coords[3..n_atoms * 3].to_vec()
}

/// Build MNF using vectorized algorithm
///
/// Returns the lexicographically smallest MNF coordinates
pub fn build_mnf_vectorized(
    coords: &[f64],           // Flat array of 3*n_atoms coordinates (x,y,z for each atom)
    n_atoms: usize,
    atom_labels: &[usize],    // Element label for each atom (0, 1, 2, etc.)
    num_origin_atoms: usize,  // Number of atoms with the first element type
    stabilizers_flat: &[i32], // Flat array of stabilizer matrices
    mod_val: f64,             // Modulus (1.0 for fractional, delta for discretized)
) -> Vec<f64> {
    // Handle single atom case
    if n_atoms == 1 {
        return vec![];
    }

    // Parse stabilizers
    let n_stabilizers = stabilizers_flat.len() / 9;
    let mut stabilizers = Vec::with_capacity(n_stabilizers);
    for i in 0..n_stabilizers {
        let offset = i * 9;
        let mat = [
            [stabilizers_flat[offset], stabilizers_flat[offset + 1], stabilizers_flat[offset + 2]],
            [stabilizers_flat[offset + 3], stabilizers_flat[offset + 4], stabilizers_flat[offset + 5]],
            [stabilizers_flat[offset + 6], stabilizers_flat[offset + 7], stabilizers_flat[offset + 8]],
        ];
        stabilizers.push(mat);
    }

    // Step 1: Apply all stabilizers
    let stabilized_mats = apply_stabilizers(&stabilizers, coords, n_atoms, mod_val);

    // Step 2: Generate all shifts for each stabilized motif
    let mut all_shifted = Vec::new();
    for stab_coords in stabilized_mats {
        let shifted = generate_all_shifts(&stab_coords, n_atoms, num_origin_atoms, mod_val);
        all_shifted.extend(shifted);
    }

    // Step 3: Sort each shifted motif
    let mut sorted_mats = Vec::with_capacity(all_shifted.len());
    for coords in all_shifted {
        let sorted = sort_coords(&coords, atom_labels, n_atoms);
        sorted_mats.push(sorted);
    }

    // Step 4: Extract MNF strings
    let mut mnf_strings: Vec<Vec<f64>> = sorted_mats
        .iter()
        .map(|coords| extract_mnf(coords, n_atoms))
        .collect();

    // Step 5: Find lexicographically smallest
    mnf_strings.sort_by(|a, b| {
        for (x, y) in a.iter().zip(b.iter()) {
            match x.partial_cmp(y) {
                Some(Ordering::Equal) => continue,
                Some(ord) => return ord,
                None => continue,
            }
        }
        Ordering::Equal
    });

    mnf_strings[0].clone()
}

/// Build MNFs for many coordinate sets in batch
///
/// This processes all coordinate sets in one call, sharing stabilizer parsing and
/// avoiding Python/Rust boundary overhead.
///
/// Returns a vector of MNF coordinate vectors (one per input coordinate set)
pub fn build_mnf_batch(
    coords_batch: &[Vec<f64>],    // Vector of coordinate sets, each [x1,y1,z1,x2,y2,z2,...]
    n_atoms: usize,
    atom_labels: &[usize],        // Element label for each atom (shared across all)
    num_origin_atoms: usize,      // Number of origin atoms (shared across all)
    stabilizers_flat: &[i32],     // Flat array of stabilizer matrices (shared across all)
    mod_val: f64,                 // Modulus (shared across all)
) -> Vec<Vec<f64>> {
    // Handle single atom case
    if n_atoms == 1 {
        return vec![vec![]; coords_batch.len()];
    }

    // Parse stabilizers once for all coordinate sets
    let n_stabilizers = stabilizers_flat.len() / 9;
    let mut stabilizers = Vec::with_capacity(n_stabilizers);
    for i in 0..n_stabilizers {
        let offset = i * 9;
        let mat = [
            [stabilizers_flat[offset], stabilizers_flat[offset + 1], stabilizers_flat[offset + 2]],
            [stabilizers_flat[offset + 3], stabilizers_flat[offset + 4], stabilizers_flat[offset + 5]],
            [stabilizers_flat[offset + 6], stabilizers_flat[offset + 7], stabilizers_flat[offset + 8]],
        ];
        stabilizers.push(mat);
    }

    // Process each coordinate set
    let mut results = Vec::with_capacity(coords_batch.len());

    for coords in coords_batch {
        // Step 1: Apply all stabilizers
        let stabilized_mats = apply_stabilizers(&stabilizers, coords, n_atoms, mod_val);

        // Step 2: Generate all shifts for each stabilized motif
        let mut all_shifted = Vec::new();
        for stab_coords in stabilized_mats {
            let shifted = generate_all_shifts(&stab_coords, n_atoms, num_origin_atoms, mod_val);
            all_shifted.extend(shifted);
        }

        // Step 3: Sort each shifted motif
        let mut sorted_mats = Vec::with_capacity(all_shifted.len());
        for coord_set in all_shifted {
            let sorted = sort_coords(&coord_set, atom_labels, n_atoms);
            sorted_mats.push(sorted);
        }

        // Step 4: Extract MNF strings
        let mut mnf_strings: Vec<Vec<f64>> = sorted_mats
            .iter()
            .map(|coords| extract_mnf(coords, n_atoms))
            .collect();

        // Step 5: Find lexicographically smallest
        mnf_strings.sort_by(|a, b| {
            for (x, y) in a.iter().zip(b.iter()) {
                match x.partial_cmp(y) {
                    Some(Ordering::Equal) => continue,
                    Some(ord) => return ord,
                    None => continue,
                }
            }
            Ordering::Equal
        });

        results.push(mnf_strings[0].clone());
    }

    results
}
