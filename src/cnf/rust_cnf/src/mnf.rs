/// MNF (Motif Normal Form) construction in Rust
///
/// This module implements the vectorized MNF construction algorithm,
/// which finds the lexicographically smallest representation of an atomic motif.

use std::cmp::Ordering;

use crate::linalg::mat_inv_f64;

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

/// Sort coordinates by (atom_label, x, y, z) lexicographically
/// coords: flat array [x1, y1, z1, x2, y2, z2, ...]
/// Returns sorted coordinates in same format
fn sort_coords_by_labels(coords: &[f64], atom_labels: &[usize], n_atoms: usize) -> Vec<f64> {
    // Create vector of (label, x, y, z, original_index)
    let mut indexed_coords: Vec<(usize, f64, f64, f64, usize)> = Vec::with_capacity(n_atoms);

    for i in 0..n_atoms {
        let offset = i * 3;
        indexed_coords.push((
            atom_labels[i],
            coords[offset],
            coords[offset + 1],
            coords[offset + 2],
            i,
        ));
    }

    // Sort by (label, x, y, z) - lexicographic ordering
    indexed_coords.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .then(a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal))
            .then(a.3.partial_cmp(&b.3).unwrap_or(Ordering::Equal))
    });

    // Extract sorted coordinates
    let mut sorted = Vec::with_capacity(coords.len());
    for (_, x, y, z, _) in indexed_coords {
        sorted.push(x);
        sorted.push(y);
        sorted.push(z);
    }

    sorted
}

/// Generate motif perturbations
/// coords_with_origin: flat array WITH origin at [0,0,0], so length = n_atoms * 3
/// Returns list of perturbed coordinate arrays (each as Vec<i32>)
fn generate_motif_perturbations(coords_with_origin: &[i32], n_atoms: usize) -> Vec<Vec<i32>> {
    let mut perturbations = Vec::new();
    let coord_len = n_atoms * 3;

    // Generate ±1 for each individual coordinate
    for idx in 0..coord_len {
        for adj in [-1, 1] {
            let mut perturbed = coords_with_origin.to_vec();
            perturbed[idx] += adj;
            perturbations.push(perturbed);
        }
    }

    // Generate ±1 for each atom (all 3 coordinates)
    for atom_idx in 0..n_atoms {
        for adj in [-1, 1] {
            let mut perturbed = coords_with_origin.to_vec();
            let offset = atom_idx * 3;
            perturbed[offset] += adj;
            perturbed[offset + 1] += adj;
            perturbed[offset + 2] += adj;
            perturbations.push(perturbed);
        }
    }

    perturbations
}

/// Find and canonicalize motif neighbors
///
/// This function finds all motif neighbors by:
/// 1. Applying each stabilizer to the motif
/// 2. Generating perturbations (±1 on coordinates)
/// 3. Batch canonicalizing all perturbations
/// 4. Returning unique canonical results
///
/// Arguments:
/// - motif_coords: Coordinates WITHOUT origin, length = (n_atoms-1)*3
/// - atoms: Atom symbols INCLUDING origin, length = n_atoms
/// - stabilizers_flat: Flattened 3x3 stabilizer matrices
/// - delta: Discretization parameter
///
/// Returns: List of canonical motif coordinates (WITHOUT origin)
pub fn find_and_canonicalize_motif_neighbors(
    motif_coords: &[i32],
    atoms: &[String],
    stabilizers_flat: &[i32],
    delta: i32,
) -> Vec<Vec<i32>> {
    let n_atoms = atoms.len();

    // Pre-compute atom labels
    let atom_labels = compute_atom_labels(atoms);

    // Count origin atoms (atoms matching the first atom)
    let num_origin_atoms = atoms.iter().filter(|a| *a == &atoms[0]).count();

    // Reshape stabilizers
    let n_stabilizers = stabilizers_flat.len() / 9;
    let mut stabilizers: Vec<[[i32; 3]; 3]> = Vec::with_capacity(n_stabilizers);
    for i in 0..n_stabilizers {
        let offset = i * 9;
        let mat = [
            [stabilizers_flat[offset], stabilizers_flat[offset + 1], stabilizers_flat[offset + 2]],
            [stabilizers_flat[offset + 3], stabilizers_flat[offset + 4], stabilizers_flat[offset + 5]],
            [stabilizers_flat[offset + 6], stabilizers_flat[offset + 7], stabilizers_flat[offset + 8]],
        ];
        stabilizers.push(mat);
    }

    // Convert motif_coords to f64 and prepend origin
    let mut coords_f64 = vec![0.0, 0.0, 0.0]; // Origin at [0, 0, 0]
    for &c in motif_coords {
        coords_f64.push(c as f64);
    }

    // Collect all perturbations across all stabilizers
    let mut all_perturbations: Vec<Vec<f64>> = Vec::new();

    for stabilizer in &stabilizers {
        // Apply stabilizer transformation
        let inv = mat_inv_f64(stabilizer);
        let mut transformed = matrix_mult_coords(&inv, &coords_f64, n_atoms);
        move_into_bounds(&mut transformed, delta as f64);

        // Sort by (atom_label, x, y, z)
        let sorted = sort_coords_by_labels(&transformed, &atom_labels, n_atoms);

        // Extract coords excluding origin and convert to i32
        let sorted_mnf: Vec<i32> = sorted[3..]  // Skip first 3 (origin)
            .iter()
            .map(|&c| c.round() as i32)
            .collect();

        // Prepend origin for perturbation generation
        let mut coords_with_origin = vec![0, 0, 0];
        coords_with_origin.extend_from_slice(&sorted_mnf);

        // Generate perturbations
        let perturbations = generate_motif_perturbations(&coords_with_origin, n_atoms);

        // Convert perturbations to f64 for canonicalization
        for perturb in perturbations {
            let perturb_f64: Vec<f64> = perturb.iter().map(|&c| c as f64).collect();
            all_perturbations.push(perturb_f64);
        }
    }

    // Flatten stabilizers for build_mnf_batch
    let mut stabilizers_flat = Vec::with_capacity(stabilizers.len() * 9);
    for stab in &stabilizers {
        for row in stab {
            for &val in row {
                stabilizers_flat.push(val);
            }
        }
    }

    // Batch canonicalize all perturbations
    let canonical_results = build_mnf_batch(
        &all_perturbations,
        n_atoms,
        &atom_labels,
        num_origin_atoms,
        &stabilizers_flat,
        delta as f64,
    );

    // Convert results to i32 and deduplicate
    use std::collections::HashSet;
    let mut unique_results: HashSet<Vec<i32>> = HashSet::new();

    for result_f64 in canonical_results {
        let result_i32: Vec<i32> = result_f64.iter().map(|&c| c.round() as i32).collect();
        unique_results.insert(result_i32);
    }

    // Convert to Vec and return
    unique_results.into_iter().collect()
}
