/// Neighbor finding for CNF navigation
///
/// This module provides pure-Rust functions for finding lattice and motif neighbors.
/// These are extracted from the Python bindings in lib.rs to enable pathfinding.

use std::collections::HashSet;

/// Find all neighbor tuples (lattice + motif) for a given CNF state
///
/// Args:
///     vonorms_i32: Current CNF vonorms as integers (7 values)
///     coords_i32: Current CNF coords as integers (flattened, excludes origin)
///     elements: Atom element symbols
///     n_atoms: Number of atoms (including origin)
///     xi: Lattice step size
///     delta: Integer discretization factor
///
/// Returns:
///     Vec of (vonorms_tuple, coords_tuple) representing all unique neighbors
///
/// Note: Computes stabilizers internally from vonorms
pub(crate) fn find_neighbor_tuples(
    vonorms_i32: &[i32],
    coords_i32: &[i32],
    elements: &[String],
    n_atoms: usize,
    xi: f64,
    delta: i32,
) -> Vec<(Vec<i32>, Vec<i32>)> {
    // Compute stabilizers from vonorms
    let vonorms_f64: Vec<f64> = vonorms_i32.iter().map(|&v| v as f64).collect();
    let mut vonorms_arr = [0.0; 7];
    for (i, &v) in vonorms_f64.iter().enumerate().take(7) {
        vonorms_arr[i] = v;
    }
    let stabilizers_flat = crate::lnf::find_stabilizers_raw(&vonorms_arr);

    // Step 1: Find lattice neighbors
    let lattice_neighbors = find_lattice_neighbor_tuples_internal(
        vonorms_i32,
        coords_i32,
        &stabilizers_flat,
        elements,
        n_atoms,
        delta,
        xi,
    );

    // Step 2: Find motif neighbors
    let motif_neighbors = find_motif_neighbor_tuples_internal(
        vonorms_i32,
        coords_i32,
        &stabilizers_flat,
        elements,
        n_atoms,
        delta,
    );

    // Step 3: Combine all neighbors
    let mut all_neighbors = lattice_neighbors;
    all_neighbors.extend(motif_neighbors);

    // Step 4: Deduplicate by (vonorms, coords) tuple
    let unique_neighbors: HashSet<(Vec<i32>, Vec<i32>)> = all_neighbors.into_iter().collect();

    // Convert back to Vec and return
    unique_neighbors.into_iter().collect()
}

/// Internal function for finding lattice neighbors
/// Combines: step generation → validation → canonicalization
fn find_lattice_neighbor_tuples_internal(
    vonorms_i32: &[i32],
    coords_i32: &[i32],
    stabilizers_flat: &[i32],
    elements: &[String],
    n_atoms: usize,
    delta: i32,
    _xi: f64,
) -> Vec<(Vec<i32>, Vec<i32>)> {
    // Convert i32 vonorms to f64 for processing
    let mut vonorms_f64 = [0.0; 7];
    for (i, &v) in vonorms_i32.iter().enumerate().take(7) {
        vonorms_f64[i] = v as f64;
    }

    // Convert i32 coords to f64 and reshape to coordinate matrix (3, N) format
    // coords_i32 has format [x1, y1, z1, x2, y2, z2, ...] for (n_atoms - 1) atoms (excluding origin)
    // We need to add origin atom at [0, 0, 0]
    let n_stored_atoms = coords_i32.len() / 3;
    let mut coord_matrix = vec![0.0; 3 * n_atoms]; // (3, N) flattened

    // Set origin at index 0
    coord_matrix[0] = 0.0; // x[0]
    coord_matrix[n_atoms] = 0.0; // y[0]
    coord_matrix[2 * n_atoms] = 0.0; // z[0]

    // Copy remaining atoms
    for atom_idx in 0..n_stored_atoms {
        let x = coords_i32[atom_idx * 3] as f64;
        let y = coords_i32[atom_idx * 3 + 1] as f64;
        let z = coords_i32[atom_idx * 3 + 2] as f64;

        coord_matrix[atom_idx + 1] = x; // x coordinate
        coord_matrix[n_atoms + atom_idx + 1] = y; // y coordinate
        coord_matrix[2 * n_atoms + atom_idx + 1] = z; // z coordinate
    }

    // Step 1: Generate raw step data (stabilizers computed internally)
    let raw_steps = crate::compute_step_data_raw_internal(
        &vonorms_f64,
        &coord_matrix,
        n_atoms,
        delta,
    );

    // Step 2: Validate steps
    let validated_steps = crate::validate_step_data_internal(raw_steps);

    // Step 3: Canonicalize each validated step
    // Note: elements parameter is needed for atom labels - for now we'll need to pass it in
    // This is a TODO - the function signature needs elements
    let results = crate::canonicalize_cnfs_batch_internal(validated_steps, elements, delta);

    // Deduplicate
    let unique_neighbors: HashSet<(Vec<i32>, Vec<i32>)> = results.into_iter().collect();
    unique_neighbors.into_iter().collect()
}

/// Internal function for finding motif neighbors
/// Motif neighbors keep vonorms the same and only change coords
fn find_motif_neighbor_tuples_internal(
    vonorms_i32: &[i32],
    coords_i32: &[i32],
    stabilizers_flat: &[i32],
    elements: &[String],
    _n_atoms: usize,
    delta: i32,
) -> Vec<(Vec<i32>, Vec<i32>)> {
    // Call the existing mnf function which returns canonical coords
    let canonical_coords_list = crate::mnf::find_and_canonicalize_motif_neighbors(
        coords_i32,
        elements,
        stabilizers_flat,
        delta,
    );

    // Pair each result with the unchanged vonorms
    let vonorms_vec = vonorms_i32.to_vec();
    canonical_coords_list
        .into_iter()
        .map(|coords| (vonorms_vec.clone(), coords))
        .collect()
}
