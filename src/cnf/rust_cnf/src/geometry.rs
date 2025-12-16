/// Geometry utilities for crystal structure reconstruction and distance calculations

use crate::permutations::compute_conorms;

/// Convert vonorms to 3x3 lattice matrix (row vectors)
///
/// This implements the algorithm from vonorm_list.py:to_generators()
///
/// Args:
///     vonorms: 7 vonorm values
///     xi: lattice step size (scaling factor)
///
/// Returns:
///     3x3 matrix where each row is a lattice vector
pub fn vonorms_to_lattice_matrix(vonorms: &[f64; 7], xi: f64) -> [[f64; 3]; 3] {
    // Scale vonorms and conorms by xi
    let physical_vonorms: Vec<f64> = vonorms.iter().map(|&v| v * xi).collect();
    let raw_conorms = compute_conorms(vonorms);
    let physical_conorms: Vec<f64> = raw_conorms.iter().map(|&c| c * xi).collect();

    // Validate: no zero or negative primary vonorms
    for i in 0..3 {
        if physical_vonorms[i] <= 0.0 {
            panic!("Invalid vonorms: primary vonorms must be positive");
        }
    }

    // Extract specific conorm values (dot products)
    let v0_dot_v1 = physical_conorms[0];
    let v0_dot_v2 = physical_conorms[1];
    let v1_dot_v2 = physical_conorms[3];

    // Compute vector norms from vonorms (squared norms)
    let v0_norm = physical_vonorms[0].sqrt();
    let v1_norm = physical_vonorms[1].sqrt();
    let v2_norm = physical_vonorms[2].sqrt();

    // Compute components using Gram-Schmidt-like geometry
    let cos_x = v0_dot_v1 / (v0_norm * v1_norm);
    let x = cos_x.clamp(-1.0, 1.0); // Clip to valid cosine range
    let y = (1.0 - x * x).max(0.0).sqrt();

    let cos_a = v0_dot_v2 / (v0_norm * v2_norm);
    let a = cos_a.clamp(-1.0, 1.0);

    let b = if y > 1e-9 {
        let term_for_b = v1_dot_v2 / (v1_norm * v2_norm);
        (1.0 / y) * (term_for_b - x * a)
    } else {
        0.0
    };

    let c_squared = 1.0 - a * a - b * b;
    let c = c_squared.max(0.0).sqrt();

    // Build lattice vectors in standard orientation
    let v0 = [v0_norm, 0.0, 0.0];
    let v1 = [v1_norm * x, v1_norm * y, 0.0];
    let v2 = [v2_norm * a, v2_norm * b, v2_norm * c];

    [v0, v1, v2]
}

/// Convert MNF integer coordinates to cartesian positions
///
/// Args:
///     coords: Flattened MNF coordinate list (excludes origin atom)
///     n_atoms: Total number of atoms (including origin)
///     delta: Integer discretization factor
///     lattice_matrix: 3x3 lattice matrix (row vectors)
///
/// Returns:
///     Flattened array of cartesian positions (n_atoms * 3 values)
pub fn coords_to_cartesian_positions(
    coords: &[i32],
    n_atoms: usize,
    delta: i32,
    lattice_matrix: &[[f64; 3]; 3],
) -> Vec<f64> {
    let mut cartesian_positions = Vec::with_capacity(n_atoms * 3);

    // First atom is at origin
    cartesian_positions.extend_from_slice(&[0.0, 0.0, 0.0]);

    // Convert remaining atoms
    for i in 0..(n_atoms - 1) {
        let offset = i * 3;

        // Convert to fractional coordinates
        let frac_x = coords[offset] as f64 / delta as f64;
        let frac_y = coords[offset + 1] as f64 / delta as f64;
        let frac_z = coords[offset + 2] as f64 / delta as f64;

        // Convert to cartesian: lattice_matrix^T @ frac_coords
        // (lattice vectors are rows, so we do column @ rows^T)
        let cart_x = frac_x * lattice_matrix[0][0] + frac_y * lattice_matrix[1][0] + frac_z * lattice_matrix[2][0];
        let cart_y = frac_x * lattice_matrix[0][1] + frac_y * lattice_matrix[1][1] + frac_z * lattice_matrix[2][1];
        let cart_z = frac_x * lattice_matrix[0][2] + frac_y * lattice_matrix[1][2] + frac_z * lattice_matrix[2][2];

        cartesian_positions.push(cart_x);
        cartesian_positions.push(cart_y);
        cartesian_positions.push(cart_z);
    }

    cartesian_positions
}

/// Compute pairwise distances between atoms with periodic boundary conditions
///
/// Uses minimum image convention to handle PBC.
///
/// Args:
///     positions: Flattened cartesian positions (n_atoms * 3 values)
///     n_atoms: Number of atoms
///     lattice_matrix: 3x3 lattice matrix (row vectors)
///     inv_lattice_opt: Optional pre-computed inverse lattice matrix.
///                      If None, will be computed from lattice_matrix.
///
/// Returns:
///     Flattened upper-triangular distance matrix (n_choose_2 values)
/// Compute pairwise distances with periodic boundary conditions
///
/// For each pair of atoms, checks all 27 periodic images to find the minimum distance.
/// The 27 images come from: 3 directions (x,y,z) × 3 offsets per direction (-1,0,+1) = 3³ = 27
pub fn compute_pairwise_distances_pbc(
    positions: &[f64],
    n_atoms: usize,
    lattice_matrix: &[[f64; 3]; 3],
) -> Vec<f64> {
    let n_pairs = (n_atoms * (n_atoms - 1)) / 2;
    let mut distances = Vec::with_capacity(n_pairs);

    for i in 0..n_atoms {
        for j in (i + 1)..n_atoms {
            let mut min_dist_sq = f64::MAX;

            // Check all 27 periodic images (-1, 0, +1 in each direction)
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        // Compute offset from lattice vectors (stored as rows)
                        let offset_x = dx as f64 * lattice_matrix[0][0] +
                                      dy as f64 * lattice_matrix[1][0] +
                                      dz as f64 * lattice_matrix[2][0];
                        let offset_y = dx as f64 * lattice_matrix[0][1] +
                                      dy as f64 * lattice_matrix[1][1] +
                                      dz as f64 * lattice_matrix[2][1];
                        let offset_z = dx as f64 * lattice_matrix[0][2] +
                                      dy as f64 * lattice_matrix[1][2] +
                                      dz as f64 * lattice_matrix[2][2];

                        // Compute distance with this periodic image
                        let diff_x = positions[j * 3] + offset_x - positions[i * 3];
                        let diff_y = positions[j * 3 + 1] + offset_y - positions[i * 3 + 1];
                        let diff_z = positions[j * 3 + 2] + offset_z - positions[i * 3 + 2];

                        let dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
                        min_dist_sq = min_dist_sq.min(dist_sq);
                    }
                }
            }

            distances.push(min_dist_sq.sqrt());
        }
    }

    distances
}

/// Filter neighbors by minimum pairwise distance
///
/// For each neighbor:
/// 1. Reconstruct lattice from vonorms
/// 2. Reconstruct atom positions from coords
/// 3. Compute pairwise distances with PBC (checking all 27 periodic images)
/// 4. Check if all distances exceed threshold
///
/// Args:
///     neighbor_tuples: Vec of (vonorms_tuple, coords_tuple)
///     n_atoms: Number of atoms (including origin)
///     xi: Lattice step size
///     delta: Integer discretization factor
///     min_distance: Minimum allowed pairwise distance (Angstroms)
///
/// Returns:
///     Filtered list of neighbor tuples
pub fn filter_neighbors_by_min_distance(
    neighbor_tuples: &[(Vec<i32>, Vec<i32>)],
    n_atoms: usize,
    xi: f64,
    delta: i32,
    min_distance: f64,
) -> Vec<(Vec<i32>, Vec<i32>)> {
    let mut filtered = Vec::new();

    for (vonorms_tuple, coords_tuple) in neighbor_tuples {
        // Convert vonorms to f64 array
        let mut vonorms = [0.0; 7];
        for (i, &v) in vonorms_tuple.iter().take(7).enumerate() {
            vonorms[i] = v as f64;
        }

        // Reconstruct lattice
        let lattice_matrix = vonorms_to_lattice_matrix(&vonorms, xi);

        // Reconstruct positions
        let positions = coords_to_cartesian_positions(coords_tuple, n_atoms, delta, &lattice_matrix);

        // Compute pairwise distances with PBC
        let distances = compute_pairwise_distances_pbc(&positions, n_atoms, &lattice_matrix);

        // Check if all distances exceed threshold
        let passes_filter = distances.iter().all(|&d| d > min_distance);

        if passes_filter {
            filtered.push((vonorms_tuple.clone(), coords_tuple.clone()));
        }
    }

    filtered
}
