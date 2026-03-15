/// Neighbor finding for CNF navigation
///
/// This module provides pure-Rust functions for finding lattice and motif neighbors.
/// It includes both the public API and the internal helper functions used by Python bindings.

use std::collections::{HashSet, HashMap};

use crate::linalg::{mat_mul, mat_inv_f64, flat9_to_mat3x3, slice_to_vonorms, parse_flat_to_matrices, IDENTITY_3X3};
use crate::permutations::{compute_conorms, find_zero_indices_exact, get_s4_representatives};
use crate::mnf::compute_atom_labels;
use crate::lnf;
use crate::mnf;

// =============================================================================
// Step Vector Generation
// =============================================================================

/// Generate step vectors (equivalent to LatticeStep.all_step_vecs())
fn generate_step_vectors() -> Vec<Vec<i32>> {
    let mut steps = Vec::new();

    for first_idx in 0..7 {
        for second_idx in (first_idx + 1)..7 {
            let mut vec = vec![0i32; 7];
            vec[first_idx] = 1;

            let is_primary_first = first_idx < 4;
            let is_primary_second = second_idx < 4;

            if is_primary_first && is_primary_second {
                vec[second_idx] = -1;
            } else if is_primary_first && !is_primary_second {
                vec[second_idx] = 1;
            } else {  // first is secondary
                vec[second_idx] = -1;
            }

            steps.push(vec.clone());

            // Add negative version
            let neg_vec: Vec<i32> = vec.iter().map(|&x| -x).collect();
            steps.push(neg_vec);
        }
    }

    steps
}

// =============================================================================
// Step Data Computation (used by Python bindings)
// =============================================================================

/// Compute raw step data for lattice neighbor finding
/// Note: Computes stabilizers internally from vonorms
pub(crate) fn compute_step_data_raw_internal(
    input_vonorms: &[f64; 7],
    motif_coord_matrix: &[f64],  // Flat (3*N) array
    n_atoms: usize,
    motif_delta: i32,
) -> Vec<(Vec<i32>, Vec<f64>, Vec<f64>, Vec<i32>)> {
    // Compute conorms and zero indices
    let conorms = compute_conorms(input_vonorms);
    let zero_indices = find_zero_indices_exact(&conorms);

    // Get S4 representatives using precomputed data
    let s4_reps = get_s4_representatives(input_vonorms, &zero_indices);

    // Compute current stabilizers from input vonorms
    let s1_flat = lnf::find_stabilizers_raw(input_vonorms);
    let s1_matrices = parse_flat_to_matrices(&s1_flat);

    // Generate step vectors (same as LatticeStep.all_step_vecs())
    let step_vecs = generate_step_vectors();

    let mut all_step_data: Vec<(Vec<i32>, Vec<f64>, Vec<f64>, Vec<i32>)> = Vec::new();

    // Process each S4 representative
    for s4_rep in s4_reps {
        let permuted_vonorms = s4_rep.permuted_vonorms.to_vec();
        let transform_mats = s4_rep.transition_mats;

        // Compute stabilizers in Rust
        let s2_flat = lnf::find_stabilizers_raw(&s4_rep.permuted_vonorms);

        // Parse transform and s2 matrices
        let t_matrices: Vec<[[i32; 3]; 3]> = transform_mats.iter()
            .map(|mat_vec| [
                [mat_vec[0][0], mat_vec[0][1], mat_vec[0][2]],
                [mat_vec[1][0], mat_vec[1][1], mat_vec[1][2]],
                [mat_vec[2][0], mat_vec[2][1], mat_vec[2][2]],
            ])
            .collect();
        let s2_matrices = parse_flat_to_matrices(&s2_flat);

        // Compute all s1 @ t @ s2 products and deduplicate
        let mut unique_products: HashMap<Vec<i32>, [[i32; 3]; 3]> = HashMap::new();

        for s1 in &s1_matrices {
            for t in &t_matrices[..1.min(t_matrices.len())] {
                for s2 in &s2_matrices {
                    let st = mat_mul(s1, t);
                    let product = mat_mul(&st, s2);
                    unique_products.insert(crate::linalg::mat_to_flat(&product).to_vec(), product);
                }
            }
        }

        // For each unique matrix, transform motif and generate steps
        let mut sorted_products: Vec<(Vec<i32>, [[i32; 3]; 3])> = unique_products.into_iter().collect();
        sorted_products.sort_by(|a, b| a.0.cmp(&b.0));

        for (mat_flat, mat) in sorted_products {
            // Invert matrix and transform coordinates
            let mat_inv = mat_inv_f64(&mat);

            // Transform coordinates: mat_inv @ coord_matrix
            let mut transformed_coords = vec![0.0; motif_coord_matrix.len()];
            for atom_idx in 0..n_atoms {
                for row in 0..3 {
                    let mut sum = 0.0;
                    for col in 0..3 {
                        sum += mat_inv[row][col] as f64 * motif_coord_matrix[col * n_atoms + atom_idx];
                    }
                    let val = sum.rem_euclid(motif_delta as f64);
                    transformed_coords[row * n_atoms + atom_idx] = val;
                }
            }

            // Transpose to (N, 3) format
            let mut coords_transposed = vec![0.0; n_atoms * 3];
            for atom_idx in 0..n_atoms {
                for dim in 0..3 {
                    coords_transposed[atom_idx * 3 + dim] = transformed_coords[dim * n_atoms + atom_idx];
                }
            }

            // For each step vector, compute new vonorms
            for step_vec in &step_vecs {
                let mut new_vonorms = Vec::with_capacity(7);
                for i in 0..7 {
                    new_vonorms.push(permuted_vonorms[i] + step_vec[i] as f64);
                }

                all_step_data.push((
                    step_vec.clone(),
                    new_vonorms,
                    coords_transposed.clone(),
                    mat_flat.clone(),
                ));
            }
        }
    }

    // Deduplicate by (vonorms, coords)
    let mut unique_steps: HashMap<(Vec<i32>, Vec<i32>), (Vec<i32>, Vec<f64>, Vec<f64>, Vec<i32>)> = HashMap::new();

    for (step_vec, vonorms, coords, mat) in all_step_data {
        let vonorms_key: Vec<i32> = vonorms.iter().map(|&v| v.round() as i32).collect();
        let coords_key: Vec<i32> = coords.iter().map(|&v| v.round() as i32).collect();

        let key = (vonorms_key, coords_key);
        unique_steps.entry(key).or_insert((step_vec, vonorms, coords, mat));
    }

    // Sort by key for deterministic ordering
    let mut result: Vec<_> = unique_steps.into_iter().collect();
    result.sort_by(|a, b| a.0.cmp(&b.0));

    result.into_iter().map(|(_, v)| v).collect()
}

// =============================================================================
// Step Data Validation
// =============================================================================

/// Validate step data by filtering steps that pass vonorm validation
pub(crate) fn validate_step_data_internal(
    step_data: Vec<(Vec<i32>, Vec<f64>, Vec<f64>, Vec<i32>)>
) -> Vec<(Vec<i32>, Vec<f64>, Vec<f64>, Vec<i32>)> {
    let mut validated_steps = Vec::new();

    for (step_vec, vonorms, coords, matrix) in step_data {
        let mut step_vonorms = [0.0; 7];
        for (i, &v) in vonorms.iter().enumerate().take(7) {
            step_vonorms[i] = v;
        }

        // Validate
        if !lnf::has_valid_conorms_exact(&step_vonorms) {
            continue;
        }
        if !lnf::is_obtuse(&step_vonorms) || !lnf::is_superbasis_exact(&step_vonorms) {
            continue;
        }

        validated_steps.push((step_vec, vonorms, coords, matrix));
    }

    validated_steps
}

// =============================================================================
// CNF Canonicalization
// =============================================================================

/// Canonicalize a batch of CNF step data
/// Returns Vec of (canonical_vonorms_i32, canonical_coords_i32) tuples
pub(crate) fn canonicalize_cnfs_batch_internal(
    step_data: Vec<(Vec<i32>, Vec<f64>, Vec<f64>, Vec<i32>)>,
    atoms: &[String],
    delta: i32,
) -> Vec<(Vec<i32>, Vec<i32>)> {
    let atom_labels = compute_atom_labels(atoms);
    let n_atoms = atoms.len();
    let num_origin_atoms = atom_labels.iter().filter(|&&label| label == 0).count();

    let mut results = Vec::new();

    for (_step_vec, vonorms, coords, _matrix) in step_data {
        // Step 1: Build LNF (canonicalize vonorms)
        let vonorms_arr = slice_to_vonorms(&vonorms);
        let (canonical_vonorms_vec, _, selling_flat_opt, sorting_mats) =
            lnf::build_lnf_raw_discretized(&vonorms_arr);
        let canonical_vonorms = slice_to_vonorms(&canonical_vonorms_vec);

        // Step 2: Compute transformation matrix (middle = selling @ sorting)
        let selling_mat = selling_flat_opt
            .as_ref()
            .map(|flat| flat9_to_mat3x3(flat))
            .unwrap_or(IDENTITY_3X3);
        let sorting_mat = flat9_to_mat3x3(&sorting_mats[0]);
        let middle = mat_mul(&selling_mat, &sorting_mat);

        // Step 3: Find stabilizers and combine
        let s2_flat = lnf::find_stabilizers_raw(&canonical_vonorms);
        let combined_stabs_flat = lnf::combine_middle_and_s2_stabilizers(&s2_flat, &middle);

        // Step 4: Build canonical MNF
        let canonical_coords = mnf::build_mnf_vectorized(
            &coords,
            n_atoms,
            &atom_labels,
            num_origin_atoms,
            &combined_stabs_flat,
            delta as f64,
        );

        // Convert to integers
        let vonorms_ints: Vec<i32> = canonical_vonorms[..7].iter().map(|&v| v.round() as i32).collect();
        let coords_ints: Vec<i32> = canonical_coords.iter().map(|&c| c.round() as i32).collect();

        results.push((vonorms_ints, coords_ints));
    }

    results
}

// =============================================================================
// Public API: Find Neighbor Tuples
// =============================================================================

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
        delta,
    );

    // Step 3: Combine all neighbors
    let mut all_neighbors = lattice_neighbors;
    all_neighbors.extend(motif_neighbors);

    // Step 4: Deduplicate by (vonorms, coords) tuple
    let unique_neighbors: HashSet<(Vec<i32>, Vec<i32>)> = all_neighbors.into_iter().collect();

    // Step 5: Filter self-loops
    let original = (vonorms_i32.to_vec(), coords_i32.to_vec());
    unique_neighbors.into_iter().filter(|n| *n != original).collect()
}

/// Internal function for finding lattice neighbors
fn find_lattice_neighbor_tuples_internal(
    vonorms_i32: &[i32],
    coords_i32: &[i32],
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

        coord_matrix[atom_idx + 1] = x;
        coord_matrix[n_atoms + atom_idx + 1] = y;
        coord_matrix[2 * n_atoms + atom_idx + 1] = z;
    }

    // Step 1: Generate raw step data (stabilizers computed internally)
    let raw_steps = compute_step_data_raw_internal(
        &vonorms_f64,
        &coord_matrix,
        n_atoms,
        delta,
    );

    // Step 2: Validate steps
    let validated_steps = validate_step_data_internal(raw_steps);

    // Step 3: Canonicalize each validated step
    let results = canonicalize_cnfs_batch_internal(validated_steps, elements, delta);

    // Deduplicate
    let unique_neighbors: HashSet<(Vec<i32>, Vec<i32>)> = results.into_iter().collect();
    unique_neighbors.into_iter().collect()
}

// =============================================================================
// Motif Neighbor Generation
// =============================================================================

/// Generate all possible ±1 perturbations of motif coordinates
///
/// For each atom, generates:
/// - 6 single-axis perturbations: ±1 in x, y, or z
/// - 2 diagonal perturbations: +(1,1,1) and -(1,1,1)
///
/// Args:
///     coords_with_origin: Flat array of coordinates including origin atom (3*n_atoms values)
///     n_atoms: Total number of atoms including origin
///
/// Returns:
///     Vec of perturbed coordinate arrays (each 3*n_atoms values)
fn generate_motif_perturbations(coords_with_origin: &[i32], n_atoms: usize) -> Vec<Vec<i32>> {
    let mut perturbations = Vec::new();

    // For each atom (skip origin at index 0)
    for atom_idx in 1..n_atoms {
        let base_offset = atom_idx * 3;

        // Single-axis perturbations: ±1 in x, y, z
        for dim in 0..3 {
            for delta in [-1i32, 1i32] {
                let mut perturbed = coords_with_origin.to_vec();
                perturbed[base_offset + dim] += delta;
                perturbations.push(perturbed);
            }
        }

        // Diagonal perturbations: ±(1,1,1)
        for delta in [-1i32, 1i32] {
            let mut perturbed = coords_with_origin.to_vec();
            perturbed[base_offset] += delta;
            perturbed[base_offset + 1] += delta;
            perturbed[base_offset + 2] += delta;
            perturbations.push(perturbed);
        }
    }

    perturbations
}

/// Find and canonicalize all motif neighbors
///
/// Generates all ±1 perturbations, applies modulo, and canonicalizes each using MNF algorithm.
///
/// Args:
///     motif_coords: Flat array of coordinates WITHOUT origin (3*(n_atoms-1) values)
///     atoms: Element symbols for all atoms
///     stabilizers_flat: Flat array of stabilizer matrices (9 values per matrix)
///     delta: Integer discretization factor
///
/// Returns:
///     Vec of canonical coordinate tuples (each 3*(n_atoms-1) values)
pub fn find_and_canonicalize_motif_neighbors(
    motif_coords: &[i32],
    atoms: &[String],
    stabilizers_flat: &[i32],
    delta: i32,
) -> Vec<Vec<i32>> {
    let n_atoms = atoms.len();
    let atom_labels = compute_atom_labels(atoms);
    let num_origin_atoms = atom_labels.iter().filter(|&&label| label == 0).count();

    // Add origin back to coordinates for perturbation generation
    let mut coords_with_origin = vec![0i32; n_atoms * 3];
    // Origin is at [0, 0, 0]
    // Copy remaining coords
    for i in 0..motif_coords.len() {
        let atom_idx = i / 3;
        let dim = i % 3;
        coords_with_origin[(atom_idx + 1) * 3 + dim] = motif_coords[i];
    }

    // Generate all perturbations
    let perturbations = generate_motif_perturbations(&coords_with_origin, n_atoms);

    // Canonicalize each perturbation
    let mut unique_results: HashSet<Vec<i32>> = HashSet::new();

    for perturbed in perturbations {
        // Convert to f64 and apply modulo
        let mut coords_f64: Vec<f64> = perturbed.iter().map(|&c| c as f64).collect();
        mnf::move_into_bounds(&mut coords_f64, delta as f64);

        // Build canonical MNF
        let canonical = mnf::build_mnf_vectorized(
            &coords_f64,
            n_atoms,
            &atom_labels,
            num_origin_atoms,
            stabilizers_flat,
            delta as f64,
        );

        // Convert back to integers
        let canonical_ints: Vec<i32> = canonical.iter().map(|&c| c.round() as i32).collect();
        unique_results.insert(canonical_ints);
    }

    // Filter out the original coords
    let original_ints: Vec<i32> = motif_coords.to_vec();
    unique_results.remove(&original_ints);

    unique_results.into_iter().collect()
}

/// Internal function for finding motif neighbors
fn find_motif_neighbor_tuples_internal(
    vonorms_i32: &[i32],
    coords_i32: &[i32],
    stabilizers_flat: &[i32],
    elements: &[String],
    delta: i32,
) -> Vec<(Vec<i32>, Vec<i32>)> {
    // Call the neighbor finding function
    let canonical_coords_list = find_and_canonicalize_motif_neighbors(
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
