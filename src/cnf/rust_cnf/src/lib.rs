mod permutations;
mod lnf;
mod mnf;

use pyo3::prelude::*;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};

/// Simple test function to verify Rust-Python integration works
#[pyfunction]
fn hello_rust() -> PyResult<String> {
    Ok("Hello from Rust! CNF optimization ready.".to_string())
}

/// Test function: sum an array (to verify numpy integration)
#[pyfunction]
fn sum_array<'py>(_py: Python<'py>, arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = arr.as_array();
    Ok(arr.iter().sum())
}

/// Rust implementation of build_lnf_raw for discretized vonorms (exact equality)
/// Returns: (canonical_vonorms, zero_idxs, selling_transform_flat, sorting_matrices)
#[pyfunction]
fn build_lnf_raw_rust<'py>(
    py: Python<'py>,
    vonorms: PyReadonlyArray1<f64>,
) -> PyResult<(Py<PyArray1<f64>>, Vec<usize>, Option<Vec<i32>>, Vec<Vec<i32>>)> {
    let vonorms_arr = vonorms.as_array();

    // Convert to fixed-size array
    if vonorms_arr.len() != 7 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "vonorms must have exactly 7 elements"
        ));
    }

    let mut vonorms_fixed = [0.0; 7];
    for (i, &v) in vonorms_arr.iter().enumerate() {
        vonorms_fixed[i] = v;
    }

    // Call Rust implementation (discretized version)
    let (canonical, zero_idxs, selling_flat, sorting_mats) = lnf::build_lnf_raw_discretized(&vonorms_fixed);

    // Convert back to numpy array
    let canonical_array = PyArray1::from_slice_bound(py, &canonical);

    Ok((canonical_array.into(), zero_idxs, selling_flat, sorting_mats))
}

/// Rust implementation of build_lnf_raw for float vonorms (tolerance-based)
/// Returns: (canonical_vonorms, zero_idxs, selling_transform_flat, sorting_matrices)
#[pyfunction]
fn build_lnf_raw_float_rust<'py>(
    py: Python<'py>,
    vonorms: PyReadonlyArray1<f64>,
    tol: f64,
) -> PyResult<(Py<PyArray1<f64>>, Vec<usize>, Option<Vec<i32>>, Vec<Vec<i32>>)> {
    let vonorms_arr = vonorms.as_array();

    // Convert to fixed-size array
    if vonorms_arr.len() != 7 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "vonorms must have exactly 7 elements"
        ));
    }

    let mut vonorms_fixed = [0.0; 7];
    for (i, &v) in vonorms_arr.iter().enumerate() {
        vonorms_fixed[i] = v;
    }

    // Call Rust implementation (float version with tolerance)
    let (canonical, zero_idxs, selling_flat, sorting_mats) = lnf::build_lnf_raw_float(&vonorms_fixed, tol);

    // Convert back to numpy array
    let canonical_array = PyArray1::from_slice_bound(py, &canonical);

    Ok((canonical_array.into(), zero_idxs, selling_flat, sorting_mats))
}

/// Rust implementation of stabilizer finding (exact equality for discretized)
/// Returns matrices as a 2D numpy array of shape (N, 3, 3)
#[pyfunction]
fn find_stabilizers_rust<'py>(
    py: Python<'py>,
    vonorms: PyReadonlyArray1<f64>,
) -> PyResult<Py<pyo3::PyAny>> {
    let vonorms_arr = vonorms.as_array();

    // Convert to fixed-size array
    if vonorms_arr.len() != 7 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "vonorms must have exactly 7 elements"
        ));
    }

    let mut vonorms_fixed = [0.0; 7];
    for (i, &v) in vonorms_arr.iter().enumerate() {
        vonorms_fixed[i] = v;
    }

    // Call Rust implementation - returns flat Vec<i32>
    let stabilizers_flat = lnf::find_stabilizers_raw(&vonorms_fixed);

    // Reshape to (N, 3, 3) where N = len / 9
    let n_matrices = stabilizers_flat.len() / 9;

    if n_matrices == 0 {
        // Return empty (0, 3, 3) array
        let empty: Vec<Vec<Vec<i32>>> = Vec::new();
        return Ok(pyo3::types::PyList::new_bound(py, empty).into());
    }

    // Convert to 3D structure
    let mut matrices: Vec<Vec<Vec<i32>>> = Vec::with_capacity(n_matrices);
    for i in 0..n_matrices {
        let start = i * 9;
        let mat = vec![
            stabilizers_flat[start..start+3].to_vec(),
            stabilizers_flat[start+3..start+6].to_vec(),
            stabilizers_flat[start+6..start+9].to_vec(),
        ];
        matrices.push(mat);
    }

    // Convert to nested Python lists (PyO3 will handle the conversion)
    Ok(pyo3::types::PyList::new_bound(py, matrices).into())
}

/// Rust implementation of stabilizer finding for float vonorms (tolerance-based)
/// Returns matrices as a 2D numpy array of shape (N, 3, 3)
#[pyfunction]
fn find_stabilizers_rust_float<'py>(
    py: Python<'py>,
    vonorms: PyReadonlyArray1<f64>,
    tol: f64,
) -> PyResult<Py<pyo3::PyAny>> {
    let vonorms_arr = vonorms.as_array();

    // Convert to fixed-size array
    if vonorms_arr.len() != 7 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "vonorms must have exactly 7 elements"
        ));
    }

    let mut vonorms_fixed = [0.0; 7];
    for (i, &v) in vonorms_arr.iter().enumerate() {
        vonorms_fixed[i] = v;
    }

    // Call Rust float implementation - returns flat Vec<i32>
    let stabilizers_flat = lnf::find_stabilizers_raw_float(&vonorms_fixed, tol);

    // Reshape to (N, 3, 3) where N = len / 9
    let n_matrices = stabilizers_flat.len() / 9;

    if n_matrices == 0 {
        // Return empty (0, 3, 3) array
        let empty: Vec<Vec<Vec<i32>>> = Vec::new();
        return Ok(pyo3::types::PyList::new_bound(py, empty).into());
    }

    // Convert to 3D structure
    let mut matrices: Vec<Vec<Vec<i32>>> = Vec::with_capacity(n_matrices);
    for i in 0..n_matrices {
        let start = i * 9;
        let mat = vec![
            stabilizers_flat[start..start+3].to_vec(),
            stabilizers_flat[start+3..start+6].to_vec(),
            stabilizers_flat[start+6..start+9].to_vec(),
        ];
        matrices.push(mat);
    }

    // Convert to nested Python lists (PyO3 will handle the conversion)
    Ok(pyo3::types::PyList::new_bound(py, matrices).into())
}

/// Combine and deduplicate stabilizer matrices (Rust implementation)
///
/// Takes two lists of matrices (as flat i32 arrays) and a middle matrix,
/// computes all combinations s1[i] @ middle @ s2[j], and returns unique results.
#[pyfunction]
fn combine_stabilizers_rust<'py>(
    py: Python<'py>,
    s1_flat: PyReadonlyArray1<i32>,
    s2_flat: PyReadonlyArray1<i32>,
    middle_2d: PyReadonlyArray1<i32>,
) -> PyResult<Py<pyo3::PyAny>> {
    let s1 = s1_flat.as_array();
    let s2 = s2_flat.as_array();
    let middle_flat = middle_2d.as_array();

    // Convert middle to 3x3 matrix
    if middle_flat.len() != 9 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "middle matrix must have exactly 9 elements (3x3)"
        ));
    }

    let middle = [
        [middle_flat[0], middle_flat[1], middle_flat[2]],
        [middle_flat[3], middle_flat[4], middle_flat[5]],
        [middle_flat[6], middle_flat[7], middle_flat[8]],
    ];

    // Call Rust implementation
    let result_flat = lnf::combine_stabilizers(
        s1.as_slice().unwrap(),
        s2.as_slice().unwrap(),
        &middle,
    );

    // Convert to (N, 3, 3) structure
    let n_matrices = result_flat.len() / 9;
    let mut matrices: Vec<Vec<Vec<i32>>> = Vec::with_capacity(n_matrices);

    for i in 0..n_matrices {
        let start = i * 9;
        let mat = vec![
            result_flat[start..start+3].to_vec(),
            result_flat[start+3..start+6].to_vec(),
            result_flat[start+6..start+9].to_vec(),
        ];
        matrices.push(mat);
    }

    Ok(pyo3::types::PyList::new_bound(py, matrices).into())
}

/// Build MNF (Motif Normal Form) using vectorized algorithm
#[pyfunction]
fn build_mnf_vectorized_rust<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray1<f64>,
    atom_labels: PyReadonlyArray1<i32>,
    num_origin_atoms: usize,
    stabilizers_flat: PyReadonlyArray1<i32>,
    mod_val: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let coords_arr = coords.as_array();
    let labels_arr = atom_labels.as_array();
    let stabs_arr = stabilizers_flat.as_array();

    let n_atoms = labels_arr.len();

    // Convert labels to usize
    let labels_usize: Vec<usize> = labels_arr.iter().map(|&x| x as usize).collect();

    // Call Rust implementation
    let mnf = mnf::build_mnf_vectorized(
        coords_arr.as_slice().unwrap(),
        n_atoms,
        &labels_usize,
        num_origin_atoms,
        stabs_arr.as_slice().unwrap(),
        mod_val,
    );

    // Convert to numpy array
    let mnf_array = PyArray1::from_slice_bound(py, &mnf);

    Ok(mnf_array.into())
}

/// Build MNFs for many coordinate sets in batch (vectorized across multiple motifs)
#[pyfunction]
fn build_mnf_batch_rust<'py>(
    py: Python<'py>,
    coords_batch: Vec<Vec<f64>>,     // List of coordinate arrays
    atom_labels: PyReadonlyArray1<i32>,
    num_origin_atoms: usize,
    stabilizers_flat: PyReadonlyArray1<i32>,
    mod_val: f64,
) -> PyResult<Py<pyo3::PyAny>> {
    let labels_arr = atom_labels.as_array();
    let stabs_arr = stabilizers_flat.as_array();

    let n_atoms = labels_arr.len();

    // Convert labels to usize
    let labels_usize: Vec<usize> = labels_arr.iter().map(|&x| x as usize).collect();

    // Call Rust batch implementation
    let mnf_results = mnf::build_mnf_batch(
        &coords_batch,
        n_atoms,
        &labels_usize,
        num_origin_atoms,
        stabs_arr.as_slice().unwrap(),
        mod_val,
    );

    // Convert results to Python list of numpy arrays
    let mnf_arrays: Vec<_> = mnf_results
        .into_iter()
        .map(|mnf| PyArray1::from_vec_bound(py, mnf).into_py(py))
        .collect();

    Ok(pyo3::types::PyList::new_bound(py, mnf_arrays).into())
}

/// Combined pipeline: find stabilizers for both vonorm arrays and combine them
/// Returns a flat numpy array that can be reshaped to (N, 3, 3) on Python side
#[pyfunction]
fn find_and_combine_stabilizers_rust<'py>(
    py: Python<'py>,
    vonorms1: PyReadonlyArray1<f64>,
    vonorms2: PyReadonlyArray1<f64>,
    middle_flat: PyReadonlyArray1<i32>,
) -> PyResult<Py<PyArray1<i32>>> {
    // Convert vonorms arrays to fixed-size arrays
    let vonorms1_arr = vonorms1.as_array();
    let vonorms2_arr = vonorms2.as_array();

    if vonorms1_arr.len() != 7 || vonorms2_arr.len() != 7 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "vonorms arrays must have exactly 7 elements"
        ));
    }

    let mut vonorms1_fixed = [0.0; 7];
    let mut vonorms2_fixed = [0.0; 7];
    for i in 0..7 {
        vonorms1_fixed[i] = vonorms1_arr[i];
        vonorms2_fixed[i] = vonorms2_arr[i];
    }

    // Convert middle matrix
    let middle_arr = middle_flat.as_array();
    if middle_arr.len() != 9 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "middle matrix must have exactly 9 elements (3x3)"
        ));
    }

    let mut middle = [[0i32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            middle[i][j] = middle_arr[i * 3 + j];
        }
    }

    // OLD IMPLEMENTATION (testing if optimization is incorrect):
    let s1_flat = lnf::find_stabilizers_raw(&vonorms1_fixed);
    let s2_flat = lnf::find_stabilizers_raw(&vonorms2_fixed);
    let combined_flat = lnf::combine_stabilizers(&s1_flat, &s2_flat, &middle);

    // OPTIMIZED: Only find s2 stabilizers, skip s1 orbit
    // NEW IMPLEMENTATION (commented out for testing):
    // let s2_flat = lnf::find_stabilizers_raw(&vonorms2_fixed);
    // let combined_flat = lnf::combine_middle_and_s2_stabilizers(&s2_flat, &middle);

    // Return as flat numpy array (Python will reshape to (N, 3, 3))
    let result_array = PyArray1::from_vec_bound(py, combined_flat);
    Ok(result_array.into())
}

/// Combined pipeline for float vonorms (tolerance-based)
/// Returns a flat numpy array that can be reshaped to (N, 3, 3) on Python side
#[pyfunction]
fn find_and_combine_stabilizers_rust_float<'py>(
    py: Python<'py>,
    vonorms1: PyReadonlyArray1<f64>,
    vonorms2: PyReadonlyArray1<f64>,
    middle_flat: PyReadonlyArray1<i32>,
    tol: f64,
) -> PyResult<Py<PyArray1<i32>>> {
    // Convert vonorms arrays to fixed-size arrays
    let vonorms1_arr = vonorms1.as_array();
    let vonorms2_arr = vonorms2.as_array();

    if vonorms1_arr.len() != 7 || vonorms2_arr.len() != 7 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "vonorms arrays must have exactly 7 elements"
        ));
    }

    let mut vonorms1_fixed = [0.0; 7];
    let mut vonorms2_fixed = [0.0; 7];
    for i in 0..7 {
        vonorms1_fixed[i] = vonorms1_arr[i];
        vonorms2_fixed[i] = vonorms2_arr[i];
    }

    // Convert middle matrix
    let middle_arr = middle_flat.as_array();
    if middle_arr.len() != 9 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "middle matrix must have exactly 9 elements (3x3)"
        ));
    }

    let mut middle = [[0i32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            middle[i][j] = middle_arr[i * 3 + j];
        }
    }

    // OLD IMPLEMENTATION (testing if optimization is incorrect):
    let s1_flat = lnf::find_stabilizers_raw_float(&vonorms1_fixed, tol);
    let s2_flat = lnf::find_stabilizers_raw_float(&vonorms2_fixed, tol);
    let combined_flat = lnf::combine_stabilizers(&s1_flat, &s2_flat, &middle);

    // OPTIMIZED: Only find s2 stabilizers, skip s1 orbit
    // NEW IMPLEMENTATION (commented out for testing):
    // let s2_flat = lnf::find_stabilizers_raw_float(&vonorms2_fixed, tol);
    // let combined_flat = lnf::combine_middle_and_s2_stabilizers(&s2_flat, &middle);

    // Return as flat numpy array (Python will reshape to (N, 3, 3))
    let result_array = PyArray1::from_vec_bound(py, combined_flat);
    Ok(result_array.into())
}

/// Validate a vonorm step
///
/// Combines three validation checks:
/// 1. has_valid_conorms_exact: Check if zero conorm set is valid
/// 2. is_obtuse: All conorms should be <= 0
/// 3. is_superbasis_exact: primary_sum == secondary_sum
fn validate_vonorm_step(vonorms: &[f64; 7]) -> bool {
    use crate::permutations::{compute_conorms, find_zero_indices_exact, PERMUTATIONS};

    // 1. Compute conorms
    let conorms = compute_conorms(vonorms);

    // 2. Check if zero conorm set is valid (has_valid_conorms_exact)
    let zero_indices = find_zero_indices_exact(&conorms);
    if !PERMUTATIONS.zero_to_perm_to_mats.contains_key(&zero_indices) {
        return false;
    }

    // 3. Check if obtuse (all conorms <= 0)
    if !conorms.iter().all(|&c| c <= 0.0) {
        return false;
    }

    // 4. Check if superbasis (primary_sum == secondary_sum)
    let primary_sum: f64 = vonorms[0..4].iter().sum();
    let secondary_sum: f64 = vonorms[4..7].iter().sum();
    if (primary_sum - secondary_sum).abs() > 1e-10 {
        return false;
    }

    true
}

/// Compute step data for lattice neighbor finding
///
/// This function performs all the heavy numpy operations for step data computation:
/// - Matrix multiplication via einsum-like operations
/// - Deduplication of matrices
/// - Coordinate transformations
/// - Step vector generation
/// - Vonorm validation (NEW: filters invalid steps before returning)
///
/// Returns: List of (step_vec, vonorms_tuple, transformed_coords, matrix) for VALID steps only
#[pyfunction]
fn compute_step_data_raw_rust<'py>(
    py: Python<'py>,
    current_stabilizers_flat: PyReadonlyArray1<i32>,  // Flat (N1*9) array
    input_vonorms: PyReadonlyArray1<f64>,  // Input vonorms (7 elements)
    motif_coord_matrix: PyReadonlyArray1<f64>,  // Flat (3*N) array
    n_atoms: usize,
    motif_delta: i32,
) -> PyResult<Py<pyo3::PyAny>> {
    use std::collections::HashMap;
    use numpy::PyArray2;
    use crate::permutations::{compute_conorms, find_zero_indices_exact, get_s4_representatives};

    let s1_flat = current_stabilizers_flat.as_array();
    let coord_mat = motif_coord_matrix.as_array();
    let input_vn = input_vonorms.as_array();

    // Convert input vonorms to array
    let vonorms_arr: [f64; 7] = [
        input_vn[0], input_vn[1], input_vn[2], input_vn[3],
        input_vn[4], input_vn[5], input_vn[6]
    ];

    // Compute conorms and zero indices
    let conorms = compute_conorms(&vonorms_arr);
    let zero_indices = find_zero_indices_exact(&conorms);

    // Get S4 representatives using precomputed data
    let s4_reps = get_s4_representatives(&vonorms_arr, &zero_indices);

    // Parse current stabilizers into 3x3 matrices
    let n_s1 = s1_flat.len() / 9;
    let mut s1_matrices: Vec<[[i32; 3]; 3]> = Vec::with_capacity(n_s1);
    for i in 0..n_s1 {
        let start = i * 9;
        let mut mat = [[0i32; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                mat[row][col] = s1_flat[start + row * 3 + col];
            }
        }
        s1_matrices.push(mat);
    }

    // Generate step vectors (same as LatticeStep.all_step_vecs())
    let step_vecs = generate_step_vectors();

    let mut all_step_data: Vec<(Vec<i32>, Vec<f64>, Vec<f64>, Vec<i32>)> = Vec::new();

    // Process each S4 representative
    for s4_rep in s4_reps {
        let permuted_vonorms = s4_rep.permuted_vonorms.to_vec();
        let transform_mats = s4_rep.transition_mats;
        // Convert permuted vonorms to array for stabilizer computation
        let permuted_vn_arr: [f64; 7] = [
            permuted_vonorms[0], permuted_vonorms[1], permuted_vonorms[2], permuted_vonorms[3],
            permuted_vonorms[4], permuted_vonorms[5], permuted_vonorms[6]
        ];

        // Compute stabilizers in Rust instead of receiving from Python
        let s2_flat = lnf::find_stabilizers_raw(&permuted_vn_arr);

        // Parse transform and s2 matrices
        let n_t = transform_mats.len();
        let n_s2 = s2_flat.len() / 9;

        let mut t_matrices: Vec<[[i32; 3]; 3]> = Vec::with_capacity(n_t);
        for mat_vec in &transform_mats {
            // mat_vec is Vec<Vec<i32>> (3x3 matrix)
            let mut mat = [[0i32; 3]; 3];
            for row in 0..3 {
                for col in 0..3 {
                    mat[row][col] = mat_vec[row][col];
                }
            }
            t_matrices.push(mat);
        }

        let mut s2_matrices: Vec<[[i32; 3]; 3]> = Vec::with_capacity(n_s2);
        for i in 0..n_s2 {
            let start = i * 9;
            let mut mat = [[0i32; 3]; 3];
            for row in 0..3 {
                for col in 0..3 {
                    mat[row][col] = s2_flat[start + row * 3 + col];
                }
            }
            s2_matrices.push(mat);
        }

        // Compute all s1 @ t @ s2 products and deduplicate
        let mut unique_products: HashMap<Vec<i32>, [[i32; 3]; 3]> = HashMap::new();

        for s1 in &s1_matrices {
            // Only use first transition matrix (matching Python behavior: transform_mats[:1])
            for t in &t_matrices[..1.min(t_matrices.len())] {
                for s2 in &s2_matrices {
                    // Compute s1 @ t
                    let st = matrix_multiply_i32(s1, t);
                    // Compute (s1 @ t) @ s2
                    let product = matrix_multiply_i32(&st, s2);

                    // Flatten for dedup key
                    let mut flat = Vec::with_capacity(9);
                    for row in 0..3 {
                        for col in 0..3 {
                            flat.push(product[row][col]);
                        }
                    }

                    unique_products.insert(flat, product);
                }
            }
        }

        // For each unique matrix, transform motif and generate steps
        // Sort by flattened matrix to match Python's np.unique() behavior
        let mut sorted_products: Vec<(Vec<i32>, [[i32; 3]; 3])> = unique_products.into_iter().collect();
        sorted_products.sort_by(|a, b| a.0.cmp(&b.0));

        for (mat_flat, mat) in sorted_products {
            // Invert matrix and transform coordinates
            let mat_inv = matrix_inverse_i32(&mat);

            // Transform coordinates: mat_inv @ coord_matrix
            let mut transformed_coords = vec![0.0; coord_mat.len()];
            for atom_idx in 0..n_atoms {
                for row in 0..3 {
                    let mut sum = 0.0;
                    for col in 0..3 {
                        sum += mat_inv[row][col] as f64 * coord_mat[col * n_atoms + atom_idx];
                    }
                    // Apply modulo
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
        // Round vonorms and coords for dedup key
        let vonorms_key: Vec<i32> = vonorms.iter().map(|&v| v.round() as i32).collect();
        let coords_key: Vec<i32> = coords.iter().map(|&v| (v * 1000.0).round() as i32).collect();

        let key = (vonorms_key, coords_key);
        unique_steps.entry(key).or_insert((step_vec, vonorms, coords, mat));
    }

    // Convert to Python list of tuples (no validation - Python validates during CNF construction)
    // Sort by key to ensure deterministic ordering (matching Python dict iteration)
    let mut result: Vec<_> = unique_steps.into_iter().collect();
    result.sort_by(|a, b| a.0.cmp(&b.0));

    let result_values: Vec<_> = result.into_iter().map(|(_, v)| v).collect();

    Ok(pyo3::types::PyList::new_bound(py, result_values).into())
}

// Helper: Generate step vectors (equivalent to LatticeStep.all_step_vecs())
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

// Helper: Matrix multiplication for i32 matrices
fn matrix_multiply_i32(a: &[[i32; 3]; 3], b: &[[i32; 3]; 3]) -> [[i32; 3]; 3] {
    let mut result = [[0i32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

// Helper: Matrix inversion for i32 matrices
fn matrix_inverse_i32(mat: &[[i32; 3]; 3]) -> [[f64; 3]; 3] {
    // Convert to f64, invert, then we'll keep as f64 for precision
    let mut m = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = mat[i][j] as f64;
        }
    }

    // Compute determinant
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-10 {
        panic!("Matrix is singular and cannot be inverted");
    }

    // Compute adjugate matrix
    let mut inv = [[0.0f64; 3]; 3];
    inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
    inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det;
    inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
    inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det;
    inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
    inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;
    inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;
    inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det;
    inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;

    inv
}

/// Build CNFs from validated step data
///
/// Takes the output of compute_step_data_raw_rust (which is already validated)
/// and returns non-canonical vonorms and coords for Python to canonicalize.
///
/// Returns: List of (vonorms, coords, atoms) tuples (not yet canonicalized)
#[pyfunction]
fn build_cnfs_from_step_data_rust<'py>(
    py: Python<'py>,
    step_data_list: &Bound<'py, pyo3::types::PyList>,
    _input_vonorms: PyReadonlyArray1<f64>,
    _input_coords: PyReadonlyArray1<f64>,
    atoms: Vec<String>,
    _xi: f64,
    _delta: i32,
) -> PyResult<Py<pyo3::types::PyList>> {
    use pyo3::types::{PyTuple, PyList};

    let result_list = PyList::empty_bound(py);

    // Process each step - just extract and return the data
    // Let Python's CNF constructor do the full canonicalization
    for item in step_data_list.iter() {
        let tuple: &Bound<'_, PyTuple> = item.downcast()?;

        // Extract step data: (step_vec, vonorms, coords, matrix)
        let vonorms_list: Vec<f64> = tuple.get_item(1)?.extract()?;

        // coords might be a 2D numpy array, so we need to handle it carefully
        let coords_obj = tuple.get_item(2)?;
        let coords_list: Vec<f64> = if let Ok(arr) = coords_obj.downcast::<PyArray1<f64>>() {
            // 1D array - extract directly
            arr.readonly().as_slice()?.to_vec()
        } else if let Ok(arr2d) = coords_obj.downcast::<numpy::PyArray2<f64>>() {
            // 2D array - flatten it
            arr2d.readonly().as_array().iter().copied().collect()
        } else {
            // Try extracting as a list of lists or flat list
            coords_obj.extract()?
        };

        // Return non-canonical data for Python to canonicalize
        result_list.append((
            vonorms_list,
            coords_list,
            atoms.clone()
        ))?;
    }

    Ok(result_list.into())
}

/// Batch CNF canonicalization - takes step data and returns fully canonical CNF data.
///
/// This combines LNF construction, stabilizer finding, and MNF construction
/// in a single batch operation to minimize Python/Rust boundary crossings.
///
/// Returns: List of (canonical_vonorms, canonical_coords) tuples
#[pyfunction]
fn canonicalize_cnfs_batch_rust<'py>(
    py: Python<'py>,
    step_data_list: &Bound<'py, pyo3::types::PyList>,
    atoms: Vec<String>,
    xi: f64,
    delta: i32,
) -> PyResult<Py<pyo3::types::PyList>> {
    use pyo3::types::{PyList, PyTuple};

    let result_list = PyList::empty_bound(py);
    let atom_labels = compute_atom_labels(&atoms);
    let n_atoms = atoms.len();
    // Count how many atoms have the first element type (label 0)
    let num_origin_atoms = atom_labels.iter().filter(|&&label| label == 0).count();

    // Process each candidate CNF
    for item in step_data_list.iter() {
        let tuple: &Bound<'_, PyTuple> = item.downcast()?;

        // Extract step data: (step_vec, vonorms, coords, matrix)
        let vonorms_list: Vec<f64> = tuple.get_item(1)?.extract()?;

        // Extract coords
        let coords_obj = tuple.get_item(2)?;
        let coords_list: Vec<f64> = if let Ok(arr) = coords_obj.downcast::<PyArray1<f64>>() {
            arr.readonly().as_slice()?.to_vec()
        } else if let Ok(arr2d) = coords_obj.downcast::<numpy::PyArray2<f64>>() {
            arr2d.readonly().as_array().iter().copied().collect()
        } else {
            coords_obj.extract()?
        };

        // Step 1: Build LNF (canonicalize vonorms)
        // Extend vonorms from 6 to 7 elements (7th is padding/derived)
        let mut vonorms_arr = [0.0; 7];
        for (i, &v) in vonorms_list.iter().enumerate().take(7) {
            vonorms_arr[i] = v;
        }

        let (canonical_vonorms_vec, _, selling_flat_opt, sorting_mats) =
            lnf::build_lnf_raw_discretized(&vonorms_arr);

        // Convert canonical vonorms back to array
        let mut canonical_vonorms = [0.0; 7];
        for (i, &v) in canonical_vonorms_vec.iter().enumerate().take(7) {
            canonical_vonorms[i] = v;
        }

        // Step 2: Compute transformation matrix (middle = selling @ sorting)
        let selling_mat = if let Some(selling_flat) = selling_flat_opt {
            [
                [selling_flat[0], selling_flat[1], selling_flat[2]],
                [selling_flat[3], selling_flat[4], selling_flat[5]],
                [selling_flat[6], selling_flat[7], selling_flat[8]],
            ]
        } else {
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  // Identity
        };

        // First sorting matrix is a flattened 9-element vec
        let sorting_flat = &sorting_mats[0];
        let sorting_mat = [
            [sorting_flat[0], sorting_flat[1], sorting_flat[2]],
            [sorting_flat[3], sorting_flat[4], sorting_flat[5]],
            [sorting_flat[6], sorting_flat[7], sorting_flat[8]],
        ];

        // Inline matrix multiplication (middle = selling @ sorting)
        let mut middle = [[0i32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                middle[i][j] = selling_mat[i][0] * sorting_mat[0][j]
                             + selling_mat[i][1] * sorting_mat[1][j]
                             + selling_mat[i][2] * sorting_mat[2][j];
            }
        }

        // Step 3: Find stabilizers and combine with middle transformation
        // Need BOTH s1 (original vonorms) and s2 (canonical vonorms) stabilizers
        // to compute s1 @ middle @ s2 (matching Python line 133 in cnf_constructor.py)
        let s1_flat = lnf::find_stabilizers_raw(&vonorms_arr);
        let s2_flat = lnf::find_stabilizers_raw(&canonical_vonorms);
        let combined_stabs_flat = lnf::combine_stabilizers(&s1_flat, &s2_flat, &middle);

        // Step 4: Build canonical MNF
        // Convert atom labels from i32 to usize
        let labels_usize: Vec<usize> = atom_labels.iter().map(|&x| x as usize).collect();

        let canonical_coords = mnf::build_mnf_vectorized(
            &coords_list,
            n_atoms,
            &labels_usize,
            num_origin_atoms,
            &combined_stabs_flat,
            delta as f64,
        );

        // Return canonical data
        result_list.append((
            canonical_vonorms.to_vec(),
            canonical_coords,
        ))?;
    }

    Ok(result_list.into())
}

/// Helper: Compute atom labels from atom symbols
fn compute_atom_labels(atoms: &[String]) -> Vec<i32> {
    let mut labels = Vec::new();
    let mut current_label = 0i32;
    let mut prev_atom = "";

    for atom in atoms {
        if atom != prev_atom && !prev_atom.is_empty() {
            current_label += 1;
        }
        labels.push(current_label);
        prev_atom = atom;
    }

    labels
}

/// Get S4 maximal representatives for vonorms (matching Python's maximally_ascending_equivalence_class_members)
/// Returns a list of dictionaries, each containing:
///   - 's4_key': The S4 key (sorted first 4 permutation indices)
///   - 'maximal_permuted_vonorms': The maximal vonorm tuple for this S4 group
///   - 'transition_matrices': List of transition matrices for this group
#[pyfunction]
fn get_s4_maximal_representatives_rust<'py>(
    py: Python<'py>,
    vonorms: PyReadonlyArray1<f64>,
) -> PyResult<Py<pyo3::PyAny>> {
    use crate::permutations::{compute_conorms, find_zero_indices_exact, get_s4_representatives};
    use pyo3::types::{PyDict, PyList};

    let vonorms_arr = vonorms.as_array();

    if vonorms_arr.len() != 7 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "vonorms must have exactly 7 elements"
        ));
    }

    let mut vonorms_fixed = [0.0; 7];
    for (i, &v) in vonorms_arr.iter().enumerate() {
        vonorms_fixed[i] = v;
    }

    // Compute conorms and zero indices
    let conorms = compute_conorms(&vonorms_fixed);
    let zero_indices = find_zero_indices_exact(&conorms);

    // Get S4 representatives
    let s4_reps = get_s4_representatives(&vonorms_fixed, &zero_indices);

    // Convert to Python list of dicts
    let result_list = PyList::empty_bound(py);

    for rep in s4_reps {
        let dict = PyDict::new_bound(py);

        // Add S4 key (sorted first 4 permutation indices)
        let s4_key_tuple = PyList::empty_bound(py);
        for &idx in &rep.s4_key {
            s4_key_tuple.append(idx)?;
        }
        dict.set_item("s4_key", s4_key_tuple)?;

        // Add maximal permuted vonorms as tuple
        let vonorms_tuple = PyList::empty_bound(py);
        for &v in &rep.permuted_vonorms {
            vonorms_tuple.append(v)?;
        }
        dict.set_item("maximal_permuted_vonorms", vonorms_tuple)?;

        // Add transition matrices as list of 3x3 matrices
        let mats_list = PyList::empty_bound(py);
        for mat in rep.transition_mats {
            let py_mat = PyList::empty_bound(py);
            for row in mat {
                let py_row = PyList::empty_bound(py);
                for &val in &row {
                    py_row.append(val)?;
                }
                py_mat.append(py_row)?;
            }
            mats_list.append(py_mat)?;
        }
        dict.set_item("transition_matrices", mats_list)?;

        result_list.append(dict)?;
    }

    Ok(result_list.into())
}

#[pymodule]
fn rust_cnf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_rust, m)?)?;
    m.add_function(wrap_pyfunction!(sum_array, m)?)?;
    m.add_function(wrap_pyfunction!(build_lnf_raw_rust, m)?)?;
    m.add_function(wrap_pyfunction!(build_lnf_raw_float_rust, m)?)?;
    m.add_function(wrap_pyfunction!(find_stabilizers_rust, m)?)?;
    m.add_function(wrap_pyfunction!(find_stabilizers_rust_float, m)?)?;
    m.add_function(wrap_pyfunction!(combine_stabilizers_rust, m)?)?;
    m.add_function(wrap_pyfunction!(find_and_combine_stabilizers_rust, m)?)?;
    m.add_function(wrap_pyfunction!(find_and_combine_stabilizers_rust_float, m)?)?;
    m.add_function(wrap_pyfunction!(build_mnf_vectorized_rust, m)?)?;
    m.add_function(wrap_pyfunction!(build_mnf_batch_rust, m)?)?;
    m.add_function(wrap_pyfunction!(compute_step_data_raw_rust, m)?)?;
    m.add_function(wrap_pyfunction!(build_cnfs_from_step_data_rust, m)?)?;
    m.add_function(wrap_pyfunction!(canonicalize_cnfs_batch_rust, m)?)?;
    m.add_function(wrap_pyfunction!(get_s4_maximal_representatives_rust, m)?)?;
    Ok(())
}
