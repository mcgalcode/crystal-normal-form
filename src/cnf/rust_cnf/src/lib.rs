mod linalg;
mod permutations;
mod lnf;
mod mnf;
mod geometry;
mod pathfinding;
mod bidirectional;
mod neighbors;
mod heuristics;

use pyo3::prelude::*;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};

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
    input_vonorms: PyReadonlyArray1<f64>,  // Input vonorms (7 elements)
    motif_coord_matrix: PyReadonlyArray1<f64>,  // Flat (3*N) array
    n_atoms: usize,
    motif_delta: i32,
) -> PyResult<Py<pyo3::PyAny>> {
    // Extract arrays
    let coord_mat = motif_coord_matrix.as_array();
    let input_vn = input_vonorms.as_array();

    // Convert input vonorms to array
    let vonorms_arr: [f64; 7] = [
        input_vn[0], input_vn[1], input_vn[2], input_vn[3],
        input_vn[4], input_vn[5], input_vn[6]
    ];

    // Call internal function (convert numpy arrays to slices)
    let result_values = compute_step_data_raw_internal(
        &vonorms_arr,
        coord_mat.as_slice().unwrap(),
        n_atoms,
        motif_delta,
    );

    // Convert to Python list
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

/// Internal version of compute_step_data_raw_rust that works with Rust types
/// This is the core logic extracted for reuse in pure-Rust pathfinding
/// Note: Computes stabilizers internally from vonorms
pub(crate) fn compute_step_data_raw_internal(
    input_vonorms: &[f64; 7],
    motif_coord_matrix: &[f64],  // Flat (3*N) array
    n_atoms: usize,
    motif_delta: i32,
) -> Vec<(Vec<i32>, Vec<f64>, Vec<f64>, Vec<i32>)> {
    use std::collections::HashMap;
    use crate::permutations::{compute_conorms, find_zero_indices_exact, get_s4_representatives};

    // Compute conorms and zero indices
    let conorms = compute_conorms(input_vonorms);
    let zero_indices = find_zero_indices_exact(&conorms);

    // Get S4 representatives using precomputed data
    let s4_reps = get_s4_representatives(input_vonorms, &zero_indices);

    // Compute current stabilizers from input vonorms
    let s1_flat = lnf::find_stabilizers_raw(input_vonorms);
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

        // Compute stabilizers in Rust
        let s2_flat = lnf::find_stabilizers_raw(&s4_rep.permuted_vonorms);

        // Parse transform and s2 matrices
        let n_t = transform_mats.len();
        let n_s2 = s2_flat.len() / 9;

        let mut t_matrices: Vec<[[i32; 3]; 3]> = Vec::with_capacity(n_t);
        for mat_vec in &transform_mats {
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
            for t in &t_matrices[..1.min(t_matrices.len())] {
                for s2 in &s2_matrices {
                    let st = mat_mul(s1, t);
                    let product = mat_mul(&st, s2);

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
    // Note: coords should already be integers (from modulo operation), so just round them
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

// Use shared functions
use linalg::{mat_mul, mat_inv_f64};
use mnf::compute_atom_labels;

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
    _xi: f64,
    delta: i32,
) -> PyResult<Py<pyo3::types::PyList>> {
    use pyo3::types::{PyList, PyTuple};

    // Convert Python list to Rust Vec
    let mut step_data = Vec::new();
    for item in step_data_list.iter() {
        let tuple: &Bound<'_, PyTuple> = item.downcast()?;

        // Extract step data: (step_vec, vonorms, coords, matrix)
        let step_vec: Vec<i32> = tuple.get_item(0)?.extract()?;
        let vonorms: Vec<f64> = tuple.get_item(1)?.extract()?;

        let coords_obj = tuple.get_item(2)?;
        let coords: Vec<f64> = if let Ok(arr) = coords_obj.downcast::<PyArray1<f64>>() {
            arr.readonly().as_slice()?.to_vec()
        } else if let Ok(arr2d) = coords_obj.downcast::<numpy::PyArray2<f64>>() {
            arr2d.readonly().as_array().iter().copied().collect()
        } else {
            coords_obj.extract()?
        };

        let matrix: Vec<i32> = tuple.get_item(3)?.extract()?;

        step_data.push((step_vec, vonorms, coords, matrix));
    }

    // Call internal function
    let results = canonicalize_cnfs_batch_internal(step_data, &atoms, delta);

    // Convert results back to Python
    let result_list = PyList::empty_bound(py);
    for (vonorms_ints, coords_ints) in results {
        let vonorms_tuple = PyTuple::new_bound(py, &vonorms_ints);
        let coords_tuple = PyTuple::new_bound(py, &coords_ints);
        result_list.append((vonorms_tuple, coords_tuple))?;
    }

    Ok(result_list.into())
}

/// Combined pipeline that replicates Python lines 94-122 in lattice_neighbor_finder.py
///
/// This function does EXACTLY what Python does:
/// 1. Call compute_step_data_raw_rust (line 94)
/// 2. Validate steps (lines 97-107)
/// 3. Call canonicalize_cnfs_batch_rust (lines 117-122)
#[pyfunction]
fn find_and_canonicalize_lattice_neighbors<'py>(
    py: Python<'py>,
    input_vonorms: PyReadonlyArray1<f64>,
    motif_coords_flat: PyReadonlyArray1<f64>,
    n_atoms: usize,
    motif_delta: i32,
    atoms: Vec<String>,
    xi: f64,
    delta: i32,
) -> PyResult<Py<pyo3::types::PyList>> {
    // Step 1: Get step data (line 94) - stabilizers computed internally
    let step_data = compute_step_data_raw_rust(
        py,
        input_vonorms,
        motif_coords_flat,
        n_atoms,
        motif_delta
    )?;

    // Step 2: Validate steps (lines 97-107) - extracted to separate function for testing
    let validated_steps = validate_step_data_rust(py, step_data)?;

    // Step 3: Canonicalize (lines 117-122)
    canonicalize_cnfs_batch_rust(py, &validated_steps, atoms, xi, delta)
}

/// Internal version: Validate step data by filtering steps that pass vonorm validation
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

/// Find all neighbor tuples (lattice + motif) for a given CNF state
/// Python binding for the pure-Rust neighbor finding
/// Note: Computes stabilizers internally from vonorms
#[pyfunction]
fn find_neighbor_tuples_rust<'py>(
    py: Python<'py>,
    vonorms_i32: PyReadonlyArray1<i32>,
    coords_i32: PyReadonlyArray1<i32>,
    elements: Vec<String>,
    n_atoms: usize,
    xi: f64,
    delta: i32,
) -> PyResult<Py<pyo3::types::PyList>> {
    use pyo3::types::{PyList, PyTuple};

    // Convert to slices
    let vonorms_slice = vonorms_i32.as_slice()?;
    let coords_slice = coords_i32.as_slice()?;

    // Call internal function
    let results = crate::neighbors::find_neighbor_tuples(
        vonorms_slice,
        coords_slice,
        &elements,
        n_atoms,
        xi,
        delta,
    );

    // Convert results to Python list of tuples
    let py_results = PyList::empty_bound(py);
    for (vonorms, coords) in results {
        let vonorms_tuple = PyTuple::new_bound(py, &vonorms);
        let coords_tuple = PyTuple::new_bound(py, &coords);
        py_results.append((vonorms_tuple, coords_tuple))?;
    }

    Ok(py_results.into())
}

/// Validate step data by filtering steps that pass vonorm validation
/// This replicates Python lines 97-107 in lattice_neighbor_finder.py
#[pyfunction]
fn validate_step_data_rust<'py>(
    py: Python<'py>,
    step_data: Py<pyo3::PyAny>,
) -> PyResult<Bound<'py, pyo3::types::PyList>> {
    use pyo3::types::{PyList, PyTuple};

    let step_list = step_data.downcast_bound::<PyList>(py)?;
    let validated_steps = PyList::empty_bound(py);

    for item in step_list.iter() {
        let tuple = item.downcast::<PyTuple>()?;

        // Extract vonorms from tuple[1]
        let vonorms_list: Vec<f64> = tuple.get_item(1)?.extract()?;
        let mut step_vonorms = [0.0; 7];
        for (i, &v) in vonorms_list.iter().enumerate().take(7) {
            step_vonorms[i] = v;
        }

        // Validate (lines 102-105)
        if !lnf::has_valid_conorms_exact(&step_vonorms) {
            continue;
        }
        if !lnf::is_obtuse(&step_vonorms) || !lnf::is_superbasis_exact(&step_vonorms) {
            continue;
        }

        // Append to validated list (line 107)
        validated_steps.append(item)?;
    }

    Ok(validated_steps)
}


/// Internal version: Canonicalize a batch of CNF step data
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
        let mut vonorms_arr = [0.0; 7];
        for (i, &v) in vonorms.iter().enumerate().take(7) {
            vonorms_arr[i] = v;
        }

        let (canonical_vonorms_vec, _, selling_flat_opt, sorting_mats) =
            lnf::build_lnf_raw_discretized(&vonorms_arr);

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
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        };

        let sorting_flat = &sorting_mats[0];
        let sorting_mat = [
            [sorting_flat[0], sorting_flat[1], sorting_flat[2]],
            [sorting_flat[3], sorting_flat[4], sorting_flat[5]],
            [sorting_flat[6], sorting_flat[7], sorting_flat[8]],
        ];

        let mut middle = [[0i32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                middle[i][j] = selling_mat[i][0] * sorting_mat[0][j]
                             + selling_mat[i][1] * sorting_mat[1][j]
                             + selling_mat[i][2] * sorting_mat[2][j];
            }
        }

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

/// Find and canonicalize motif neighbors
///
/// This function wraps the Rust implementation of motif neighbor finding.
///
/// Arguments:
/// - motif_coords: Coordinate list WITHOUT origin (length = (n_atoms-1)*3)
/// - atoms: List of atom symbols INCLUDING origin (length = n_atoms)
/// - stabilizers_flat: Flattened 3x3 stabilizer matrices
/// - delta: Discretization parameter
///
/// Returns: List of canonical motif coordinate lists (WITHOUT origin)
#[pyfunction]
fn find_and_canonicalize_motif_neighbors<'py>(
    py: Python<'py>,
    motif_coords: Vec<i32>,
    atoms: Vec<String>,
    stabilizers_flat: PyReadonlyArray1<i32>,
    delta: i32,
) -> PyResult<Py<pyo3::types::PyList>> {
    use pyo3::types::PyList;

    // Convert stabilizers to slice
    let stabs_slice = stabilizers_flat.as_slice()?;

    // Call Rust implementation
    let results = mnf::find_and_canonicalize_motif_neighbors(
        &motif_coords,
        &atoms,
        stabs_slice,
        delta,
    );

    // Convert results to Python list of tuples (not lists)
    // This avoids Python-side conversion overhead
    use pyo3::types::PyTuple;
    let py_results = PyList::empty_bound(py);
    for result in results {
        let coords_tuple = PyTuple::new_bound(py, &result);
        py_results.append(coords_tuple)?;
    }

    Ok(py_results.unbind())
}

#[pyfunction]
fn reconstruct_structure_from_cnf<'py>(
    _py: Python<'py>,
    vonorms: PyReadonlyArray1<f64>,
    coords: PyReadonlyArray1<i32>,
    n_atoms: usize,
    xi: f64,
    delta: i32,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    use crate::geometry::{vonorms_to_lattice_matrix, coords_to_cartesian_positions};

    // Convert vonorms to array
    let vonorms_slice = vonorms.as_slice()?;
    if vonorms_slice.len() != 7 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "vonorms must have exactly 7 elements"
        ));
    }
    let mut vonorms_arr = [0.0; 7];
    vonorms_arr.copy_from_slice(vonorms_slice);

    // Convert coords to slice
    let coords_slice = coords.as_slice()?;

    // Reconstruct lattice matrix from vonorms
    let lattice_matrix = vonorms_to_lattice_matrix(&vonorms_arr, xi);

    // Reconstruct cartesian positions from coords
    let cart_coords = coords_to_cartesian_positions(coords_slice, n_atoms, delta, &lattice_matrix);

    // Flatten lattice matrix to return
    let lattice_flat = vec![
        lattice_matrix[0][0], lattice_matrix[0][1], lattice_matrix[0][2],
        lattice_matrix[1][0], lattice_matrix[1][1], lattice_matrix[1][2],
        lattice_matrix[2][0], lattice_matrix[2][1], lattice_matrix[2][2],
    ];

    Ok((lattice_flat, cart_coords))
}

/// A* pathfinding from multiple start CNFs to multiple goal CNFs (pure Rust implementation)
///
/// Args:
///     start_points: List of (vonorms, coords) tuples for starting CNFs
///     goal_points: List of (vonorms, coords) tuples for goal CNFs
///     elements: List of element symbols for all atoms (including origin)
///     n_atoms: Number of atoms (including origin)
///     xi: Lattice step size
///     delta: Integer discretization factor
///     min_distance: Minimum allowed pairwise distance for filtering (e.g., 1.4 Angstroms)
///     max_iterations: Maximum iterations (0 for unlimited)
///     beam_width: Maximum size of open set (0 for unlimited, beam search)
///     dropout: Probability of dropping a neighbor (0.0 to 1.0). Dropped neighbors are excluded
///              from consideration for the rest of the search. Goal neighbors are never dropped.
///     greedy: If true, use greedy best-first search (f = h) instead of A* (f = g + h).
///             This ignores path cost and only considers heuristic distance to goal.
///     verbose: Print progress every 5 iterations
///
/// Returns:
///     Tuple of (path, iterations) where path is a list of flat Vec<i32> (vonorms + coords concatenated)
///     or None if no path found, and iterations is the number of iterations performed
#[pyfunction]
#[pyo3(signature = (start_points, goal_points, elements, n_atoms, xi, delta, min_distance, max_iterations, beam_width, dropout, greedy, verbose, speak_freq, heuristic_mode="manhattan", heuristic_weight=0.5, log_prefix=""))]
fn astar_pathfind_rust<'py>(
    _py: Python<'py>,
    start_points: Vec<(Vec<i32>, Vec<i32>)>,
    goal_points: Vec<(Vec<i32>, Vec<i32>)>,
    elements: Vec<String>,
    n_atoms: usize,
    xi: f64,
    delta: i32,
    min_distance: f64,
    max_iterations: usize,
    beam_width: usize,
    dropout: f64,
    greedy: bool,
    verbose: bool,
    speak_freq: usize,
    heuristic_mode: &str,
    heuristic_weight: f64,
    log_prefix: &str,
) -> PyResult<(Option<Vec<Vec<i32>>>, usize)> {
    use crate::pathfinding::astar_pathfind;
    use crate::heuristics::HeuristicMode;

    // Validate inputs
    if start_points.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "start_points cannot be empty"
        ));
    }
    if goal_points.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "goal_points cannot be empty"
        ));
    }

    let expected_coords_len = (n_atoms - 1) * 3;

    // Validate all start points
    for (i, (vonorms, coords)) in start_points.iter().enumerate() {
        if vonorms.len() != 7 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("start_points[{}] vonorms must have exactly 7 elements", i)
            ));
        }
        if coords.len() != expected_coords_len {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("start_points[{}] coords must have {} elements ((n_atoms-1) * 3)", i, expected_coords_len)
            ));
        }
    }

    // Validate all goal points
    for (i, (vonorms, coords)) in goal_points.iter().enumerate() {
        if vonorms.len() != 7 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("goal_points[{}] vonorms must have exactly 7 elements", i)
            ));
        }
        if coords.len() != expected_coords_len {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("goal_points[{}] coords must have {} elements ((n_atoms-1) * 3)", i, expected_coords_len)
            ));
        }
    }

    if elements.len() != n_atoms {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("elements must have {} elements (n_atoms)", n_atoms)
        ));
    }

    // Parse heuristic mode
    let mode = HeuristicMode::from_str(heuristic_mode);

    // Call the Rust pathfinding function
    let (path, iterations) = astar_pathfind(
        &start_points,
        &goal_points,
        &elements,
        n_atoms,
        xi,
        delta,
        min_distance,
        max_iterations,
        beam_width,
        dropout,
        greedy,
        verbose,
        speak_freq,
        mode,
        heuristic_weight,
        log_prefix,
    );

    // Check if we were interrupted - if so, raise KeyboardInterrupt
    use std::sync::atomic::Ordering as AtomicOrdering;
    if crate::pathfinding::WAS_INTERRUPTED.swap(false, AtomicOrdering::SeqCst) {
        return Err(pyo3::exceptions::PyKeyboardInterrupt::new_err("Search interrupted by user"));
    }

    Ok((path, iterations))
}

/// Bidirectional A* pathfinding from multiple start CNFs to multiple goal CNFs (pure Rust implementation)
///
/// Args:
///     start_points: List of (vonorms, coords) tuples for starting CNFs
///     goal_points: List of (vonorms, coords) tuples for goal CNFs
///     elements: List of element symbols for all atoms (including origin)
///     n_atoms: Number of atoms (including origin)
///     xi: Lattice step size
///     delta: Integer discretization factor
///     min_distance: Minimum allowed pairwise distance for filtering (e.g., 1.4 Angstroms)
///     max_iterations: Maximum iterations (0 for unlimited)
///     beam_width: Maximum size of each open set (0 for unlimited, beam search)
///     verbose: Print progress every 5 iterations
///
/// Returns:
///     List of flat Vec<i32> (vonorms + coords concatenated) representing the path, or None if no path found
#[pyfunction]
fn bidirectional_astar_pathfind_rust<'py>(
    _py: Python<'py>,
    start_points: Vec<(Vec<i32>, Vec<i32>)>,
    goal_points: Vec<(Vec<i32>, Vec<i32>)>,
    elements: Vec<String>,
    n_atoms: usize,
    xi: f64,
    delta: i32,
    min_distance: f64,
    max_iterations: usize,
    beam_width: usize,
    verbose: bool,
) -> PyResult<Option<Vec<Vec<i32>>>> {
    use crate::bidirectional::bidirectional_astar_pathfind;

    // Validate inputs (same as astar_pathfind_rust)
    if start_points.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "start_points cannot be empty"
        ));
    }
    if goal_points.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "goal_points cannot be empty"
        ));
    }

    let expected_coords_len = (n_atoms - 1) * 3;

    // Validate all start points
    for (i, (vonorms, coords)) in start_points.iter().enumerate() {
        if vonorms.len() != 7 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("start_points[{}] vonorms must have exactly 7 elements", i)
            ));
        }
        if coords.len() != expected_coords_len {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("start_points[{}] coords must have {} elements ((n_atoms-1) * 3)", i, expected_coords_len)
            ));
        }
    }

    // Validate all goal points
    for (i, (vonorms, coords)) in goal_points.iter().enumerate() {
        if vonorms.len() != 7 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("goal_points[{}] vonorms must have exactly 7 elements", i)
            ));
        }
        if coords.len() != expected_coords_len {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("goal_points[{}] coords must have {} elements ((n_atoms-1) * 3)", i, expected_coords_len)
            ));
        }
    }

    if elements.len() != n_atoms {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("elements must have {} elements (n_atoms)", n_atoms)
        ));
    }

    // Call the Rust bidirectional pathfinding function
    let result = bidirectional_astar_pathfind(
        &start_points,
        &goal_points,
        &elements,
        n_atoms,
        xi,
        delta,
        min_distance,
        max_iterations,
        beam_width,
        verbose,
    );

    Ok(result)
}

/// Filter neighbor tuples by minimum pairwise distance
///
/// Args:
///     neighbor_tuples: List of neighbor tuples where each tuple is vonorms + coords concatenated
///     n_atoms: Number of atoms (including origin)
///     xi: Lattice step size
///     delta: Integer discretization factor
///     min_distance: Minimum allowed pairwise distance in Angstroms
///
/// Returns:
///     Filtered list of neighbor tuples that pass the distance check
#[pyfunction]
fn filter_neighbors_by_min_distance_rust(
    neighbor_tuples: Vec<Vec<i32>>,
    n_atoms: usize,
    xi: f64,
    delta: i32,
    min_distance: f64,
) -> PyResult<Vec<Vec<i32>>> {
    // Split each neighbor tuple into (vonorms, coords)
    let split_tuples: Vec<(Vec<i32>, Vec<i32>)> = neighbor_tuples
        .iter()
        .map(|tuple| {
            let vonorms = tuple[..7].to_vec();
            let coords = tuple[7..].to_vec();
            (vonorms, coords)
        })
        .collect();

    // Filter using the existing Rust function
    let filtered = geometry::filter_neighbors_by_min_distance(
        &split_tuples,
        n_atoms,
        xi,
        delta,
        min_distance,
    );

    // Concatenate vonorms and coords back together
    let result: Vec<Vec<i32>> = filtered
        .into_iter()
        .map(|(vonorms, coords)| {
            let mut combined = vonorms;
            combined.extend(coords);
            combined
        })
        .collect();

    Ok(result)
}

#[pymodule]
fn rust_cnf(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    m.add_function(wrap_pyfunction!(find_and_canonicalize_lattice_neighbors, m)?)?;
    m.add_function(wrap_pyfunction!(validate_step_data_rust, m)?)?;
    m.add_function(wrap_pyfunction!(find_and_canonicalize_motif_neighbors, m)?)?;
    m.add_function(wrap_pyfunction!(find_neighbor_tuples_rust, m)?)?;
    m.add_function(wrap_pyfunction!(reconstruct_structure_from_cnf, m)?)?;
    m.add_function(wrap_pyfunction!(astar_pathfind_rust, m)?)?;
    m.add_function(wrap_pyfunction!(bidirectional_astar_pathfind_rust, m)?)?;
    m.add_function(wrap_pyfunction!(filter_neighbors_by_min_distance_rust, m)?)?;
    Ok(())
}
