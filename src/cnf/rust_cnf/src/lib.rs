mod permutations;
mod lnf;
mod mnf;

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

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

    // Find stabilizers for both inputs (already returns flat Vec<i32>)
    let s1_flat = lnf::find_stabilizers_raw(&vonorms1_fixed);
    let s2_flat = lnf::find_stabilizers_raw(&vonorms2_fixed);

    // Combine and deduplicate
    let combined_flat = lnf::combine_stabilizers(&s1_flat, &s2_flat, &middle);

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

    // Find stabilizers for both inputs using float version (tolerance-based)
    let s1_flat = lnf::find_stabilizers_raw_float(&vonorms1_fixed, tol);
    let s2_flat = lnf::find_stabilizers_raw_float(&vonorms2_fixed, tol);

    // Combine and deduplicate
    let combined_flat = lnf::combine_stabilizers(&s1_flat, &s2_flat, &middle);

    // Return as flat numpy array (Python will reshape to (N, 3, 3))
    let result_array = PyArray1::from_vec_bound(py, combined_flat);
    Ok(result_array.into())
}

/// Python module definition
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
    Ok(())
}
