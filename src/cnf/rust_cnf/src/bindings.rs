/// Python bindings for CNF operations (PyO3)
///
/// This module contains all #[pyfunction] wrappers that expose Rust functionality to Python.

use pyo3::prelude::*;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};

use crate::linalg::{flat9_to_mat3x3, slice_to_vonorms, flat_matrices_to_nested};
use crate::neighbors::{compute_step_data_raw_internal, canonicalize_cnfs_batch_internal};
use crate::{lnf, mnf, geometry};

// =============================================================================
// Helper functions for PyO3 type conversions
// =============================================================================

/// Validate and convert a vonorms slice to a fixed-size array
#[inline]
fn validate_and_convert_vonorms(slice: &[f64]) -> PyResult<[f64; 7]> {
    if slice.len() != 7 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "vonorms must have exactly 7 elements"
        ));
    }
    Ok(slice_to_vonorms(slice))
}

/// Validate and convert a middle matrix from flat array
#[inline]
fn validate_and_convert_middle(slice: &[i32]) -> PyResult<[[i32; 3]; 3]> {
    if slice.len() != 9 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "middle matrix must have exactly 9 elements (3x3)"
        ));
    }
    Ok(flat9_to_mat3x3(slice))
}

// =============================================================================
// LNF (Lattice Normal Form) Bindings
// =============================================================================

/// Rust implementation of build_lnf_raw for discretized vonorms (exact equality)
/// Returns: (canonical_vonorms, zero_idxs, selling_transform_flat, sorting_matrices)
#[pyfunction]
pub fn build_lnf_raw_rust<'py>(
    py: Python<'py>,
    vonorms: PyReadonlyArray1<f64>,
) -> PyResult<(Py<PyArray1<f64>>, Vec<usize>, Option<Vec<i32>>, Vec<Vec<i32>>)> {
    let vonorms_slice = vonorms.as_slice()?;
    let vonorms_fixed = validate_and_convert_vonorms(vonorms_slice)?;

    let (canonical, zero_idxs, selling_flat, sorting_mats) = lnf::build_lnf_raw_discretized(&vonorms_fixed);

    let canonical_array = PyArray1::from_slice_bound(py, &canonical);
    Ok((canonical_array.into(), zero_idxs, selling_flat, sorting_mats))
}

/// Rust implementation of build_lnf_raw for float vonorms (tolerance-based)
/// Returns: (canonical_vonorms, zero_idxs, selling_transform_flat, sorting_matrices)
#[pyfunction]
pub fn build_lnf_raw_float_rust<'py>(
    py: Python<'py>,
    vonorms: PyReadonlyArray1<f64>,
    tol: f64,
) -> PyResult<(Py<PyArray1<f64>>, Vec<usize>, Option<Vec<i32>>, Vec<Vec<i32>>)> {
    let vonorms_slice = vonorms.as_slice()?;
    let vonorms_fixed = validate_and_convert_vonorms(vonorms_slice)?;

    let (canonical, zero_idxs, selling_flat, sorting_mats) = lnf::build_lnf_raw_float(&vonorms_fixed, tol);

    let canonical_array = PyArray1::from_slice_bound(py, &canonical);
    Ok((canonical_array.into(), zero_idxs, selling_flat, sorting_mats))
}

// =============================================================================
// Stabilizer Bindings
// =============================================================================

/// Rust implementation of stabilizer finding (exact equality for discretized)
#[pyfunction]
pub fn find_stabilizers_rust<'py>(
    py: Python<'py>,
    vonorms: PyReadonlyArray1<f64>,
) -> PyResult<Py<pyo3::PyAny>> {
    let vonorms_slice = vonorms.as_slice()?;
    let vonorms_fixed = validate_and_convert_vonorms(vonorms_slice)?;

    let stabilizers_flat = lnf::find_stabilizers_raw(&vonorms_fixed);
    let matrices = flat_matrices_to_nested(&stabilizers_flat);
    Ok(pyo3::types::PyList::new_bound(py, matrices).into())
}

/// Rust implementation of stabilizer finding for float vonorms (tolerance-based)
#[pyfunction]
pub fn find_stabilizers_rust_float<'py>(
    py: Python<'py>,
    vonorms: PyReadonlyArray1<f64>,
    tol: f64,
) -> PyResult<Py<pyo3::PyAny>> {
    let vonorms_slice = vonorms.as_slice()?;
    let vonorms_fixed = validate_and_convert_vonorms(vonorms_slice)?;

    let stabilizers_flat = lnf::find_stabilizers_raw_float(&vonorms_fixed, tol);
    let matrices = flat_matrices_to_nested(&stabilizers_flat);
    Ok(pyo3::types::PyList::new_bound(py, matrices).into())
}

/// Combine and deduplicate stabilizer matrices (Rust implementation)
#[pyfunction]
pub fn combine_stabilizers_rust<'py>(
    py: Python<'py>,
    s1_flat: PyReadonlyArray1<i32>,
    s2_flat: PyReadonlyArray1<i32>,
    middle_2d: PyReadonlyArray1<i32>,
) -> PyResult<Py<pyo3::PyAny>> {
    let middle_slice = middle_2d.as_slice()?;
    let middle = validate_and_convert_middle(middle_slice)?;

    let result_flat = lnf::combine_stabilizers(
        s1_flat.as_slice()?,
        s2_flat.as_slice()?,
        &middle,
    );

    let matrices = flat_matrices_to_nested(&result_flat);
    Ok(pyo3::types::PyList::new_bound(py, matrices).into())
}

/// Combined pipeline: find stabilizers for both vonorm arrays and combine them
#[pyfunction]
pub fn find_and_combine_stabilizers_rust<'py>(
    py: Python<'py>,
    vonorms1: PyReadonlyArray1<f64>,
    vonorms2: PyReadonlyArray1<f64>,
    middle_flat: PyReadonlyArray1<i32>,
) -> PyResult<Py<PyArray1<i32>>> {
    let vonorms1_fixed = validate_and_convert_vonorms(vonorms1.as_slice()?)?;
    let vonorms2_fixed = validate_and_convert_vonorms(vonorms2.as_slice()?)?;
    let middle = validate_and_convert_middle(middle_flat.as_slice()?)?;

    let s1_flat = lnf::find_stabilizers_raw(&vonorms1_fixed);
    let s2_flat = lnf::find_stabilizers_raw(&vonorms2_fixed);
    let combined_flat = lnf::combine_stabilizers(&s1_flat, &s2_flat, &middle);

    let result_array = PyArray1::from_vec_bound(py, combined_flat);
    Ok(result_array.into())
}

/// Combined pipeline for float vonorms (tolerance-based)
#[pyfunction]
pub fn find_and_combine_stabilizers_rust_float<'py>(
    py: Python<'py>,
    vonorms1: PyReadonlyArray1<f64>,
    vonorms2: PyReadonlyArray1<f64>,
    middle_flat: PyReadonlyArray1<i32>,
    tol: f64,
) -> PyResult<Py<PyArray1<i32>>> {
    let vonorms1_fixed = validate_and_convert_vonorms(vonorms1.as_slice()?)?;
    let vonorms2_fixed = validate_and_convert_vonorms(vonorms2.as_slice()?)?;
    let middle = validate_and_convert_middle(middle_flat.as_slice()?)?;

    let s1_flat = lnf::find_stabilizers_raw_float(&vonorms1_fixed, tol);
    let s2_flat = lnf::find_stabilizers_raw_float(&vonorms2_fixed, tol);
    let combined_flat = lnf::combine_stabilizers(&s1_flat, &s2_flat, &middle);

    let result_array = PyArray1::from_vec_bound(py, combined_flat);
    Ok(result_array.into())
}

// =============================================================================
// MNF (Motif Normal Form) Bindings
// =============================================================================

/// Build MNF (Motif Normal Form) using vectorized algorithm
#[pyfunction]
pub fn build_mnf_vectorized_rust<'py>(
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
    let labels_usize: Vec<usize> = labels_arr.iter().map(|&x| x as usize).collect();

    let mnf_result = mnf::build_mnf_vectorized(
        coords_arr.as_slice().unwrap(),
        n_atoms,
        &labels_usize,
        num_origin_atoms,
        stabs_arr.as_slice().unwrap(),
        mod_val,
    );

    let mnf_array = PyArray1::from_slice_bound(py, &mnf_result);
    Ok(mnf_array.into())
}

/// Build MNFs for many coordinate sets in batch
#[pyfunction]
pub fn build_mnf_batch_rust<'py>(
    py: Python<'py>,
    coords_batch: Vec<Vec<f64>>,
    atom_labels: PyReadonlyArray1<i32>,
    num_origin_atoms: usize,
    stabilizers_flat: PyReadonlyArray1<i32>,
    mod_val: f64,
) -> PyResult<Py<pyo3::PyAny>> {
    let labels_arr = atom_labels.as_array();
    let stabs_arr = stabilizers_flat.as_array();

    let n_atoms = labels_arr.len();
    let labels_usize: Vec<usize> = labels_arr.iter().map(|&x| x as usize).collect();

    let mnf_results = mnf::build_mnf_batch(
        &coords_batch,
        n_atoms,
        &labels_usize,
        num_origin_atoms,
        stabs_arr.as_slice().unwrap(),
        mod_val,
    );

    let mnf_arrays: Vec<_> = mnf_results
        .into_iter()
        .map(|mnf| PyArray1::from_vec_bound(py, mnf).into_py(py))
        .collect();

    Ok(pyo3::types::PyList::new_bound(py, mnf_arrays).into())
}

/// Find and canonicalize motif neighbors
#[pyfunction]
pub fn find_and_canonicalize_motif_neighbors<'py>(
    py: Python<'py>,
    motif_coords: Vec<i32>,
    atoms: Vec<String>,
    stabilizers_flat: PyReadonlyArray1<i32>,
    delta: i32,
) -> PyResult<Py<pyo3::types::PyList>> {
    use pyo3::types::{PyList, PyTuple};

    let stabs_slice = stabilizers_flat.as_slice()?;

    let results = crate::neighbors::find_and_canonicalize_motif_neighbors(
        &motif_coords,
        &atoms,
        stabs_slice,
        delta,
    );

    let py_results = PyList::empty_bound(py);
    for result in results {
        let coords_tuple = PyTuple::new_bound(py, &result);
        py_results.append(coords_tuple)?;
    }

    Ok(py_results.unbind())
}

// =============================================================================
// Neighbor Finding Bindings
// =============================================================================

/// Compute step data for lattice neighbor finding
#[pyfunction]
pub fn compute_step_data_raw_rust<'py>(
    py: Python<'py>,
    input_vonorms: PyReadonlyArray1<f64>,
    motif_coord_matrix: PyReadonlyArray1<f64>,
    n_atoms: usize,
    motif_delta: i32,
) -> PyResult<Py<pyo3::PyAny>> {
    let vonorms_arr = validate_and_convert_vonorms(input_vonorms.as_slice()?)?;

    let result_values = compute_step_data_raw_internal(
        &vonorms_arr,
        motif_coord_matrix.as_slice()?,
        n_atoms,
        motif_delta,
    );

    Ok(pyo3::types::PyList::new_bound(py, result_values).into())
}

/// Validate step data by filtering steps that pass vonorm validation
#[pyfunction]
pub fn validate_step_data_rust<'py>(
    py: Python<'py>,
    step_data: Py<pyo3::PyAny>,
) -> PyResult<Bound<'py, pyo3::types::PyList>> {
    use pyo3::types::{PyList, PyTuple};

    let step_list = step_data.downcast_bound::<PyList>(py)?;
    let validated_steps = PyList::empty_bound(py);

    for item in step_list.iter() {
        let tuple = item.downcast::<PyTuple>()?;

        let vonorms_list: Vec<f64> = tuple.get_item(1)?.extract()?;
        let mut step_vonorms = [0.0; 7];
        for (i, &v) in vonorms_list.iter().enumerate().take(7) {
            step_vonorms[i] = v;
        }

        if !lnf::has_valid_conorms_exact(&step_vonorms) {
            continue;
        }
        if !lnf::is_obtuse(&step_vonorms) || !lnf::is_superbasis_exact(&step_vonorms) {
            continue;
        }

        validated_steps.append(item)?;
    }

    Ok(validated_steps)
}

/// Build CNFs from validated step data (non-canonical)
#[pyfunction]
pub fn build_cnfs_from_step_data_rust<'py>(
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

    for item in step_data_list.iter() {
        let tuple: &Bound<'_, PyTuple> = item.downcast()?;

        let vonorms_list: Vec<f64> = tuple.get_item(1)?.extract()?;

        let coords_obj = tuple.get_item(2)?;
        let coords_list: Vec<f64> = if let Ok(arr) = coords_obj.downcast::<PyArray1<f64>>() {
            arr.readonly().as_slice()?.to_vec()
        } else if let Ok(arr2d) = coords_obj.downcast::<numpy::PyArray2<f64>>() {
            arr2d.readonly().as_array().iter().copied().collect()
        } else {
            coords_obj.extract()?
        };

        result_list.append((
            vonorms_list,
            coords_list,
            atoms.clone()
        ))?;
    }

    Ok(result_list.into())
}

/// Batch CNF canonicalization
#[pyfunction]
pub fn canonicalize_cnfs_batch_rust<'py>(
    py: Python<'py>,
    step_data_list: &Bound<'py, pyo3::types::PyList>,
    atoms: Vec<String>,
    _xi: f64,
    delta: i32,
) -> PyResult<Py<pyo3::types::PyList>> {
    use pyo3::types::{PyList, PyTuple};

    let mut step_data = Vec::new();
    for item in step_data_list.iter() {
        let tuple: &Bound<'_, PyTuple> = item.downcast()?;

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

    let results = canonicalize_cnfs_batch_internal(step_data, &atoms, delta);

    let result_list = PyList::empty_bound(py);
    for (vonorms_ints, coords_ints) in results {
        let vonorms_tuple = PyTuple::new_bound(py, &vonorms_ints);
        let coords_tuple = PyTuple::new_bound(py, &coords_ints);
        result_list.append((vonorms_tuple, coords_tuple))?;
    }

    Ok(result_list.into())
}

/// Combined pipeline for lattice neighbor finding
#[pyfunction]
pub fn find_and_canonicalize_lattice_neighbors<'py>(
    py: Python<'py>,
    input_vonorms: PyReadonlyArray1<f64>,
    motif_coords_flat: PyReadonlyArray1<f64>,
    n_atoms: usize,
    motif_delta: i32,
    atoms: Vec<String>,
    xi: f64,
    delta: i32,
) -> PyResult<Py<pyo3::types::PyList>> {
    let step_data = compute_step_data_raw_rust(
        py,
        input_vonorms,
        motif_coords_flat,
        n_atoms,
        motif_delta
    )?;

    let validated_steps = validate_step_data_rust(py, step_data)?;

    canonicalize_cnfs_batch_rust(py, &validated_steps, atoms, xi, delta)
}

/// Find all neighbor tuples (lattice + motif) for a given CNF state
#[pyfunction]
pub fn find_neighbor_tuples_rust<'py>(
    py: Python<'py>,
    vonorms_i32: PyReadonlyArray1<i32>,
    coords_i32: PyReadonlyArray1<i32>,
    elements: Vec<String>,
    n_atoms: usize,
    xi: f64,
    delta: i32,
) -> PyResult<Py<pyo3::types::PyList>> {
    use pyo3::types::{PyList, PyTuple};

    let vonorms_slice = vonorms_i32.as_slice()?;
    let coords_slice = coords_i32.as_slice()?;

    let results = crate::neighbors::find_neighbor_tuples(
        vonorms_slice,
        coords_slice,
        &elements,
        n_atoms,
        xi,
        delta,
    );

    let py_results = PyList::empty_bound(py);
    for (vonorms, coords) in results {
        let vonorms_tuple = PyTuple::new_bound(py, &vonorms);
        let coords_tuple = PyTuple::new_bound(py, &coords);
        py_results.append((vonorms_tuple, coords_tuple))?;
    }

    Ok(py_results.into())
}

// =============================================================================
// S4 Representatives Binding
// =============================================================================

/// Get S4 maximal representatives for vonorms
#[pyfunction]
pub fn get_s4_maximal_representatives_rust<'py>(
    py: Python<'py>,
    vonorms: PyReadonlyArray1<f64>,
) -> PyResult<Py<pyo3::PyAny>> {
    use crate::permutations::{compute_conorms, find_zero_indices_exact, get_s4_representatives};
    use pyo3::types::{PyDict, PyList};

    let vonorms_fixed = validate_and_convert_vonorms(vonorms.as_slice()?)?;

    let conorms = compute_conorms(&vonorms_fixed);
    let zero_indices = find_zero_indices_exact(&conorms);
    let s4_reps = get_s4_representatives(&vonorms_fixed, &zero_indices);

    let result_list = PyList::empty_bound(py);

    for rep in s4_reps {
        let dict = PyDict::new_bound(py);

        let s4_key_tuple = PyList::empty_bound(py);
        for &idx in &rep.s4_key {
            s4_key_tuple.append(idx)?;
        }
        dict.set_item("s4_key", s4_key_tuple)?;

        let vonorms_tuple = PyList::empty_bound(py);
        for &v in &rep.permuted_vonorms {
            vonorms_tuple.append(v)?;
        }
        dict.set_item("maximal_permuted_vonorms", vonorms_tuple)?;

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

// =============================================================================
// Geometry and Structure Reconstruction
// =============================================================================

/// Reconstruct structure from CNF coordinates
#[pyfunction]
pub fn reconstruct_structure_from_cnf<'py>(
    _py: Python<'py>,
    vonorms: PyReadonlyArray1<f64>,
    coords: PyReadonlyArray1<i32>,
    n_atoms: usize,
    xi: f64,
    delta: i32,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    use crate::geometry::{vonorms_to_lattice_matrix, coords_to_cartesian_positions};

    let vonorms_arr = validate_and_convert_vonorms(vonorms.as_slice()?)?;
    let coords_slice = coords.as_slice()?;

    let lattice_matrix = vonorms_to_lattice_matrix(&vonorms_arr, xi);
    let cart_coords = coords_to_cartesian_positions(coords_slice, n_atoms, delta, &lattice_matrix);

    let lattice_flat: Vec<f64> = lattice_matrix.iter()
        .flat_map(|row| row.iter().copied())
        .collect();

    Ok((lattice_flat, cart_coords))
}

/// Filter neighbor tuples by minimum pairwise distance
#[pyfunction]
pub fn filter_neighbors_by_min_distance_rust(
    neighbor_tuples: Vec<Vec<i32>>,
    n_atoms: usize,
    xi: f64,
    delta: i32,
    min_distance: f64,
) -> PyResult<Vec<Vec<i32>>> {
    let split_tuples: Vec<(Vec<i32>, Vec<i32>)> = neighbor_tuples
        .iter()
        .map(|tuple| {
            let vonorms = tuple[..7].to_vec();
            let coords = tuple[7..].to_vec();
            (vonorms, coords)
        })
        .collect();

    let filtered = geometry::filter_neighbors_by_min_distance(
        &split_tuples,
        n_atoms,
        xi,
        delta,
        min_distance,
    );

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

// =============================================================================
// A* Pathfinding Bindings
// =============================================================================

/// Validate A* pathfinding inputs
fn validate_pathfind_inputs(
    start_points: &[(Vec<i32>, Vec<i32>)],
    goal_points: &[(Vec<i32>, Vec<i32>)],
    elements: &[String],
    n_atoms: usize,
) -> PyResult<()> {
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

    Ok(())
}

/// A* pathfinding from multiple start CNFs to multiple goal CNFs
#[pyfunction]
#[pyo3(signature = (start_points, goal_points, elements, n_atoms, xi, delta, min_distance, max_iterations, beam_width, dropout, greedy, verbose, speak_freq, heuristic_mode="manhattan", heuristic_weight=0.5, log_prefix=""))]
pub fn astar_pathfind_rust<'py>(
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

    validate_pathfind_inputs(&start_points, &goal_points, &elements, n_atoms)?;

    let mode = HeuristicMode::from_str(heuristic_mode);

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

    use std::sync::atomic::Ordering as AtomicOrdering;
    if crate::pathfinding::WAS_INTERRUPTED.swap(false, AtomicOrdering::SeqCst) {
        return Err(pyo3::exceptions::PyKeyboardInterrupt::new_err("Search interrupted by user"));
    }

    Ok((path, iterations))
}
