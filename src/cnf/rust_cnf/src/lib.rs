//! Rust CNF (Crystal Normal Form) Library
//!
//! This crate provides high-performance implementations of CNF operations
//! for crystallographic structure analysis and pathfinding.

mod linalg;
mod permutations;
mod selling;
mod stabilizers;
mod lnf;
mod mnf;
mod geometry;
mod pathfinding;
mod neighbors;
mod heuristics;
mod bindings;

use pyo3::prelude::*;

// Re-export binding functions for the Python module
use bindings::{
    build_lnf_raw_rust,
    build_lnf_raw_float_rust,
    find_stabilizers_rust,
    find_stabilizers_rust_float,
    combine_stabilizers_rust,
    find_and_combine_stabilizers_rust,
    find_and_combine_stabilizers_rust_float,
    build_mnf_vectorized_rust,
    build_mnf_batch_rust,
    find_and_canonicalize_motif_neighbors,
    compute_step_data_raw_rust,
    validate_step_data_rust,
    build_cnfs_from_step_data_rust,
    canonicalize_cnfs_batch_rust,
    find_and_canonicalize_lattice_neighbors,
    find_neighbor_tuples_rust,
    get_s4_maximal_representatives_rust,
    reconstruct_structure_from_cnf,
    filter_neighbors_by_min_distance_rust,
    astar_pathfind_rust,
};

#[pymodule]
fn rust_cnf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // LNF functions
    m.add_function(wrap_pyfunction!(build_lnf_raw_rust, m)?)?;
    m.add_function(wrap_pyfunction!(build_lnf_raw_float_rust, m)?)?;

    // Stabilizer functions
    m.add_function(wrap_pyfunction!(find_stabilizers_rust, m)?)?;
    m.add_function(wrap_pyfunction!(find_stabilizers_rust_float, m)?)?;
    m.add_function(wrap_pyfunction!(combine_stabilizers_rust, m)?)?;
    m.add_function(wrap_pyfunction!(find_and_combine_stabilizers_rust, m)?)?;
    m.add_function(wrap_pyfunction!(find_and_combine_stabilizers_rust_float, m)?)?;

    // MNF functions
    m.add_function(wrap_pyfunction!(build_mnf_vectorized_rust, m)?)?;
    m.add_function(wrap_pyfunction!(build_mnf_batch_rust, m)?)?;
    m.add_function(wrap_pyfunction!(find_and_canonicalize_motif_neighbors, m)?)?;

    // Neighbor finding functions
    m.add_function(wrap_pyfunction!(compute_step_data_raw_rust, m)?)?;
    m.add_function(wrap_pyfunction!(build_cnfs_from_step_data_rust, m)?)?;
    m.add_function(wrap_pyfunction!(canonicalize_cnfs_batch_rust, m)?)?;
    m.add_function(wrap_pyfunction!(find_and_canonicalize_lattice_neighbors, m)?)?;
    m.add_function(wrap_pyfunction!(validate_step_data_rust, m)?)?;
    m.add_function(wrap_pyfunction!(find_neighbor_tuples_rust, m)?)?;

    // S4 representatives
    m.add_function(wrap_pyfunction!(get_s4_maximal_representatives_rust, m)?)?;

    // Structure reconstruction and filtering
    m.add_function(wrap_pyfunction!(reconstruct_structure_from_cnf, m)?)?;
    m.add_function(wrap_pyfunction!(filter_neighbors_by_min_distance_rust, m)?)?;

    // Pathfinding
    m.add_function(wrap_pyfunction!(astar_pathfind_rust, m)?)?;

    Ok(())
}
