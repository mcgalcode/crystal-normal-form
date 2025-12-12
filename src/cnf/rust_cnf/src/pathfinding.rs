/// A* pathfinding implementation for CNF navigation
///
/// This module provides an A* pathfinding algorithm that works entirely in Rust,
/// allowing efficient pathfinding through the CNF neighbor graph.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

use crate::neighbors::find_neighbor_tuples;
use crate::geometry::filter_neighbors_by_min_distance;

/// A node in the A* search, ordered by f_score (lower is better)
#[derive(Clone)]
struct AStarNode {
    f_score: f64,
    g_score: f64,
    vonorms: Vec<i32>,
    coords: Vec<i32>,
    counter: usize, // For tie-breaking
}

impl PartialEq for AStarNode {
    fn eq(&self, other: &Self) -> bool {
        self.f_score == other.f_score && self.counter == other.counter
    }
}

impl Eq for AStarNode {}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering: smaller f_score has higher priority
        other.f_score.partial_cmp(&self.f_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.counter.cmp(&self.counter))
    }
}

/// Compute Euclidean distance heuristic between two CNF states
fn euclidean_heuristic(coords1: &[i32], coords2: &[i32]) -> f64 {
    assert_eq!(coords1.len(), coords2.len());

    let sum_squares: f64 = coords1.iter()
        .zip(coords2.iter())
        .map(|(&c1, &c2)| {
            let diff = c1 as f64 - c2 as f64;
            diff * diff
        })
        .sum();

    sum_squares.sqrt()
}

/// Create a hash key from vonorms and coords for deduplication
fn make_key(vonorms: &[i32], coords: &[i32]) -> Vec<i32> {
    let mut key = Vec::with_capacity(vonorms.len() + coords.len());
    key.extend_from_slice(vonorms);
    key.extend_from_slice(coords);
    key
}

/// Check if two CNF states are equal
fn states_equal(vonorms1: &[i32], coords1: &[i32], vonorms2: &[i32], coords2: &[i32]) -> bool {
    vonorms1 == vonorms2 && coords1 == coords2
}

/// A* pathfinding from start to goal CNF
///
/// Args:
///     start_vonorms: Starting CNF vonorms
///     start_coords: Starting CNF coords
///     goal_vonorms: Goal CNF vonorms
///     goal_coords: Goal CNF coords
///     n_atoms: Number of atoms
///     xi: Lattice step size
///     delta: Integer discretization factor
///     min_distance: Minimum allowed distance for filtering (e.g., 1.4)
///     max_iterations: Maximum iterations (0 for unlimited)
///     verbose: Print progress every 100 iterations
///
/// Returns:
///     Option containing path as Vec of (vonorms, coords) tuples, or None if no path found
pub fn astar_pathfind(
    start_vonorms: &[i32],
    start_coords: &[i32],
    goal_vonorms: &[i32],
    goal_coords: &[i32],
    n_atoms: usize,
    xi: f64,
    delta: i32,
    min_distance: f64,
    max_iterations: usize,
    verbose: bool,
) -> Option<Vec<(Vec<i32>, Vec<i32>)>> {
    // Check if start equals goal
    if states_equal(start_vonorms, start_coords, goal_vonorms, goal_coords) {
        return Some(vec![(start_vonorms.to_vec(), start_coords.to_vec())]);
    }

    // Priority queue (min-heap by f_score)
    let mut open_set = BinaryHeap::new();
    let mut counter = 0usize;

    // Initial heuristic
    let h_start = euclidean_heuristic(start_coords, goal_coords);
    open_set.push(AStarNode {
        f_score: h_start,
        g_score: 0.0,
        vonorms: start_vonorms.to_vec(),
        coords: start_coords.to_vec(),
        counter,
    });
    counter += 1;

    // Closed set: visited nodes
    let mut closed_set: HashSet<Vec<i32>> = HashSet::new();

    // Track best path: key -> parent key
    let mut came_from: HashMap<Vec<i32>, Vec<i32>> = HashMap::new();

    // Track g_scores: key -> cost from start
    let mut g_score: HashMap<Vec<i32>, f64> = HashMap::new();
    g_score.insert(make_key(start_vonorms, start_coords), 0.0);

    let mut iterations = 0usize;
    let start_time = std::time::Instant::now();

    while let Some(current_node) = open_set.pop() {
        if max_iterations > 0 && iterations >= max_iterations {
            eprintln!("Reached max iterations: {}", max_iterations);
            return None;
        }

        iterations += 1;

        let current_key = make_key(&current_node.vonorms, &current_node.coords);

        // Verbose logging
        if verbose && iterations % 100 == 0 {
            let elapsed = start_time.elapsed();
            eprintln!("Iteration {}: closed={}, g={:.3}, h={:.3}, f={:.3}, time={:.2}s",
                     iterations,
                     closed_set.len(),
                     current_node.g_score,
                     current_node.f_score - current_node.g_score,
                     current_node.f_score,
                     elapsed.as_secs_f64());
        }

        // Check if we reached the goal
        if states_equal(&current_node.vonorms, &current_node.coords, goal_vonorms, goal_coords) {
            return Some(reconstruct_path(
                &came_from,
                &current_node.vonorms,
                &current_node.coords,
            ));
        }

        // Skip if already in closed set
        if closed_set.contains(&current_key) {
            continue;
        }

        closed_set.insert(current_key.clone());

        // Get neighbors
        // TODO: Update to use new find_neighbor_tuples signature
        // let vonorms_f64: Vec<f64> = current_node.vonorms.iter().map(|&v| v as f64).collect();
        // let neighbor_tuples = find_neighbor_tuples(&current_node.vonorms, &current_node.coords, ...);
        let neighbor_tuples = vec![]; // Placeholder

        // Filter by minimum distance
        let filtered_neighbors = filter_neighbors_by_min_distance(
            &neighbor_tuples,
            n_atoms,
            xi,
            delta,
            min_distance,
        );

        // Explore each neighbor
        for (neighbor_vonorms, neighbor_coords) in filtered_neighbors {
            let neighbor_key = make_key(&neighbor_vonorms, &neighbor_coords);

            if closed_set.contains(&neighbor_key) {
                continue;
            }

            // Edge cost: use heuristic as distance measure
            let edge_cost = euclidean_heuristic(&current_node.coords, &neighbor_coords);
            let tentative_g = current_node.g_score + edge_cost;

            // Check if this is a better path
            let current_g = g_score.get(&neighbor_key).copied().unwrap_or(f64::INFINITY);

            if tentative_g < current_g {
                // Update path
                came_from.insert(neighbor_key.clone(), current_key.clone());
                g_score.insert(neighbor_key.clone(), tentative_g);

                // Calculate f_score
                let h = euclidean_heuristic(&neighbor_coords, goal_coords);
                let f = tentative_g + h;

                // Add to open set
                open_set.push(AStarNode {
                    f_score: f,
                    g_score: tentative_g,
                    vonorms: neighbor_vonorms,
                    coords: neighbor_coords,
                    counter,
                });
                counter += 1;
            }
        }
    }

    // No path found
    None
}

/// Reconstruct the path from start to goal using the came_from map
fn reconstruct_path(
    came_from: &HashMap<Vec<i32>, Vec<i32>>,
    goal_vonorms: &[i32],
    goal_coords: &[i32],
) -> Vec<(Vec<i32>, Vec<i32>)> {
    let mut path = Vec::new();
    let mut current_key = make_key(goal_vonorms, goal_coords);

    // Extract vonorms and coords from the current key
    let vonorm_len = 7; // CNFs always have 7 vonorms
    let (vonorms, coords) = current_key.split_at(vonorm_len);
    path.push((vonorms.to_vec(), coords.to_vec()));

    while let Some(parent_key) = came_from.get(&current_key) {
        let (vonorms, coords) = parent_key.split_at(vonorm_len);
        path.push((vonorms.to_vec(), coords.to_vec()));
        current_key = parent_key.clone();
    }

    path.reverse();
    path
}
