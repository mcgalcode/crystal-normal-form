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
pub struct AStarNode {
    pub f_score: f64,
    pub g_score: f64,
    pub h_score: f64,
    pub vonorms: Vec<i32>,
    pub coords: Vec<i32>,
    pub counter: usize, // For tie-breaking
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
        // NOTE: We do NOT use counter for tie-breaking to match Python's behavior
        // where dataclass only compares f_score (other fields have compare=False)
        other.f_score.partial_cmp(&self.f_score)
            .unwrap_or(Ordering::Equal)
    }
}

/// Compute squared Euclidean distance heuristic between two CNF states
/// Uses sum of squared differences WITHOUT sqrt
/// This is inadmissible (overestimates) but works well in practice for CNF navigation
#[allow(dead_code)]
fn euclidean_heuristic(vonorms1: &[i32], coords1: &[i32], vonorms2: &[i32], coords2: &[i32]) -> f64 {
    assert_eq!(coords1.len(), coords2.len());
    assert_eq!(vonorms1.len(), vonorms2.len());

    // Distance in vonorm space
    let vonorm_dist_sq: f64 = vonorms1.iter()
        .zip(vonorms2.iter())
        .map(|(&v1, &v2)| {
            let diff = v1 as f64 - v2 as f64;
            diff * diff
        })
        .sum();

    // Distance in coord space
    let coord_dist_sq: f64 = coords1.iter()
        .zip(coords2.iter())
        .map(|(&c1, &c2)| {
            let diff = c1 as f64 - c2 as f64;
            diff * diff
        })
        .sum();

    // Combined SQUARED distance (no sqrt)
    vonorm_dist_sq + coord_dist_sq
}

/// Compute Manhattan distance (L1) heuristic between two CNF states
/// Uses sum of absolute differences, multiplied by 2
/// This matches the Python implementation's manhattan_distance heuristic
fn manhattan_heuristic(vonorms1: &[i32], coords1: &[i32], vonorms2: &[i32], coords2: &[i32]) -> f64 {
    assert_eq!(coords1.len(), coords2.len());
    assert_eq!(vonorms1.len(), vonorms2.len());

    // Manhattan distance in vonorm space
    let vonorm_dist: f64 = vonorms1.iter()
        .zip(vonorms2.iter())
        .map(|(&v1, &v2)| {
            (v1 - v2).abs() as f64
        })
        .sum();

    // Manhattan distance in coord space
    let coord_dist: f64 = coords1.iter()
        .zip(coords2.iter())
        .map(|(&c1, &c2)| {
            (c1 - c2).abs() as f64
        })
        .sum();

    (vonorm_dist + coord_dist) * 10.0
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

/// A* pathfinding from multiple start points to multiple goal CNFs
///
/// Args:
///     start_points: Vec of (vonorms, coords) tuples for starting CNFs
///     goal_points: Vec of (vonorms, coords) tuples for goal CNFs
///     elements: Atom element symbols
///     n_atoms: Number of atoms
///     xi: Lattice step size
///     delta: Integer discretization factor
///     min_distance: Minimum allowed distance for filtering (e.g., 1.4)
///     max_iterations: Maximum iterations (0 for unlimited)
///     beam_width: Maximum size of open set (0 for unlimited, beam search)
///     verbose: Print progress every 5 iterations
///
/// Returns:
///     Option containing path as Vec of flat Vec<i32> (vonorms + coords concatenated), or None if no path found
pub fn astar_pathfind(
    start_points: &[(Vec<i32>, Vec<i32>)],
    goal_points: &[(Vec<i32>, Vec<i32>)],
    elements: &[String],
    n_atoms: usize,
    xi: f64,
    delta: i32,
    min_distance: f64,
    max_iterations: usize,
    beam_width: usize,
    verbose: bool,
) -> Option<Vec<Vec<i32>>> {
    // Check if any start equals any goal
    for (start_vonorms, start_coords) in start_points {
        for (goal_vonorms, goal_coords) in goal_points {
            if states_equal(start_vonorms, start_coords, goal_vonorms, goal_coords) {
                // Return flat concatenated format
                let mut flat = start_vonorms.clone();
                flat.extend_from_slice(start_coords);
                return Some(vec![flat]);
            }
        }
    }

    // Priority queue (min-heap by f_score)
    let mut open_set = BinaryHeap::new();
    let mut counter = 0usize;

    // Initialize with all start points
    for (start_vonorms, start_coords) in start_points {
        // Compute minimum heuristic to any goal
        let h_start = goal_points.iter()
            .map(|(goal_vonorms, goal_coords)| manhattan_heuristic(start_vonorms, start_coords, goal_vonorms, goal_coords))
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(0.0);

        open_set.push(AStarNode {
            f_score: h_start,
            g_score: 0.0,
            h_score: h_start,
            vonorms: start_vonorms.clone(),
            coords: start_coords.clone(),
            counter,
        });
        counter += 1;
    }

    // Closed set: visited nodes
    let mut closed_set: HashSet<Vec<i32>> = HashSet::new();

    // Track best path: key -> parent key
    let mut came_from: HashMap<Vec<i32>, Vec<i32>> = HashMap::new();

    // Track g_scores: key -> cost from start
    let mut g_score: HashMap<Vec<i32>, f64> = HashMap::new();

    // Initialize g_scores for all start points
    for (start_vonorms, start_coords) in start_points {
        g_score.insert(make_key(start_vonorms, start_coords), 0.0);
    }

    let mut iterations = 0usize;
    let start_time = std::time::Instant::now();

    if verbose {
        eprintln!("Starting A* search with {} start states and {} goal states", start_points.len(), goal_points.len());
    }

    while let Some(current_node) = open_set.pop() {
        if max_iterations > 0 && iterations >= max_iterations {
            if verbose {
                eprintln!("Reached max iterations ({})", max_iterations);
            }
            return None;
        }

        iterations += 1;

        let current_key = make_key(&current_node.vonorms, &current_node.coords);

        // Verbose logging - match Python format
        if verbose && iterations % 5 == 0 {
            let elapsed = start_time.elapsed();
            eprintln!("Step {}:  open={}, closed={}, f={:.2}, g={:.2}, h={:.6}, Elapsed: {:.2}s",
                     iterations,
                     open_set.len() + 1,  // +1 because we just popped current_node
                     closed_set.len(),
                     current_node.f_score,
                     current_node.g_score,
                     current_node.h_score,
                     elapsed.as_secs_f64());
        }

        // Check if we reached any goal
        for (goal_vonorms, goal_coords) in goal_points.iter() {
            if states_equal(&current_node.vonorms, &current_node.coords, goal_vonorms, goal_coords) {
                if verbose {
                    eprintln!("\n✅ Found goal after {} iterations!", iterations);
                }
                return Some(reconstruct_path(
                    &came_from,
                    &current_node.vonorms,
                    &current_node.coords,
                ));
            }
        }

        // Skip if already in closed set
        if closed_set.contains(&current_key) {
            continue;
        }

        closed_set.insert(current_key.clone());

        // Get neighbors
        let neighbor_tuples = find_neighbor_tuples(
            &current_node.vonorms,
            &current_node.coords,
            elements,
            n_atoms,
            xi,
            delta,
        );

        if iterations == 1 && verbose {
            eprintln!("First iteration: found {} raw neighbors", neighbor_tuples.len());
        }

        // Filter by minimum distance
        let filtered_neighbors = filter_neighbors_by_min_distance(
            &neighbor_tuples,
            n_atoms,
            xi,
            delta,
            min_distance,
        );

        // Sort neighbors for deterministic behavior (avoids hash randomization issues)
        let mut filtered_neighbors = filtered_neighbors;
        filtered_neighbors.sort_by(|a, b| {
            a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1))
        });

        // Explore each neighbor
        for (neighbor_vonorms, neighbor_coords) in filtered_neighbors {
            let neighbor_key = make_key(&neighbor_vonorms, &neighbor_coords);

            if closed_set.contains(&neighbor_key) {
                continue;
            }

            // Edge cost: uniform cost 1
            let edge_cost = 1.0;
            let tentative_g = current_node.g_score + edge_cost;

            // Check if this is a better path
            let current_g = g_score.get(&neighbor_key).copied().unwrap_or(f64::INFINITY);

            if tentative_g < current_g {
                // Update path
                came_from.insert(neighbor_key.clone(), current_key.clone());
                g_score.insert(neighbor_key.clone(), tentative_g);

                // Calculate f_score using minimum heuristic to any goal
                let h = goal_points.iter()
                    .map(|(goal_vonorms, goal_coords)| manhattan_heuristic(&neighbor_vonorms, &neighbor_coords, goal_vonorms, goal_coords))
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .unwrap_or(0.0);
                let f = tentative_g + h;

                // Add to open set
                open_set.push(AStarNode {
                    f_score: f,
                    g_score: tentative_g,
                    h_score: h,
                    vonorms: neighbor_vonorms,
                    coords: neighbor_coords,
                    counter,
                });
                counter += 1;
            }
        }

        // Beam search: prune open set if it exceeds beam_width
        if beam_width > 0 && open_set.len() > beam_width {
            // Convert heap to sorted vec, keep only beam_width best nodes
            let mut nodes: Vec<_> = open_set.into_iter().collect();
            nodes.sort_by(|a, b| {
                a.f_score.partial_cmp(&b.f_score)
                    .unwrap_or(Ordering::Equal)
            });
            nodes.truncate(beam_width);

            // Rebuild heap
            open_set = BinaryHeap::from(nodes);
        }
    }

    // No path found
    if verbose {
        eprintln!("\n❌ No path found after {} iterations", iterations);
    }
    None
}

/// Reconstruct the path from start to goal using the came_from map
/// Returns path as flat concatenated vectors (vonorms + coords)
fn reconstruct_path(
    came_from: &HashMap<Vec<i32>, Vec<i32>>,
    goal_vonorms: &[i32],
    goal_coords: &[i32],
) -> Vec<Vec<i32>> {
    let mut path = Vec::new();
    let mut current_key = make_key(goal_vonorms, goal_coords);

    // Add current key (already concatenated)
    path.push(current_key.clone());

    while let Some(parent_key) = came_from.get(&current_key) {
        path.push(parent_key.clone());
        current_key = parent_key.clone();
    }

    path.reverse();
    path
}
