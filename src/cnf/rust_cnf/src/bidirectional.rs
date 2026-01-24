/// Bidirectional A* pathfinding implementation for CNF navigation

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

use crate::pathfinding::AStarNode;
use crate::neighbors::find_neighbor_tuples;
use crate::geometry::filter_neighbors_by_min_distance;

/// Compute Manhattan distance (L1) heuristic between two CNF states
/// Uses sum of absolute differences, multiplied by 10
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

    // Combined Manhattan distance, multiplied by 10 (matches Python implementation)
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

/// Reconstruct bidirectional path from meeting point
fn reconstruct_bidirectional_path(
    meeting_point: &[i32],
    forward_came_from: &HashMap<Vec<i32>, Vec<i32>>,
    backward_came_from: &HashMap<Vec<i32>, Vec<i32>>,
) -> Vec<Vec<i32>> {
    // Build forward path (from start to meeting point)
    let mut forward_path = Vec::new();
    let mut current = meeting_point.to_vec();

    forward_path.push(current.clone());
    while let Some(parent) = forward_came_from.get(&current) {
        forward_path.push(parent.clone());
        current = parent.clone();
    }
    forward_path.reverse();

    // Build backward path (from meeting point to goal)
    let mut backward_path = Vec::new();
    current = meeting_point.to_vec();

    while let Some(parent) = backward_came_from.get(&current) {
        backward_path.push(parent.clone());
        current = parent.clone();
    }

    // Combine paths
    let mut complete_path = forward_path;
    complete_path.extend(backward_path);
    complete_path
}

/// Bidirectional A* pathfinding - searches from both start and goal simultaneously
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
///     beam_width: Maximum size of each open set (0 for unlimited, beam search)
///     verbose: Print progress every 5 iterations
///
/// Returns:
///     Option containing path as Vec of flat Vec<i32> (vonorms + coords concatenated), or None if no path found
pub fn bidirectional_astar_pathfind(
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
                let mut flat = start_vonorms.clone();
                flat.extend_from_slice(start_coords);
                return Some(vec![flat]);
            }
        }
    }

    // Forward search structures
    let mut forward_open = BinaryHeap::new();
    let mut forward_closed: HashSet<Vec<i32>> = HashSet::new();
    let mut forward_came_from: HashMap<Vec<i32>, Vec<i32>> = HashMap::new();
    let mut forward_g_score: HashMap<Vec<i32>, f64> = HashMap::new();
    let mut forward_counter = 0usize;

    // Backward search structures
    let mut backward_open = BinaryHeap::new();
    let mut backward_closed: HashSet<Vec<i32>> = HashSet::new();
    let mut backward_came_from: HashMap<Vec<i32>, Vec<i32>> = HashMap::new();
    let mut backward_g_score: HashMap<Vec<i32>, f64> = HashMap::new();
    let mut backward_counter = 0usize;

    // Initialize forward search with all start states
    for (start_vonorms, start_coords) in start_points {
        let start_key = make_key(start_vonorms, start_coords);
        forward_g_score.insert(start_key.clone(), 0.0);

        let h = goal_points.iter()
            .map(|(goal_vonorms, goal_coords)| manhattan_heuristic(start_vonorms, start_coords, goal_vonorms, goal_coords))
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(0.0);

        forward_open.push(AStarNode {
            f_score: h,
            g_score: 0.0,
            h_score: h,
            vonorms: start_vonorms.clone(),
            coords: start_coords.clone(),
            counter: forward_counter,
        });
        forward_counter += 1;
    }

    // Initialize backward search with all goal states
    for (goal_vonorms, goal_coords) in goal_points {
        let goal_key = make_key(goal_vonorms, goal_coords);
        backward_g_score.insert(goal_key.clone(), 0.0);

        let h = start_points.iter()
            .map(|(start_vonorms, start_coords)| manhattan_heuristic(goal_vonorms, goal_coords, start_vonorms, start_coords))
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(0.0);

        backward_open.push(AStarNode {
            f_score: h,
            g_score: 0.0,
            h_score: h,
            vonorms: goal_vonorms.clone(),
            coords: goal_coords.clone(),
            counter: backward_counter,
        });
        backward_counter += 1;
    }

    let mut iterations = 0usize;
    let start_time = std::time::Instant::now();

    // Track g and h for logging
    let mut last_forward_g = 0.0;
    let mut last_forward_h = 0.0;
    let mut last_backward_g = 0.0;
    let mut last_backward_h = 0.0;

    if verbose {
        eprintln!("Starting bidirectional A* search:");
        eprintln!("  {} start states, {} goal states", start_points.len(), goal_points.len());
    }

    while !forward_open.is_empty() && !backward_open.is_empty() {
        iterations += 1;

        if max_iterations > 0 && iterations > max_iterations {
            if verbose {
                eprintln!("Reached max iterations ({})", max_iterations);
            }
            return None;
        }

        if verbose && iterations % 5 == 0 {
            let elapsed = start_time.elapsed();
            eprintln!("Step {}: fwd_open={}, fwd_closed={}, fwd_g={:.1}, fwd_h={:.1}, bwd_open={}, bwd_closed={}, bwd_g={:.1}, bwd_h={:.1}, elapsed={:.2}s",
                     iterations,
                     forward_open.len(),
                     forward_closed.len(),
                     last_forward_g,
                     last_forward_h,
                     backward_open.len(),
                     backward_closed.len(),
                     last_backward_g,
                     last_backward_h,
                     elapsed.as_secs_f64());
        }

        // Expand from forward search
        if let Some(current_node) = forward_open.pop() {
            last_forward_g = current_node.g_score;
            last_forward_h = current_node.h_score;
            let current_key = make_key(&current_node.vonorms, &current_node.coords);

            // Check if we've met the backward search
            if backward_closed.contains(&current_key) {
                let forward_g = forward_g_score.get(&current_key).copied().unwrap_or(0.0);
                let backward_g = backward_g_score.get(&current_key).copied().unwrap_or(0.0);
                let path_length = forward_g + backward_g;

                if verbose {
                    eprintln!("\n✅ Found meeting point at iteration {}! Path length: {}", iterations, path_length);
                }

                return Some(reconstruct_bidirectional_path(
                    &current_key,
                    &forward_came_from,
                    &backward_came_from,
                ));
            }

            // Skip if already visited in forward search
            if forward_closed.contains(&current_key) {
                // Continue to backward search without incrementing iteration
                // (we didn't actually expand this node)
            } else {
                forward_closed.insert(current_key.clone());

                // Get neighbors
                let neighbors = find_neighbor_tuples(
                    &current_node.vonorms,
                    &current_node.coords,
                    elements,
                    n_atoms,
                    xi,
                    delta,
                );

                // Filter by minimum distance
                let filtered_neighbors = filter_neighbors_by_min_distance(
                    &neighbors,
                    n_atoms,
                    xi,
                    delta,
                    min_distance,
                );

                // Sort for deterministic behavior
                let mut filtered_neighbors = filtered_neighbors;
                filtered_neighbors.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

                // Explore each neighbor
                for (neighbor_vonorms, neighbor_coords) in filtered_neighbors {
                    let neighbor_key = make_key(&neighbor_vonorms, &neighbor_coords);

                    if forward_closed.contains(&neighbor_key) {
                        continue;
                    }

                    let edge_cost = 1.0;
                    let tentative_g = current_node.g_score + edge_cost;
                    let current_g = forward_g_score.get(&neighbor_key).copied().unwrap_or(f64::INFINITY);

                    if tentative_g < current_g {
                        forward_came_from.insert(neighbor_key.clone(), current_key.clone());
                        forward_g_score.insert(neighbor_key.clone(), tentative_g);

                        let h = goal_points.iter()
                            .map(|(gv, gc)| manhattan_heuristic(&neighbor_vonorms, &neighbor_coords, gv, gc))
                            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                            .unwrap_or(0.0);

                        forward_open.push(AStarNode {
                            f_score: tentative_g + h,
                            g_score: tentative_g,
                            h_score: h,
                            vonorms: neighbor_vonorms,
                            coords: neighbor_coords,
                            counter: forward_counter,
                        });
                        forward_counter += 1;
                    }
                }

                // Beam search: prune forward open set if it exceeds beam_width
                if beam_width > 0 && forward_open.len() > beam_width {
                    let mut nodes: Vec<_> = forward_open.into_iter().collect();
                    nodes.sort_by(|a, b| a.f_score.partial_cmp(&b.f_score).unwrap_or(Ordering::Equal));
                    nodes.truncate(beam_width);
                    forward_open = BinaryHeap::from(nodes);
                }
            }
        }

        // Expand from backward search
        if let Some(current_node) = backward_open.pop() {
            last_backward_g = current_node.g_score;
            last_backward_h = current_node.h_score;
            let current_key = make_key(&current_node.vonorms, &current_node.coords);

            // Check if we've met the forward search
            if forward_closed.contains(&current_key) {
                let forward_g = forward_g_score.get(&current_key).copied().unwrap_or(0.0);
                let backward_g = backward_g_score.get(&current_key).copied().unwrap_or(0.0);
                let path_length = forward_g + backward_g;

                if verbose {
                    eprintln!("\n✅ Found meeting point at iteration {}! Path length: {}", iterations, path_length);
                }

                return Some(reconstruct_bidirectional_path(
                    &current_key,
                    &forward_came_from,
                    &backward_came_from,
                ));
            }

            // Skip if already visited in backward search
            if backward_closed.contains(&current_key) {
                continue;
            }

            backward_closed.insert(current_key.clone());

            // Get neighbors
            let neighbors = find_neighbor_tuples(
                &current_node.vonorms,
                &current_node.coords,
                elements,
                n_atoms,
                xi,
                delta,
            );

            // Filter by minimum distance
            let filtered_neighbors = filter_neighbors_by_min_distance(
                &neighbors,
                n_atoms,
                xi,
                delta,
                min_distance,
            );

            // Sort for deterministic behavior
            let mut filtered_neighbors = filtered_neighbors;
            filtered_neighbors.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

            // Explore each neighbor
            for (neighbor_vonorms, neighbor_coords) in filtered_neighbors {
                let neighbor_key = make_key(&neighbor_vonorms, &neighbor_coords);

                if backward_closed.contains(&neighbor_key) {
                    continue;
                }

                let edge_cost = 1.0;
                let tentative_g = current_node.g_score + edge_cost;
                let current_g = backward_g_score.get(&neighbor_key).copied().unwrap_or(f64::INFINITY);

                if tentative_g < current_g {
                    backward_came_from.insert(neighbor_key.clone(), current_key.clone());
                    backward_g_score.insert(neighbor_key.clone(), tentative_g);

                    let h = start_points.iter()
                        .map(|(sv, sc)| manhattan_heuristic(&neighbor_vonorms, &neighbor_coords, sv, sc))
                        .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                        .unwrap_or(0.0);

                    backward_open.push(AStarNode {
                        f_score: tentative_g + h,
                        g_score: tentative_g,
                        h_score: h,
                        vonorms: neighbor_vonorms,
                        coords: neighbor_coords,
                        counter: backward_counter,
                    });
                    backward_counter += 1;
                }
            }

            // Beam search: prune backward open set if it exceeds beam_width
            if beam_width > 0 && backward_open.len() > beam_width {
                let mut nodes: Vec<_> = backward_open.into_iter().collect();
                nodes.sort_by(|a, b| a.f_score.partial_cmp(&b.f_score).unwrap_or(Ordering::Equal));
                nodes.truncate(beam_width);
                backward_open = BinaryHeap::from(nodes);
            }
        }
    }

    // No path found
    if verbose {
        eprintln!("\n❌ No path found after {} iterations", iterations);
    }
    None
}
