/// Unimodular Manhattan heuristic for A* pathfinding
///
/// Pre-computes all unimodular-transformed variants of goal CNFs, then returns
/// the minimum Manhattan (L1) distance over all variants. This gives a much
/// tighter estimate than plain Manhattan distance, which overestimates at
/// Voronoi class boundaries where canonical orderings change.

use std::collections::HashSet;
use crate::linalg::{mat_inv, mat_to_flat};
use crate::mnf::compute_atom_labels;
use crate::permutations::PERMUTATIONS;

// =============================================================================
// Manhattan Heuristic
// =============================================================================

/// Compute Manhattan distance (L1) heuristic between two CNF states.
///
/// Returns sum of absolute differences multiplied by 10 (matches Python implementation).
pub fn manhattan_heuristic(
    vonorms1: &[i32], coords1: &[i32],
    vonorms2: &[i32], coords2: &[i32],
) -> f64 {
    debug_assert_eq!(coords1.len(), coords2.len());
    debug_assert_eq!(vonorms1.len(), vonorms2.len());

    let vonorm_dist: i32 = vonorms1.iter()
        .zip(vonorms2.iter())
        .map(|(&v1, &v2)| (v1 - v2).abs())
        .sum();

    let coord_dist: i32 = coords1.iter()
        .zip(coords2.iter())
        .map(|(&c1, &c2)| (c1 - c2).abs())
        .sum();

    (vonorm_dist + coord_dist) as f64 * 10.0
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Heuristic mode for A* search
#[derive(Clone, Copy, Debug)]
pub enum HeuristicMode {
    /// Plain Manhattan distance (L1) with 10x scaling
    Manhattan,
    /// 168 vonorm-permutation-derived matrices (one per permutation)
    UnimodularLight,
    /// One matrix per (zero_set, conorm_perm) across all coforms
    UnimodularPartial,
    /// All matrices across all coforms
    UnimodularFull,
}

impl HeuristicMode {
    pub fn from_str(s: &str) -> Self {
        match s {
            "manhattan" => HeuristicMode::Manhattan,
            "unimodular_light" => HeuristicMode::UnimodularLight,
            "unimodular_partial" => HeuristicMode::UnimodularPartial,
            "unimodular_full" => HeuristicMode::UnimodularFull,
            _ => panic!("Unknown heuristic mode: '{}'. Expected one of: manhattan, unimodular_light, unimodular_partial, unimodular_full", s),
        }
    }
}

/// Pre-computed goal variants for unimodular manhattan heuristic
pub struct GoalVariants {
    /// Each row is a flattened [vonorms(7) ++ motif_coords((n_atoms-1)*3)] variant
    pub variants: Vec<Vec<i32>>,
    pub weight: f64,
}

// ---------------------------------------------------------------------------
// Voronoi index -> superbasis column vector mapping (for light mode)
// ---------------------------------------------------------------------------

/// Column vectors mapping Voronoi vector index to the 3D column of the
/// unimodular matrix. Used in light mode to construct matrices from vonorm
/// permutation indices [1:4].
///
/// Matches Python's VORONOI_IDX_TO_COLUMN in vonorm_unimodular.py
const VORONOI_IDX_TO_COLUMN: [[i32; 3]; 7] = [
    [-1, -1, -1], // idx 0: v3 = -(v0+v1+v2)
    [ 1,  0,  0], // idx 1: v0
    [ 0,  1,  0], // idx 2: v1
    [ 0,  0,  1], // idx 3: v2
    [ 0, -1, -1], // idx 4: v0+v1
    [-1,  0, -1], // idx 5: v0+v2
    [-1, -1,  0], // idx 6: v0+v3
];

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Build a 3x3 unimodular matrix from 3 Voronoi vector indices.
/// Corresponds to Python's `VonormPermutationMatrix.from_vector_idxs(perm[1:4])`.
fn matrix_from_voronoi_idxs(idxs: [usize; 3]) -> [[i32; 3]; 3] {
    let c0 = VORONOI_IDX_TO_COLUMN[idxs[0]];
    let c1 = VORONOI_IDX_TO_COLUMN[idxs[1]];
    let c2 = VORONOI_IDX_TO_COLUMN[idxs[2]];
    // Matrix columns -> row-major [row][col]
    [
        [c0[0], c1[0], c2[0]],
        [c0[1], c1[1], c2[1]],
        [c0[2], c1[2], c2[2]],
    ]
}

/// Positive modulo (Euclidean remainder), result in [0, m)
#[inline]
fn pos_mod(a: i32, m: i32) -> i32 {
    ((a % m) + m) % m
}

/// Transform coordinates: result[atom][dim] = (U_inv @ coords[atom]) mod delta
///
/// `full_coords` is flat [x0,y0,z0, x1,y1,z1, ...] for n_atoms atoms.
fn transform_coords(
    u_inv: &[[i32; 3]; 3],
    full_coords: &[i32],
    n_atoms: usize,
    delta: i32,
) -> Vec<i32> {
    let mut result = vec![0i32; n_atoms * 3];
    for j in 0..n_atoms {
        let off = j * 3;
        let x = full_coords[off];
        let y = full_coords[off + 1];
        let z = full_coords[off + 2];
        result[off]     = pos_mod(u_inv[0][0] * x + u_inv[0][1] * y + u_inv[0][2] * z, delta);
        result[off + 1] = pos_mod(u_inv[1][0] * x + u_inv[1][1] * y + u_inv[1][2] * z, delta);
        result[off + 2] = pos_mod(u_inv[2][0] * x + u_inv[2][1] * y + u_inv[2][2] * z, delta);
    }
    result
}

/// Sort atoms by (label, x, y, z) lexicographically.
/// `coords` is flat [x0,y0,z0, x1,y1,z1, ...].
fn sort_atoms(coords: &[i32], atom_labels: &[usize], n_atoms: usize) -> Vec<i32> {
    let mut indices: Vec<usize> = (0..n_atoms).collect();
    indices.sort_by(|&a, &b| {
        atom_labels[a]
            .cmp(&atom_labels[b])
            .then_with(|| coords[a * 3].cmp(&coords[b * 3]))
            .then_with(|| coords[a * 3 + 1].cmp(&coords[b * 3 + 1]))
            .then_with(|| coords[a * 3 + 2].cmp(&coords[b * 3 + 2]))
    });

    let mut sorted = Vec::with_capacity(n_atoms * 3);
    for &idx in &indices {
        sorted.push(coords[idx * 3]);
        sorted.push(coords[idx * 3 + 1]);
        sorted.push(coords[idx * 3 + 2]);
    }
    sorted
}

/// Apply origin shifts and atom sorting, returning completed variant tuples.
///
/// For each of the first `num_origin_atoms` atoms, shift so that atom becomes
/// the origin, sort by (label, x, y, z), extract MNF (skip origin), and
/// concatenate with permuted vonorms.
fn apply_shifts_and_sort(
    permuted_vonorms: &[i32],
    transformed_coords: &[i32],
    atom_labels: &[usize],
    n_atoms: usize,
    num_origin_atoms: usize,
    delta: i32,
) -> Vec<Vec<i32>> {
    let mut variants = Vec::with_capacity(num_origin_atoms);
    let motif_len = (n_atoms - 1) * 3;

    for origin_idx in 0..num_origin_atoms {
        let off = origin_idx * 3;
        let sx = -transformed_coords[off];
        let sy = -transformed_coords[off + 1];
        let sz = -transformed_coords[off + 2];

        let mut shifted = vec![0i32; n_atoms * 3];
        for j in 0..n_atoms {
            let jo = j * 3;
            shifted[jo]     = pos_mod(transformed_coords[jo]     + sx, delta);
            shifted[jo + 1] = pos_mod(transformed_coords[jo + 1] + sy, delta);
            shifted[jo + 2] = pos_mod(transformed_coords[jo + 2] + sz, delta);
        }

        let sorted = sort_atoms(&shifted, atom_labels, n_atoms);

        // Build variant: vonorms(7) ++ motif_coords (skip first atom = origin)
        let mut variant = Vec::with_capacity(7 + motif_len);
        variant.extend_from_slice(permuted_vonorms);
        variant.extend_from_slice(&sorted[3..]); // skip origin
        variants.push(variant);
    }

    variants
}


// ---------------------------------------------------------------------------
// Variant generation
// ---------------------------------------------------------------------------

/// Generate all unimodular variants for a single goal CNF.
fn generate_variants_for_goal(
    vonorms: &[i32],
    motif_coords: &[i32],
    elements: &[String],
    delta: i32,
    mode: HeuristicMode,
) -> Vec<Vec<i32>> {
    let atom_labels = compute_atom_labels(elements);
    let n_atoms = elements.len();
    let num_origin_atoms = atom_labels.iter().filter(|&&l| l == 0).count();

    // Reconstruct full coords: origin at (0,0,0) + stored motif coords
    let mut full_coords = vec![0i32; n_atoms * 3];
    full_coords[3..].copy_from_slice(motif_coords);

    let mut variants: Vec<Vec<i32>> = Vec::new();

    match mode {
        HeuristicMode::UnimodularLight => {
            // Iterate over all 168 vonorm permutations (no matrix dedup since
            // different vonorm perms with the same matrix give different permuted
            // vonorms, producing distinct variants).
            for vonorm_perm in PERMUTATIONS.conorm_to_vonorm_perm.values() {
                let permuted_v: Vec<i32> = vonorm_perm.iter().map(|&i| vonorms[i]).collect();

                let idxs = [vonorm_perm[1], vonorm_perm[2], vonorm_perm[3]];
                let u_mat = matrix_from_voronoi_idxs(idxs);
                let u_inv = mat_inv(&u_mat);

                let transformed = transform_coords(&u_inv, &full_coords, n_atoms, delta);
                let new_variants = apply_shifts_and_sort(
                    &permuted_v,
                    &transformed,
                    &atom_labels,
                    n_atoms,
                    num_origin_atoms,
                    delta,
                );
                variants.extend(new_variants);
            }
        }

        HeuristicMode::UnimodularPartial | HeuristicMode::UnimodularFull => {
            let partial = matches!(mode, HeuristicMode::UnimodularPartial);
            let mut seen_mats: HashSet<[i32; 9]> = HashSet::new();

            for (_zero_set, perm_to_mats) in &PERMUTATIONS.zero_to_perm_to_mats {
                for (conorm_perm, mat_list) in perm_to_mats {
                    let vonorm_perm = match PERMUTATIONS.conorm_to_vonorm_perm.get(conorm_perm) {
                        Some(vp) => vp,
                        None => continue,
                    };
                    let permuted_v: Vec<i32> =
                        vonorm_perm.iter().map(|&i| vonorms[i]).collect();

                    let mats_to_use = if partial { &mat_list[..1] } else { &mat_list[..] };

                    for mat_vecs in mats_to_use {
                        let u_mat = [
                            [mat_vecs[0][0], mat_vecs[0][1], mat_vecs[0][2]],
                            [mat_vecs[1][0], mat_vecs[1][1], mat_vecs[1][2]],
                            [mat_vecs[2][0], mat_vecs[2][1], mat_vecs[2][2]],
                        ];

                        if !seen_mats.insert(mat_to_flat(&u_mat)) {
                            continue;
                        }

                        let u_inv = mat_inv(&u_mat);
                        let transformed =
                            transform_coords(&u_inv, &full_coords, n_atoms, delta);
                        let new_variants = apply_shifts_and_sort(
                            &permuted_v,
                            &transformed,
                            &atom_labels,
                            n_atoms,
                            num_origin_atoms,
                            delta,
                        );
                        variants.extend(new_variants);
                    }
                }
            }
        }

        HeuristicMode::Manhattan => unreachable!(),
    }

    // Deduplicate variants
    let mut seen: HashSet<Vec<i32>> = HashSet::new();
    variants.retain(|v| seen.insert(v.clone()));
    variants
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Pre-compute all goal variants for the unimodular manhattan heuristic.
///
/// Called once at the start of A* search. The returned `GoalVariants` is then
/// passed to `unimodular_heuristic()` at every node expansion.
pub fn precompute_goal_variants(
    goal_points: &[(Vec<i32>, Vec<i32>)],
    elements: &[String],
    delta: i32,
    mode: HeuristicMode,
    weight: f64,
) -> GoalVariants {
    let mut all_variants = Vec::new();

    for (vonorms, coords) in goal_points {
        let gv = generate_variants_for_goal(vonorms, coords, elements, delta, mode);
        all_variants.extend(gv);
    }

    if !all_variants.is_empty() {
        eprintln!(
            "Precomputed {} goal variants (mode: {:?}, weight: {:.2})",
            all_variants.len(),
            mode,
            weight,
        );
    }

    GoalVariants {
        variants: all_variants,
        weight,
    }
}

/// Compute the unimodular manhattan heuristic value for a single node.
///
/// Returns `min_over_variants(L1_distance) * weight`.
pub fn unimodular_heuristic(
    vonorms: &[i32],
    coords: &[i32],
    goal_variants: &GoalVariants,
) -> f64 {
    let tuple_len = vonorms.len() + coords.len();
    let mut min_dist = i64::MAX;

    for variant in &goal_variants.variants {
        debug_assert_eq!(variant.len(), tuple_len);

        let mut dist: i64 = 0;
        // Vonorms part
        for (i, &v) in vonorms.iter().enumerate() {
            dist += (v as i64 - variant[i] as i64).abs();
        }
        // Coords part
        let offset = vonorms.len();
        for (i, &c) in coords.iter().enumerate() {
            dist += (c as i64 - variant[offset + i] as i64).abs();
        }

        if dist < min_dist {
            min_dist = dist;
        }
    }

    min_dist as f64 * goal_variants.weight
}
