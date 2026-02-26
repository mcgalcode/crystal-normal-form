"""A* pathfinding between two crystal structures."""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run A* pathfinding between two crystal structures"
    )
    parser.add_argument("start_cif", help="Path to starting structure CIF file")
    parser.add_argument("end_cif", help="Path to ending structure CIF file")
    parser.add_argument("output_json", help="Path to output JSON file")
    parser.add_argument("--xi", type=float, default=0.2, help="Lattice discretization (default: 0.2)")
    parser.add_argument("--delta", type=int, default=None, help="Motif discretization (default: 30, or computed from --atom-step-length)")
    parser.add_argument("--atom-step-length", type=float, default=None,
                        help="Target step length in Å (alternative to --delta)")
    parser.add_argument("--min-distance", type=float, default=0.0,
                        help="Minimum pairwise distance filter in Å (default: 0.0)")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Neighbor dropout probability (default: 0.0)")
    parser.add_argument("--max-iterations", type=int, default=100000,
                        help="Maximum iterations (default: 100000)")
    parser.add_argument("--beam-width", type=int, default=1000,
                        help="Beam width (default: 1000)")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy search instead of A*")
    parser.add_argument("--python", action="store_true",
                        help="Use Python implementation instead of Rust")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress output")

    args = parser.parse_args()

    # Deferred imports
    from pymatgen.core import Structure
    from cnf.navigation.astar import pathfind_from_cifs
    from cnf.navigation import compute_delta_for_step_size

    verbose = not args.quiet

    # Compute delta from atom-step-length if provided
    if args.atom_step_length is not None and args.delta is not None:
        print("Warning: both --delta and --atom-step-length provided, using --atom-step-length")

    if args.atom_step_length is not None:
        start_struct = Structure.from_file(args.start_cif)
        end_struct = Structure.from_file(args.end_cif)
        delta = max(
            compute_delta_for_step_size(start_struct, args.atom_step_length),
            compute_delta_for_step_size(end_struct, args.atom_step_length)
        )
        if verbose:
            print(f"Computed delta={delta} from atom-step-length={args.atom_step_length} Å")
    elif args.delta is not None:
        delta = args.delta
    else:
        delta = 30

    result = pathfind_from_cifs(
        args.start_cif,
        args.end_cif,
        xi=args.xi,
        delta=delta,
        min_distance=args.min_distance,
        max_iterations=args.max_iterations,
        beam_width=args.beam_width,
        greedy=args.greedy,
        dropout=args.dropout,
        use_python=args.python,
        verbose=verbose,
    )

    # Save result
    result.to_json(args.output_json)
    if verbose:
        print(f"Saved to {args.output_json}")

    if result.attempts[0].found:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
