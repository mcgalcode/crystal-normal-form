"""Rust A* pathfinding between two crystal structures"""

import sys
import argparse


def main():
    """Main entry point for the cnf-pathfind script"""
    parser = argparse.ArgumentParser(
        description="Run A* pathfinding between two crystal structures and save the path to JSON"
    )
    parser.add_argument("start_cif", help="Path to starting structure CIF file")
    parser.add_argument("end_cif", help="Path to ending structure CIF file")
    parser.add_argument("output_json", help="Path to output JSON file")
    parser.add_argument("--xi", type=float, default=0.2, help="Lattice step size (default: 0.2)")
    parser.add_argument("--delta", type=int, default=None, help="Integer discretization factor (default: 30, or computed from --atom-step-length)")
    parser.add_argument("--atom-step-length", type=float, default=None,
                        help="Target step length in Å (alternative to --delta, computes delta automatically)")
    parser.add_argument("--min-distance", type=float, default=0.0,
                        help="Minimum allowed pairwise distance for filtering in Angstroms (default: 0.0)")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Frequency of nodes in the graph being randomly dropped (default: 0.0)")
    parser.add_argument("--max-iterations", type=int, default=100000,
                        help="Maximum pathfinding iterations (default: 100000)")
    parser.add_argument("--python", action="store_true",
                        help="Use Python A* implementation instead of Rust (default: Rust)")
    parser.add_argument("--bidirectional", action="store_true",
                        help="Use Bidirectional A* implementation - works with both Python and Rust (default: false)")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy search implementation - works with both Python and Rust (default: false)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress logging output (default: false)")
    parser.add_argument("--beam-width", type=int, default=1000,
                        help="Beam width for beam search - limits open set size (default: 1000)")
    parser.add_argument("--heuristic-mode", default="manhattan",
                        help="The heuristic to use in this search: manhattan, unimodular_light, unimodular_partial, unimodular_full")
    parser.add_argument("--heuristic-weight", type=float, default=0.5,
                        help="The prefactor on the heuristic - used to control aggressiveness of search.")


    args = parser.parse_args()

    # Deferred imports to keep --help fast
    from pymatgen.core import Structure
    from cnf.navigation.astar import pathfind_and_save_from_cifs
    from cnf.navigation.astar.params import PathFindingParameters
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
        if verbose:
            print(f"Using explicit delta={delta}")
    else:
        delta = 30  # default
        if verbose:
            print(f"Using default delta={delta}")

    params = PathFindingParameters(
        xi=args.xi,
        delta=delta,
        min_distance=args.min_distance,
        max_iterations=args.max_iterations,
        heuristic_mode=args.heuristic_mode,
        heuristic_weight=args.heuristic_weight,
        beam_width=args.beam_width,
        greedy=args.greedy,
        dropout=args.dropout,
    )

    result = pathfind_and_save_from_cifs(
        args.start_cif,
        args.end_cif,
        args.output_json,
        params,
        use_python=args.python,
        verbose=verbose,
    )
    success = result.path is not None

    if success:
        print("\n✅ Pathfinding completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Pathfinding failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
