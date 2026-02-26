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
    parser.add_argument("--delta", type=int, default=30, help="Integer discretization factor (default: 30)")
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
    parser.add_argument("--beam-width", type=int, default=None,
                        help="Beam width for beam search - limits open set size (works with both unidirectional and bidirectional)")
    parser.add_argument("--heuristic-mode", default=None,
                        help="The heuristic to use in this search: manhattan, unimodular_light, unimodular_partial, unimodular_full")
    parser.add_argument("--heuristic-weight", type=float, default=0.5,
                        help="The prefactor on the heuristic - used to control aggressiveness of search.")


    args = parser.parse_args()

    verbose = not args.quiet

    success = pathfind_and_save(
        args.start_cif,
        args.end_cif,
        args.output_json,
        xi=args.xi,
        delta=args.delta,
        min_distance=args.min_distance,
        max_iterations=args.max_iterations,
        use_python=args.python,
        bidirectional=args.bidirectional,
        greedy=args.greedy,
        beam_width=args.beam_width,
        dropout=args.dropout,
        verbose=verbose,
        heuristic_mode=args.heuristic_mode,
        heuristic_weight=args.heuristic_weight
    )

    if success:
        print("\n✅ Pathfinding completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Pathfinding failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
