"""Unified CLI for CNF barrier search workflow.

Usage:
    cnf search <start.cif> <end.cif> [options]
    cnf sample <start.cif> <end.cif> --xi <xi> --delta <delta> [options]
    cnf sample <start.cif> <end.cif> --from <search_result.json> [options]
    cnf sweep <start.cif> <end.cif> --xi <xi> --delta <delta> --ceiling <eV> [options]
    cnf sweep <start.cif> <end.cif> --from <sample_result.json> [options]
    cnf ratchet <start.cif> <end.cif> --xi <xi> --delta <delta> --ceiling <eV> [options]
    cnf ratchet <start.cif> <end.cif> --from <sweep_result.json> [options]
    cnf find-barrier <start.cif> <end.cif> [options]

Verbosity levels:
    (default)   Phase-level progress (pass summaries, round info)
    -v          + A* iteration progress within each search
    -vv         + Additional detail (reserved for future use)
    -q          Silent mode (no output except errors)
"""

import argparse
import json
import sys
from pathlib import Path


def load_previous_result(filepath):
    """Load a previous phase result from JSON."""
    with open(filepath) as f:
        return json.load(f)


def save_result(result, output_dir, default_name):
    """Save result to JSON file."""
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        result.to_json(str(out_path / default_name))
        print(f"Saved to: {out_path / default_name}")


def get_verbosity(args) -> int:
    """Calculate verbosity level from args.

    Returns:
        0 = silent (--quiet)
        1 = phase output (default, no flags)
        2 = + A* iteration progress (-v)
        3+ = more detail (-vv, -vvv)
    """
    if args.quiet:
        return 0
    return 1 + args.verbose


def search_command(args):
    """Phase 1: Parameter search to find optimal xi/delta/min_distance."""
    from pymatgen.core import Structure
    from cnf import UnitCell
    from cnf.navigation.astar.iterative import search

    start = Structure.from_file(args.start)
    end = Structure.from_file(args.end)

    start_uc = UnitCell.from_pymatgen_structure(start)
    end_uc = UnitCell.from_pymatgen_structure(end)

    xi_values = [float(x) for x in args.xi.split(',')] if args.xi else None
    atom_steps = [float(x) for x in args.atom_steps.split(',')] if args.atom_steps else None

    result = search(
        start_uc=start_uc,
        end_uc=end_uc,
        xi_values=xi_values,
        atom_step_lengths=atom_steps,
        max_iterations=args.max_iters,
        beam_width=args.beam_width,
        dropout=args.dropout,
        n_workers=args.workers,
        verbosity=get_verbosity(args),
        output_dir=args.output,
    )

    if result.success:
        print(f"\nRecommended: xi={result.recommended_xi}, delta={result.recommended_delta}, min_distance={result.recommended_min_distance:.3f}")
        save_result(result, args.output, "search_result.json")
    else:
        print("\nNo parameters found that produce valid paths.")
        sys.exit(1)


def sample_command(args):
    """Phase 2: Sample diverse paths to discover initial energy ceiling."""
    from pymatgen.core import Structure
    from cnf import UnitCell
    from cnf.navigation.astar.iterative import sample
    from cnf.navigation.endpoints import get_endpoint_cnfs
    from cnf.calculation.grace import GraceCalcProvider

    # Load parameters from previous result if --from is specified
    if args.from_result:
        prev = load_previous_result(args.from_result)
        xi = prev.get('recommended_xi') or prev.get('context', {}).get('xi')
        delta = prev.get('recommended_delta') or prev.get('context', {}).get('delta')
        min_distance = prev.get('recommended_min_distance')
        if xi is None or delta is None:
            print(f"Error: Could not extract xi/delta from {args.from_result}")
            sys.exit(1)
        print(f"Loaded from {args.from_result}: xi={xi}, delta={delta}, min_distance={min_distance}")
    else:
        if args.xi is None or args.delta is None:
            print("Error: --xi and --delta are required (or use --from)")
            sys.exit(1)
        xi = args.xi
        delta = args.delta
        min_distance = args.min_distance

    start = Structure.from_file(args.start)
    end = Structure.from_file(args.end)

    start_uc = UnitCell.from_pymatgen_structure(start)
    end_uc = UnitCell.from_pymatgen_structure(end)

    start_cnfs, goal_cnfs = get_endpoint_cnfs(start_uc, end_uc, xi=xi, delta=delta)

    calc_provider = GraceCalcProvider(model_path=args.model)

    result = sample(
        start_cnfs=start_cnfs,
        goal_cnfs=goal_cnfs,
        calc_provider=calc_provider,
        num_samples=args.num_samples,
        dropout_range=(args.dropout_min, args.dropout_max),
        min_distance=min_distance,
        max_iterations=args.max_iters,
        beam_width=args.beam_width,
        n_workers=args.workers,
        verbosity=get_verbosity(args),
        output_dir=args.output,
    )

    print(f"\nFound {len(result.paths)} paths")
    if result.best_barrier:
        print(f"Best barrier: {result.best_barrier:.4f} eV")
    save_result(result, args.output, "sample_result.json")


def sweep_command(args):
    """Phase 3: Parallel ceiling sweep."""
    from pymatgen.core import Structure
    from cnf import UnitCell
    from cnf.navigation.astar.iterative import sweep
    from cnf.calculation.grace import GraceCalcProvider

    # Load parameters from previous result if --from is specified
    if args.from_result:
        prev = load_previous_result(args.from_result)
        xi = prev.get('context', {}).get('xi')
        delta = prev.get('context', {}).get('delta')
        # Get best_barrier from SearchResult or compute from attempts
        ceiling = prev.get('best_barrier')
        if ceiling is None:
            # Try to find best barrier from attempts
            attempts = prev.get('attempts', [])
            barriers = [a['path']['barrier'] for a in attempts if a.get('path') and a['path'].get('barrier')]
            ceiling = min(barriers) if barriers else None
        if xi is None or delta is None or ceiling is None:
            print(f"Error: Could not extract xi/delta/ceiling from {args.from_result}")
            sys.exit(1)
        print(f"Loaded from {args.from_result}: xi={xi}, delta={delta}, ceiling={ceiling:.4f}")
    else:
        if args.xi is None or args.delta is None or args.ceiling is None:
            print("Error: --xi, --delta, and --ceiling are required (or use --from)")
            sys.exit(1)
        xi = args.xi
        delta = args.delta
        ceiling = args.ceiling

    start = Structure.from_file(args.start)
    end = Structure.from_file(args.end)

    start_uc = UnitCell.from_pymatgen_structure(start)
    end_uc = UnitCell.from_pymatgen_structure(end)

    calc_provider = GraceCalcProvider(model_path=args.model)

    result = sweep(
        start_uc=start_uc,
        end_uc=end_uc,
        max_ceiling=ceiling,
        calc_provider=calc_provider,
        xi=xi,
        delta=delta,
        num_ceilings=args.num_ceilings,
        attempts_per_ceiling=args.attempts,
        max_passes=args.max_passes,
        xi_factor=args.xi_factor,
        delta_factor=args.delta_factor,
        dropout=args.dropout,
        max_iterations=args.max_iters,
        beam_width=args.beam_width,
        n_workers=args.workers,
        verbosity=get_verbosity(args),
        output_dir=args.output,
    )

    print(f"\nFound {len(result.all_paths)} total paths")
    if result.best_barrier:
        print(f"Best barrier: {result.best_barrier:.4f} eV")
    save_result(result, args.output, "sweep_result.json")


def ratchet_command(args):
    """Phase 4: Serial barrier refinement with ratcheting ceiling."""
    from pymatgen.core import Structure
    from cnf import UnitCell
    from cnf.navigation.astar.iterative import ratchet
    from cnf.navigation.endpoints import get_endpoint_cnfs
    from cnf.calculation.grace import GraceCalculator

    # Load parameters from previous result if --from is specified
    if args.from_result:
        prev = load_previous_result(args.from_result)
        # Handle both SearchResult and CeilingSweepResult formats
        if 'context' in prev:
            xi = prev['context'].get('xi')
            delta = prev['context'].get('delta')
        elif 'results' in prev and prev['results']:
            xi = prev['results'][0].get('context', {}).get('xi')
            delta = prev['results'][0].get('context', {}).get('delta')
        else:
            xi = delta = None

        # Get best barrier
        ceiling = prev.get('best_barrier')
        if ceiling is None and 'results' in prev:
            # CeilingSweepResult - find best across all results
            all_barriers = []
            for r in prev.get('results', []):
                for a in r.get('attempts', []):
                    if a.get('path') and a['path'].get('barrier'):
                        all_barriers.append(a['path']['barrier'])
            ceiling = min(all_barriers) if all_barriers else None
        if ceiling is None:
            # SearchResult format
            attempts = prev.get('attempts', [])
            barriers = [a['path']['barrier'] for a in attempts if a.get('path') and a['path'].get('barrier')]
            ceiling = min(barriers) if barriers else None

        if xi is None or delta is None or ceiling is None:
            print(f"Error: Could not extract xi/delta/ceiling from {args.from_result}")
            sys.exit(1)
        print(f"Loaded from {args.from_result}: xi={xi}, delta={delta}, ceiling={ceiling:.4f}")
    else:
        if args.xi is None or args.delta is None or args.ceiling is None:
            print("Error: --xi, --delta, and --ceiling are required (or use --from)")
            sys.exit(1)
        xi = args.xi
        delta = args.delta
        ceiling = args.ceiling

    start = Structure.from_file(args.start)
    end = Structure.from_file(args.end)

    start_uc = UnitCell.from_pymatgen_structure(start)
    end_uc = UnitCell.from_pymatgen_structure(end)

    start_cnfs, goal_cnfs = get_endpoint_cnfs(start_uc, end_uc, xi=xi, delta=delta)

    calc = GraceCalculator(model_path=args.model) if args.model else GraceCalculator()

    result = ratchet(
        start_cnfs=start_cnfs,
        goal_cnfs=goal_cnfs,
        initial_ceiling=ceiling,
        energy_calc=calc,
        paths_per_round=args.paths_per_round,
        max_rounds=args.max_rounds,
        dropout=args.dropout,
        min_dropout=args.min_dropout,
        max_iterations=args.max_iters,
        beam_width=args.beam_width,
        verbosity=get_verbosity(args),
        output_dir=args.output,
    )

    print(f"\nCompleted {len(result.results)} rounds")
    if result.best_barrier:
        print(f"Final barrier: {result.best_barrier:.4f} eV")
    if result.final_ceiling:
        print(f"Final ceiling: {result.final_ceiling:.4f} eV")
    save_result(result, args.output, "ratchet_result.json")


def find_barrier_command(args):
    """Run the full 4-phase barrier search workflow."""
    from pymatgen.core import Structure
    from cnf import UnitCell
    from cnf.navigation.astar.iterative import search, sample, sweep, ratchet
    from cnf.navigation.endpoints import get_endpoint_cnfs
    from cnf.calculation.grace import GraceCalculator, GraceCalcProvider

    start = Structure.from_file(args.start)
    end = Structure.from_file(args.end)

    start_uc = UnitCell.from_pymatgen_structure(start)
    end_uc = UnitCell.from_pymatgen_structure(end)

    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # For phases that support parallel execution, use provider
    calc_provider = GraceCalcProvider(model_path=args.model)
    # For phases that don't support parallel yet, use calculator instance
    calc = calc_provider()

    # Parse xi/atom-steps if provided
    xi_values = [float(x) for x in args.xi.split(',')] if args.xi else None
    atom_steps = [float(x) for x in args.atom_steps.split(',')] if args.atom_steps else None

    # Determine max_iters for early phases (use default if not specified)
    # Later phases will use learned estimates, optionally capped by user's max_iters
    DEFAULT_EARLY_MAX_ITERS = 5000
    early_max_iters = args.max_iters if args.max_iters is not None else DEFAULT_EARLY_MAX_ITERS
    user_cap = args.max_iters  # None means no cap

    # Phase 1: Parameter search
    print("=" * 60)
    print("PHASE 1: Parameter Search")
    print("=" * 60)
    verbosity = get_verbosity(args)
    print(f"  max_iters={early_max_iters}" + (" (default)" if user_cap is None else ""))

    search_result = search(
        start_uc=start_uc,
        end_uc=end_uc,
        xi_values=xi_values,
        atom_step_lengths=atom_steps,
        max_iterations=early_max_iters,
        beam_width=args.beam_width,
        dropout=args.dropout,
        n_workers=args.workers,
        verbosity=verbosity,
        output_dir=str(output_dir / "phase1_search") if output_dir else None,
    )

    if not search_result.success:
        print("\nPhase 1 failed: No parameters found.")
        sys.exit(1)

    xi = search_result.recommended_xi
    delta = search_result.recommended_delta
    min_distance = search_result.recommended_min_distance
    print(f"\nPhase 1 complete: xi={xi}, delta={delta}, min_distance={min_distance:.3f}")
    if output_dir:
        search_result.to_json(str(output_dir / "search_result.json"))

    # Phase 2: Sampling
    print("\n" + "=" * 60)
    print("PHASE 2: Path Sampling")
    print("=" * 60)
    print(f"  max_iters={early_max_iters}" + (" (default)" if user_cap is None else ""))
    start_cnfs, goal_cnfs = get_endpoint_cnfs(start_uc, end_uc, xi=xi, delta=delta)

    sample_result = sample(
        start_cnfs=start_cnfs,
        goal_cnfs=goal_cnfs,
        calc_provider=calc_provider,
        num_samples=args.num_samples,
        dropout_range=(args.dropout_min, args.dropout_max),
        min_distance=min_distance,
        max_iterations=early_max_iters,
        beam_width=args.beam_width,
        n_workers=args.workers,
        verbosity=verbosity,
        output_dir=str(output_dir / "phase2_sample") if output_dir else None,
    )

    if not sample_result.paths:
        print("\nPhase 2 failed: No paths found.")
        sys.exit(1)

    ceiling = sample_result.best_barrier
    print(f"\nPhase 2 complete: {len(sample_result.paths)} paths, best barrier={ceiling:.4f} eV")
    if output_dir:
        sample_result.to_json(str(output_dir / "sample_result.json"))

    # Compute informed max_iters for later phases based on Phase 2 results
    # Use max iterations from successful attempts with headroom
    sample_max_iters = sample_result.max_successful_iterations
    if sample_max_iters:
        # Base estimate: max observed * 1.5 headroom, but at least 500
        baseline_max_iters = max(500, int(sample_max_iters * 1.5))
        # Sweep uses finer resolution, so paths are longer
        # Scale by resolution change: xi_factor=0.9 means 1/0.9 ≈ 1.11x longer paths
        resolution_scale = 1.0 / args.xi_factor  # ~1.11 for default
        sweep_max_iters_computed = int(baseline_max_iters * resolution_scale)

        # Apply user cap if specified
        if user_cap is not None and sweep_max_iters_computed > user_cap:
            sweep_max_iters = user_cap
            print(f"  max_iters update: sample_max={sample_max_iters} → "
                  f"baseline={baseline_max_iters} → computed={sweep_max_iters_computed} "
                  f"(scale={resolution_scale:.2f})")
            print(f"  WARNING: computed estimate {sweep_max_iters_computed} exceeds "
                  f"--max-iters cap {user_cap}")
        else:
            sweep_max_iters = sweep_max_iters_computed
            print(f"  max_iters update: sample_max={sample_max_iters} → "
                  f"baseline={baseline_max_iters} → sweep={sweep_max_iters} "
                  f"(scale={resolution_scale:.2f})")
    else:
        sweep_max_iters = early_max_iters
        print(f"  max_iters: using default {sweep_max_iters} (no successful samples)")

    # Phase 3: Ceiling sweep
    print("\n" + "=" * 60)
    print("PHASE 3: Ceiling Sweep")
    print("=" * 60)
    print(f"  max_iters={sweep_max_iters}")
    sweep_result = sweep(
        start_uc=start_uc,
        end_uc=end_uc,
        max_ceiling=ceiling,
        calc_provider=calc_provider,
        xi=xi,
        delta=delta,
        num_ceilings=args.num_ceilings,
        attempts_per_ceiling=args.attempts,
        max_passes=args.max_passes,
        xi_factor=args.xi_factor,
        dropout=args.sweep_dropout,
        max_iterations=sweep_max_iters,
        beam_width=args.beam_width,
        n_workers=args.workers,
        verbosity=verbosity,
        output_dir=str(output_dir / "phase3_sweep") if output_dir else None,
    )

    if sweep_result.best_barrier:
        ceiling = sweep_result.best_barrier
    print(f"\nPhase 3 complete: {len(sweep_result.all_paths)} paths, best barrier={ceiling:.4f} eV")
    if output_dir:
        sweep_result.to_json(str(output_dir / "sweep_result.json"))

    # Compute informed max_iters for ratchet based on sweep results
    # Collect iteration stats from all successful sweep attempts
    sweep_successful_iters = []
    for sr in sweep_result.results:
        sweep_successful_iters.extend(sr.successful_iterations)
    if sweep_successful_iters:
        # Ratchet starts from sweep's max with headroom
        sweep_max_observed = max(sweep_successful_iters)
        ratchet_max_iters_computed = max(500, int(sweep_max_observed * 1.5))

        # Apply user cap if specified
        if user_cap is not None and ratchet_max_iters_computed > user_cap:
            ratchet_max_iters = user_cap
            print(f"  max_iters update: sweep_max={sweep_max_observed} → "
                  f"computed={ratchet_max_iters_computed}")
            print(f"  WARNING: computed estimate {ratchet_max_iters_computed} exceeds "
                  f"--max-iters cap {user_cap}")
        else:
            ratchet_max_iters = ratchet_max_iters_computed
            print(f"  max_iters update: sweep_max={sweep_max_observed} → "
                  f"ratchet={ratchet_max_iters}")
    else:
        # Fall back to sweep estimate or user max
        ratchet_max_iters = sweep_max_iters
        print(f"  max_iters: using sweep value {ratchet_max_iters} (no successful sweeps)")

    # Phase 4: Ratchet refinement
    print("\n" + "=" * 60)
    print("PHASE 4: Ratchet Refinement")
    print("=" * 60)
    print(f"  max_iters={ratchet_max_iters} (adaptive scaling enabled)")
    start_cnfs, goal_cnfs = get_endpoint_cnfs(start_uc, end_uc, xi=xi, delta=delta)

    ratchet_result = ratchet(
        start_cnfs=start_cnfs,
        goal_cnfs=goal_cnfs,
        initial_ceiling=ceiling,
        energy_calc=calc,
        paths_per_round=args.paths_per_round,
        max_rounds=args.max_rounds,
        dropout=args.dropout,
        max_iterations=ratchet_max_iters,
        beam_width=args.beam_width,
        verbosity=verbosity,
        output_dir=str(output_dir / "phase4_ratchet") if output_dir else None,
    )

    if output_dir:
        ratchet_result.to_json(str(output_dir / "ratchet_result.json"))

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Parameters: xi={xi}, delta={delta}, min_distance={min_distance:.3f}")
    print(f"Phase 2 barrier: {sample_result.best_barrier:.4f} eV")
    print(f"Phase 3 barrier: {sweep_result.best_barrier:.4f} eV" if sweep_result.best_barrier else "Phase 3: no improvement")
    if ratchet_result.best_barrier:
        print(f"Phase 4 barrier: {ratchet_result.best_barrier:.4f} eV")
        print(f"\nFinal barrier: {ratchet_result.best_barrier:.4f} eV")
    else:
        print(f"\nFinal barrier: {ceiling:.4f} eV")
    if output_dir:
        print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        prog='cnf',
        description='CNF barrier search workflow',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Common arguments
    def add_common_args(p):
        p.add_argument('start', help='Start structure CIF file')
        p.add_argument('end', help='End structure CIF file')
        p.add_argument('--output', '-o', help='Output directory')
        p.add_argument('-v', '--verbose', action='count', default=0,
                       help='Increase verbosity (-v for A* progress, -vv for more)')
        p.add_argument('-q', '--quiet', action='store_true', help='Silent mode')
        p.add_argument('--beam-width', type=int, default=1000, help='Beam width (default: 1000)')
        p.add_argument('--max-iters', type=int, default=5000, help='Max iterations (default: 5000)')

    def add_discretization_args(p, required=True):
        p.add_argument('--xi', type=float, required=False, help='Lattice discretization (Angstrom^2)')
        p.add_argument('--delta', type=int, required=False, help='Motif discretization')

    def add_energy_args(p):
        p.add_argument('--model', help='Path to local GRACE model (uses foundation model if not specified)')

    def add_from_arg(p, help_text):
        p.add_argument('--from', dest='from_result', help=help_text)

    # search
    search_parser = subparsers.add_parser('search', help='Phase 1: Parameter search')
    add_common_args(search_parser)
    search_parser.add_argument('--xi', help='Comma-separated xi values (default: 1.5,1.25,1.0,0.75)')
    search_parser.add_argument('--atom-steps', help='Comma-separated atom step lengths in Angstrom (default: 0.4,0.3,0.2,0.1)')
    search_parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability (default: 0.0)')
    search_parser.add_argument('--workers', type=int, default=0, help='Number of parallel workers (default: 0 = auto)')
    search_parser.set_defaults(func=search_command)

    # sample
    sample_parser = subparsers.add_parser('sample', help='Phase 2: Path sampling')
    add_common_args(sample_parser)
    add_discretization_args(sample_parser, required=False)
    add_energy_args(sample_parser)
    add_from_arg(sample_parser, 'Load xi/delta/min_distance from search result JSON')
    sample_parser.add_argument('--num-samples', type=int, default=20, help='Number of samples (default: 20)')
    sample_parser.add_argument('--dropout-min', type=float, default=0.05, help='Min dropout (default: 0.05)')
    sample_parser.add_argument('--dropout-max', type=float, default=0.1, help='Max dropout (default: 0.1)')
    sample_parser.add_argument('--min-distance', type=float, help='Minimum interatomic distance filter')
    sample_parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers (default: 1, 0=auto)')
    sample_parser.set_defaults(func=sample_command)

    # sweep
    sweep_parser = subparsers.add_parser('sweep', help='Phase 3: Ceiling sweep')
    add_common_args(sweep_parser)
    add_discretization_args(sweep_parser, required=False)
    add_energy_args(sweep_parser)
    add_from_arg(sweep_parser, 'Load xi/delta/ceiling from sample result JSON')
    sweep_parser.add_argument('--ceiling', type=float, help='Max energy ceiling (eV)')
    sweep_parser.add_argument('--num-ceilings', type=int, default=5, help='Number of ceiling levels (default: 5)')
    sweep_parser.add_argument('--attempts', type=int, default=3, help='Sweep: A* searches per ceiling level for path diversity (default: 3)')
    sweep_parser.add_argument('--max-passes', type=int, default=3, help='Max refinement passes (default: 3)')
    sweep_parser.add_argument('--xi-factor', type=float, default=0.9, help='Xi reduction factor (default: 0.9)')
    sweep_parser.add_argument('--delta-factor', type=float, default=1.1, help='Delta increase factor (default: 1.1)')
    sweep_parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability (default: 0.1)')
    sweep_parser.add_argument('--workers', type=int, default=0, help='Number of parallel workers (default: 0 = auto)')
    sweep_parser.set_defaults(func=sweep_command)

    # ratchet
    ratchet_parser = subparsers.add_parser('ratchet', help='Phase 4: Barrier refinement')
    add_common_args(ratchet_parser)
    add_discretization_args(ratchet_parser, required=False)
    add_energy_args(ratchet_parser)
    add_from_arg(ratchet_parser, 'Load xi/delta/ceiling from sweep or sample result JSON')
    ratchet_parser.add_argument('--ceiling', type=float, help='Initial energy ceiling (eV)')
    ratchet_parser.add_argument('--paths-per-round', type=int, default=10, help='Paths per round (default: 10)')
    ratchet_parser.add_argument('--max-rounds', type=int, default=20, help='Max rounds (default: 20)')
    ratchet_parser.add_argument('--dropout', type=float, default=0.1, help='Initial dropout (default: 0.1)')
    ratchet_parser.add_argument('--min-dropout', type=float, default=0.1, help='Minimum dropout (default: 0.1)')
    ratchet_parser.set_defaults(func=ratchet_command)

    # find-barrier (full workflow)
    fb_parser = subparsers.add_parser('find-barrier', help='Run full 4-phase barrier search')
    fb_parser.add_argument('start', help='Start structure CIF file')
    fb_parser.add_argument('end', help='End structure CIF file')
    fb_parser.add_argument('--output', '-o', help='Output directory')
    fb_parser.add_argument('-v', '--verbose', action='count', default=0,
                           help='Increase verbosity (-v for A* progress, -vv for more)')
    fb_parser.add_argument('-q', '--quiet', action='store_true', help='Silent mode')
    fb_parser.add_argument('--model', help='Sample/Sweep/Ratchet: Path to local GRACE model')
    # Phase 1 args
    fb_parser.add_argument('--xi', help='Search: Comma-separated xi values (default: 1.5,1.25,1.0,0.75)')
    fb_parser.add_argument('--atom-steps', help='Search: Comma-separated atom step lengths in Å (default: 0.4,0.3,0.2,0.1)')
    # Common search args
    fb_parser.add_argument('--beam-width', type=int, default=1000, help='All phases: Max open-set size (default: 1000)')
    fb_parser.add_argument('--max-iters', type=int, default=None, help='All phases: Max A* iterations cap (default: no cap, uses learned estimates)')
    fb_parser.add_argument('--dropout', type=float, default=0.1, help='Search/Ratchet: Dropout probability (default: 0.1)')
    fb_parser.add_argument('--workers', type=int, default=0, help='Phases 1-3: Number of parallel workers (default: 0=auto)')
    # Phase 2 args
    fb_parser.add_argument('--num-samples', type=int, default=20, help='Sample: Number of path samples (default: 20)')
    fb_parser.add_argument('--dropout-min', type=float, default=0.05, help='Sample: Min dropout (default: 0.05)')
    fb_parser.add_argument('--dropout-max', type=float, default=0.1, help='Sample: Max dropout (default: 0.1)')
    # Phase 3 args
    fb_parser.add_argument('--num-ceilings', type=int, default=5, help='Sweep: Number of ceiling levels (default: 5)')
    fb_parser.add_argument('--attempts', type=int, default=3, help='Sweep: A* searches per ceiling for diversity (default: 3)')
    fb_parser.add_argument('--max-passes', type=int, default=3, help='Sweep: Resolution refinement passes (default: 3)')
    fb_parser.add_argument('--sweep-dropout', type=float, default=0.1, help='Sweep: Dropout probability (default: 0.1)')
    fb_parser.add_argument('--xi-factor', type=float, default=0.9, help='Sweep: Xi reduction per pass (default: 0.9)')
    # Phase 4 args
    fb_parser.add_argument('--paths-per-round', type=int, default=10, help='Ratchet: A* searches per round (default: 10)')
    fb_parser.add_argument('--max-rounds', type=int, default=20, help='Ratchet: Max refinement rounds (default: 20)')
    fb_parser.set_defaults(func=find_barrier_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
