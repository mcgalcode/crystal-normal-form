"""Unified CLI for CNF crystal structure pathfinding.

Usage:
    cnf search <start.cif> <end.cif> [options]
    cnf sample <start.cif> <end.cif> --from <search_result.json> [options]
    cnf sweep <start.cif> <end.cif> --from <sample_result.json> [options]
    cnf ratchet <start.cif> <end.cif> --from <sweep_result.json> [options]
    cnf find-barrier <start.cif> <end.cif> [options]
    cnf astar <start.cif> <end.cif> <output.json> [options]

    cnf waterfill-db setup <start.cif> <end.cif> <search_dir> [options]
    cnf waterfill-db status <partition_dir> [options]
    cnf waterfill-db run <partition_dir> [options]
    cnf waterfill-db check <partition_dir> [options]
"""

import argparse

from cnf.cli import barrier
from cnf.cli import waterfill_db


def main():
    parser = argparse.ArgumentParser(
        prog='cnf',
        description='CNF crystal structure pathfinding',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Register barrier search commands (search, sample, sweep, ratchet, astar, find-barrier)
    barrier.register_subparsers(subparsers)

    # Register waterfill-db commands (waterfill-db setup|status|run|check)
    waterfill_db.register_subparsers(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
