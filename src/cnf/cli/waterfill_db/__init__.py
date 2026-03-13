"""Waterfill database CLI commands.

Commands:
    setup       Set up a partitioned database for waterfill search
    status      Show database statistics
    run         Run waterfill search on the database
    check       Check if search has reached endpoints
"""

from cnf.cli.waterfill_db import setup, status, run, check


def register_subparsers(parent_subparsers):
    """Register waterfill-db subcommand with its own subparsers."""
    wf_parser = parent_subparsers.add_parser(
        'waterfill-db',
        help='Database-backed waterfill search commands'
    )
    wf_subparsers = wf_parser.add_subparsers(dest='wf_command', required=True)

    setup.register_parser(wf_subparsers)
    status.register_parser(wf_subparsers)
    run.register_parser(wf_subparsers)
    check.register_parser(wf_subparsers)
