"""Felix CLI entry point.

Usage:
    felix init <name> [--template research|analysis|review]
    felix run <config.yaml> [--provider NAME] [--verbose]
    felix version
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``felix`` console command."""
    parser = argparse.ArgumentParser(
        prog="felix",
        description="Felix Agent SDK — helical multi-agent orchestration",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- init ---
    init_p = subparsers.add_parser("init", help="Scaffold a new Felix project")
    init_p.add_argument("name", help="Project directory name")
    init_p.add_argument(
        "--template", "-t",
        default="research",
        choices=["research", "analysis", "review"],
        help="Workflow template (default: research)",
    )

    # --- run ---
    run_p = subparsers.add_parser("run", help="Run a workflow from YAML config")
    run_p.add_argument("config", help="Path to felix.yaml")
    run_p.add_argument("--provider", "-p", default=None, help="Override provider name")
    run_p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    # --- version ---
    subparsers.add_parser("version", help="Show SDK version")

    args = parser.parse_args(argv)

    if args.command == "init":
        from felix_agent_sdk.cli.init_command import run_init

        return run_init(args.name, args.template)

    if args.command == "run":
        from felix_agent_sdk.cli.run_command import run_workflow

        return run_workflow(args.config, args.provider, args.verbose)

    if args.command == "version":
        from felix_agent_sdk._version import __version__

        print(f"felix-agent-sdk {__version__}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
