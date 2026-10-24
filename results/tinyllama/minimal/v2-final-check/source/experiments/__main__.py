"""List, inspect or run a registered experiment: python -m experiments --help."""
import argparse
import json
import os

from .workspace import Workspace
from .process import run_process


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="list registered experiments without ML dependencies")
    run = sub.add_parser("run", help="run a suite; remaining arguments pass to its existing entrypoint")
    run.add_argument("experiment")
    run.add_argument("--dry-run", action="store_true", help="print command and paths without creating files")
    status = sub.add_parser("status", help="read latest or selected run without using a GPU")
    status.add_argument("experiment")
    status.add_argument("--run-dir")
    args, remaining = parser.parse_known_args(argv)
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    if args.action != "run" and remaining:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")
    try:
        workspace = Workspace.discover()
        registry = workspace.registry()
        if args.action == "list":
            print(json.dumps(registry, ensure_ascii=False, indent=2))
            return 0
        if args.experiment not in registry:
            parser.error(f"unknown experiment: {args.experiment}; choose from {', '.join(registry)}")
        entry = registry[args.experiment]
        if args.action == "status":
            print(json.dumps(workspace.status(args.experiment, entry, args.run_dir), ensure_ascii=False, indent=2))
            return 0
        command = workspace.command(entry, remaining)
        environment = workspace.environment()
        if args.dry_run:
            print(json.dumps({"command": command, "cwd": str(workspace.repo), "environment": environment},
                             ensure_ascii=False, indent=2))
            return 0
        return run_process(command, cwd=workspace.repo, env={**os.environ, **environment})
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
