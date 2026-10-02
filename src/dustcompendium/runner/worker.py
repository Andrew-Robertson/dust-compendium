"""Run one Hyperion solve and validate its output on the execution node.

Hyperion can exit with status zero after aborting before it writes any SEDs.
Checking the HDF5 output in this wrapper makes the scheduler's exit status
meaningful. It also avoids asking a submission node to validate a file while
its NFS metadata cache still holds the model's incomplete, in-flight state.
"""

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .solver import is_solved, solver_command

__all__ = ["INVALID_OUTPUT_STATUS", "main", "solve"]

#: Distinct from ordinary solver failures and the scheduler's negative statuses.
INVALID_OUTPUT_STATUS = 65


def solve(source: Path, result: Path, tasks: int = 1) -> int:
    """Run Hyperion, then require the resulting file to contain peeled SEDs."""
    try:
        completed = subprocess.run(solver_command(source, result, tasks=tasks), check=False)
    except OSError as error:
        print(f"failed to start Hyperion: {error}", file=sys.stderr)
        return INVALID_OUTPUT_STATUS
    if completed.returncode != 0:
        return completed.returncode
    if not is_solved(result):
        print(
            "Hyperion exited with status zero but its output contains no peeled SEDs",
            file=sys.stderr,
        )
        return INVALID_OUTPUT_STATUS
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point used inside local and Slurm jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--tasks", type=int, default=1)
    arguments = parser.parse_args(argv)
    return solve(arguments.source, arguments.result, tasks=arguments.tasks)


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
