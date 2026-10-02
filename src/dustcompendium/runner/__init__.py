"""Running a campaign's models, here or on a cluster."""

from .base import Job, JobResult, Resources, Scheduler
from .local import LocalScheduler
from .slurm import SlurmScheduler
from .solver import SOLVERS, is_solved, solver_command, wait_for_solved, which_solver

#: Schedulers by the name a configuration or the command line uses.
SCHEDULERS: dict[str, type[Scheduler]] = {
    "local": LocalScheduler,
    "slurm": SlurmScheduler,
}


def scheduler(name: str, **options: object) -> Scheduler:
    """Build a scheduler by name.

    Raises
    ------
    KeyError
        If the name is not a known scheduler. The message lists those that are.
    """
    try:
        factory = SCHEDULERS[name]
    except KeyError:
        known = ", ".join(sorted(SCHEDULERS))
        raise KeyError(f"unknown scheduler {name!r}; known schedulers are {known}") from None
    return factory(**options)  # type: ignore[arg-type]


__all__ = [
    "SCHEDULERS",
    "SOLVERS",
    "Job",
    "JobResult",
    "LocalScheduler",
    "Resources",
    "Scheduler",
    "SlurmScheduler",
    "is_solved",
    "scheduler",
    "solver_command",
    "wait_for_solved",
    "which_solver",
]
