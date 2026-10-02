r"""Locating and invoking the Hyperion solver binaries.

These are Fortran programs which ``pip`` does not install -- neither the wheel
nor the sdist carries them -- so they have to be built from the Hyperion source
and put on ``PATH``. See the installation notes in the README.
"""

import shutil
import time
from collections.abc import Sequence
from pathlib import Path

__all__ = ["SOLVERS", "is_solved", "solver_command", "wait_for_solved", "which_solver"]

#: Solver binaries by grid geometry, serial and MPI. The models here are
#: cylindrical, so ``cylindrical`` is the one that matters.
SOLVERS = {
    "cylindrical": ("hyperion_cyl", "hyperion_cyl_mpi"),
    "cartesian": ("hyperion_car", "hyperion_car_mpi"),
    "spherical": ("hyperion_sph", "hyperion_sph_mpi"),
}


def which_solver(geometry: str = "cylindrical", parallel: bool = False) -> str:
    """Find a solver binary on ``PATH``.

    Raises
    ------
    KeyError
        If the geometry is not one Hyperion provides.
    FileNotFoundError
        If the binary is not on ``PATH``, with a pointer to the build notes.
    """
    try:
        serial, mpi = SOLVERS[geometry]
    except KeyError:
        known = ", ".join(sorted(SOLVERS))
        raise KeyError(f"unknown geometry {geometry!r}; known geometries are {known}") from None
    name = mpi if parallel else serial
    found = shutil.which(name)
    if found is None:
        raise FileNotFoundError(
            f"{name} is not on PATH. The Hyperion solvers are Fortran binaries which pip "
            "does not install; build them from the Hyperion source and add their bin "
            "directory to PATH. See the installation notes in the README."
        )
    return found


def solver_command(
    source: Path,
    result: Path,
    geometry: str = "cylindrical",
    tasks: int = 1,
    launcher: Sequence[str] = ("mpirun", "-np"),
    overwrite: bool = True,
) -> list[str]:
    """The command which solves one model.

    Parameters
    ----------
    source
        The model input file, as written by
        :func:`~dustcompendium.hyperion_model.write_model`.
    result
        Where the solved model should go.
    geometry
        Which solver to use; the models here are ``cylindrical``.
    tasks
        Number of MPI tasks. One uses the serial solver and no launcher.
    launcher
        How to start an MPI job, as a prefix taking the task count.
    overwrite
        Pass the solver's ``-f``, so that an existing output is replaced rather
        than the run refusing.
    """
    if tasks < 1:
        raise ValueError(f"tasks must be at least one, got {tasks}")
    binary = which_solver(geometry, parallel=tasks > 1)
    prefix = [*launcher, str(tasks)] if tasks > 1 else []
    flags = ["-f"] if overwrite else []
    return [*prefix, binary, *flags, str(source), str(result)]


def is_solved(path: Path) -> bool:
    """Whether a solved model actually holds the SEDs it was meant to produce.

    An exit status of zero is not enough. Hyperion aborts on some conditions --
    a photon at a frequency its dust is not defined at, for one -- by printing
    an error and stopping, and *still exits zero*, leaving an output file whose
    peeled group is empty. Taking the status at face value would record such a
    model as solved, skip it on every later run, and only surface it as a gap
    when the tabulation was assembled.
    """
    import h5py

    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as handle:
            groups = handle.get("Peeled")
            return bool(groups is not None and len(groups) > 0)
    except OSError:
        # A truncated or unreadable file is not a solved one.
        return False


def wait_for_solved(paths: Sequence[Path], attempts: int = 31, interval: float = 1.0) -> set[Path]:
    """Wait briefly for completed outputs to become visible.

    A scheduler can report a compute job complete before a shared filesystem's
    metadata cache on the submitting node exposes the groups just written to
    its HDF5 output. Check all paths together, so even a genuinely aborted
    batch costs at most one retry window rather than one window per model.

    Returns the subset of ``paths`` which contain SEDs before the retry window
    expires. The default allows roughly thirty seconds for shared-filesystem
    visibility while preserving :func:`is_solved` as a cheap, immediate check
    when scanning an existing campaign.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be at least one, got {attempts}")
    if interval < 0.0:
        raise ValueError(f"interval cannot be negative, got {interval}")

    pending = {Path(path) for path in paths}
    solved: set[Path] = set()
    for attempt in range(attempts):
        newly_solved = {path for path in pending if is_solved(path)}
        solved.update(newly_solved)
        pending.difference_update(newly_solved)
        if not pending or attempt + 1 == attempts:
            break
        time.sleep(interval)
    return solved
