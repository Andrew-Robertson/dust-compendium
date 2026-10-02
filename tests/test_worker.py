"""Tests for execution-node Hyperion output validation."""

import subprocess
from pathlib import Path

from dustcompendium.runner.worker import INVALID_OUTPUT_STATUS, solve


def test_a_valid_output_preserves_the_solver_success(monkeypatch):
    monkeypatch.setattr("dustcompendium.runner.worker.solver_command", lambda *args, **kwargs: ["hyperion"])
    monkeypatch.setattr(
        "dustcompendium.runner.worker.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )
    monkeypatch.setattr("dustcompendium.runner.worker.is_solved", lambda path: True)
    assert solve(Path("in.hdf5"), Path("out.rtout")) == 0


def test_a_zero_exit_without_seds_becomes_a_failure(monkeypatch):
    monkeypatch.setattr("dustcompendium.runner.worker.solver_command", lambda *args, **kwargs: ["hyperion"])
    monkeypatch.setattr(
        "dustcompendium.runner.worker.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )
    monkeypatch.setattr("dustcompendium.runner.worker.is_solved", lambda path: False)
    assert solve(Path("in.hdf5"), Path("out.rtout")) == INVALID_OUTPUT_STATUS


def test_a_real_solver_failure_is_preserved(monkeypatch):
    monkeypatch.setattr("dustcompendium.runner.worker.solver_command", lambda *args, **kwargs: ["hyperion"])
    monkeypatch.setattr(
        "dustcompendium.runner.worker.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 7),
    )
    monkeypatch.setattr(
        "dustcompendium.runner.worker.is_solved",
        lambda path: (_ for _ in ()).throw(AssertionError("should not validate a failed solve")),
    )
    assert solve(Path("in.hdf5"), Path("out.rtout")) == 7
