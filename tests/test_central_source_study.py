"""Analytic invariants and source construction for the study-only inner profiles."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

HERE = Path(__file__).resolve().parents[1] / "studies/tied-central-source"
spec = importlib.util.spec_from_file_location("central_physics", HERE / "physics.py")
physics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(physics)


@pytest.mark.parametrize("kind", ["cavity", "core"])
@pytest.mark.parametrize("epsilon", [0.003, 0.01])
def test_mass_and_central_column(kind, epsilon):
    p = physics.InnerHernquist(0.1, kind, epsilon)
    target = 2 * np.pi * 0.1**3 * (10 / 11) ** 2
    numerical = quad(
        lambda r: 4 * np.pi * r * r * float(p.density(r, 0)), 0, 1.0, points=[0.1 * epsilon], epsabs=1e-13
    )[0]
    assert numerical == pytest.approx(target, rel=1e-9)
    column = quad(lambda r: float(p.density(r, 0)), 0, 1.0, points=[0.1 * epsilon], epsabs=1e-12)[0]
    assert column == pytest.approx(p.central_column(), rel=1e-9)
    g = physics.galaxy(0.1, kind, epsilon)
    grid = physics.grid_for(g, "inner16")
    mass = p.cell_mass(*grid.bounds)
    assert np.min(mass) >= 0
    assert np.sum(mass) == pytest.approx(target, rel=1e-9)
    # For a whole sphere-enclosing cylinder, integration conserves the same mass.
    assert p.cell_mass(0.0, 1.0, -1.0, 1.0) == pytest.approx(target, rel=1e-9)


@pytest.mark.parametrize("q", [0.01, 0.1, 1, 10])
def test_refinement_retains_outer_walls_and_resolves_both_axes(q):
    g = physics.galaxy(q, "cavity", 0.003)
    base = physics.grid_for(g, "legacy")
    fine = physics.grid_for(g, "inner16")
    assert np.all(np.isin(base.radial_walls, fine.radial_walls))
    assert np.all(np.isin(base.vertical_walls, fine.vertical_walls))
    assert fine.radial_walls[1] / q == pytest.approx(0.003 / 16)
    assert fine.vertical_walls[fine.vertical_walls > 0][0] / q == pytest.approx(0.003 / 16)
    p = g["spheroid"].dust
    rho = p.cell_mass(*fine.bounds) / fine.cell_volumes
    for i in [0, 15, 45, 75, 90]:
        column = physics.central_grid_column(rho, fine, i)
        assert abs(column / p.central_column() - 1) < 0.01


def test_mass_factor_and_finite_core_versus_cavity():
    core = physics.InnerHernquist(1.0, "core", 0.01)
    cavity = physics.InnerHernquist(1.0, "cavity", 0.01)
    assert 1 < core.mass_factor < cavity.mass_factor < 1.0002
    assert np.isfinite(core.density(0.0, 0.0))
    assert cavity.density(0.0, 0.0) == 0
    assert core.central_column() > cavity.central_column()
    assert np.isinf(physics.InnerHernquist(1.0).central_column())


def load_study():
    old = sys.modules.get("physics")
    sys.modules["physics"] = physics
    try:
        spec = importlib.util.spec_from_file_location("central_study", HERE / "study.py")
        study = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(study)
    finally:
        if old is None:
            sys.modules.pop("physics", None)
        else:
            sys.modules["physics"] = old
    return study


def test_task_design_pairs_all_sources_and_seeds():
    study = load_study()
    tasks = study.task_list()
    assert len(tasks) == 702 and len({t["id"] for t in tasks}) == 702
    for case in ["compact", "small", "matched", "extended", "sphere"]:
        selected = [t for t in tasks if t["case"] == case]
        assert len(selected) == 5 * 3 * 3 * 3
    assert all(t["kind"] == "cusp" for t in tasks if t["case"] == "disk_control")


def test_point_source_is_not_a_small_map_source():
    pytest.importorskip("hyperion")
    from dustcompendium.dust import load_dust

    grain = Path("/Users/arobertson/Downloads/d03_4.0_4.0_A.hdf5")
    if not grain.exists():
        pytest.skip("local audited grain resource unavailable")
    study = load_study()
    task = next(t for t in study.task_list() if t["emitter"] == "central")
    design = {"wavelengths": [0.55], "inclinations": [0.0, 90.0], "photons": 10}
    model = study.build(task, design, load_dust(str(grain)))
    assert len(model.sources) == 1
    assert model.sources[0].__class__.__name__ == "PointSource"
    assert model.sources[0].position == (0.0, 0.0, 0.0)


def test_submit_pilot_as_one_core_array_with_export_dependency(tmp_path, monkeypatch):
    import argparse
    import json
    import subprocess

    study = load_study()
    study.write_json(tmp_path / "design.json", dict(tasks=study.task_list(), files={}, python=sys.executable))
    calls = []
    active = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "scontrol":
            stdout = "MaxArraySize = 10001\nSchedulerParameters = max_array_tasks=1000\n"
        elif argv[0] == "squeue":
            stdout = "\n".join(active)
        elif argv[0] == "sbatch":
            stdout = str(100 + sum(a[0] == "sbatch" for a in calls))
        else:
            raise AssertionError(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=stdout)

    monkeypatch.setattr(study.subprocess, "run", fake_run)
    args = argparse.Namespace(runs=tmp_path, pilot=True, concurrency=12, partition="obs", walltime="02:00:00")
    study.submit(args)
    submissions = list((tmp_path / "submissions").glob("submit-*"))
    assert len(submissions) == 1
    folder = submissions[0]
    receipt = json.loads((folder / "receipt.json").read_text())
    assert receipt["state"] == "submitted" and receipt["count"] == 12
    assert len(json.loads((folder / "mapping.json").read_text())) == 12
    batches = [a for a in calls if a[0] == "sbatch"]
    assert len(batches) == 2
    assert "--array=0-11%12" in batches[0]
    assert "--ntasks=1" in batches[0] and "--cpus-per-task=1" in batches[0]
    assert "--dependency=afterany:101" in batches[1]
    assert "${SLURM_ARRAY_TASK_ID:?" in (folder / "tasks.sh").read_text()
    assert "--allow-partial" in (folder / "export.sh").read_text()
    active.extend(receipt["job_ids"])
    with pytest.raises(ValueError, match="still active"):
        study.submit(args)


def test_paired_statistics_keep_opaque_cells_and_sign():
    spec = importlib.util.spec_from_file_location("central_report", HERE / "report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    b = np.array([[[1.0, 1e-7]], [[1.1, 2e-7]], [[0.9, 3e-7]]])
    a = 0.9 * b
    dt, da, _set, sea, keep = report.compare(a, b)
    assert da[0, 0] == pytest.approx(-2.5 * np.log10(0.9))
    assert sea[0, 0] == pytest.approx(0.0, abs=1e-14)
    assert not keep[0, 1] and np.isnan(da[0, 1])
    assert dt[0, 1] == pytest.approx(-2e-8)
