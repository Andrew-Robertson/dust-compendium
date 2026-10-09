"""Array submission is independent of the scientific worker and live Slurm."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "studies/tied-emulator-budget"


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pilot = load("pilot")
arrays = load("arrays")


@pytest.mark.parametrize(
    "config, expected",
    [
        ("MaxArraySize = 10001\nMaxJobCount = 250000\nSchedulerParameters = preempt_strict_order", 10001),
        ("MaxArraySize = 1001", 1001),
        ("MaxArraySize = 100001\nSchedulerParameters = foo,max_array_tasks=1000,bar", 1000),
        ("MaxArraySize = 100\nSchedulerParameters = max_array_tasks=200", 100),
        ("MaxArraySize = 1", 1),
    ],
)
def test_limits(config, expected):
    assert arrays.array_limit(config) == expected


@pytest.mark.parametrize(
    "config",
    [
        "",
        "MaxArraySize = unknown",
        "MaxArraySize = 0",
        "MaxArraySize = 10\nSchedulerParameters = max_array_tasks=0",
    ],
)
def test_unknown_or_disabled_limit_refused(config):
    with pytest.raises(ValueError):
        arrays.array_limit(config)


def arguments(tmp_path):
    manifest = tmp_path / "original.json"
    manifest.write_text('{"test": true}\n')
    return SimpleNamespace(
        manifest=manifest,
        runs=tmp_path / "runs with spaces",
        grains=tmp_path / "dust file.hdf5",
        concurrency=96,
        partition="obs",
        walltime="08:00:00",
    )


def test_prepare_frozen_maps_chunking_and_snapshot(tmp_path):
    args = arguments(tmp_path)
    folder, receipt = arrays.prepare(
        args, {"training": [f"job-{i}" for i in range(7)], "validation": ["v0"]}, 3, "MaxArraySize = 3"
    )
    assert [j["count"] for j in receipt["plan"]] == [3, 3, 1, None, 1, None]
    assert [j["label"] for j in receipt["plan"]] == [
        "training-000",
        "training-001",
        "training-002",
        "training-export",
        "validation-000",
        "validation-export",
    ]
    mapping = json.loads((folder / "training-001.json").read_text())
    assert mapping["tasks"] == ["job-3", "job-4", "job-5"]
    assert mapping["command"][0] == sys.executable
    assert str(folder / "pilot.py") in mapping["command"]
    assert str(folder / "pilot.json") in mapping["command"]
    assert "--runs" in mapping["command"]
    assert (folder / "pilot.py").read_bytes() == (HERE / "pilot.py").read_bytes()
    args.manifest.write_text("modified original")
    assert pilot.sha(folder / "pilot.json") == receipt["manifest_sha256"]
    for item in receipt["plan"]:
        subprocess.run(["bash", "-n", item["script"]], check=True)
    script = (folder / "training-001.sh").read_text()
    assert "OMP_NUM_THREADS=1" in script and '"${SLURM_ARRAY_TASK_ID:' in script


def test_submission_chain_cap_exports_and_receipt(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    folder, receipt = arrays.prepare(args, {"training": ["a", "b", "c"], "validation": ["v"]}, 2, "")
    calls = []

    def fake(argv):
        calls.append(argv)
        return SimpleNamespace(stdout=f"{100 + len(calls)};obs\n")

    monkeypatch.setattr(arrays, "command", fake)
    arrays.submit_plan(folder, receipt, args.runs)
    assert len(calls) == 5
    assert not any(a.startswith("--dependency") for a in calls[0])
    for i, argv in enumerate(calls):
        assert "--cpus-per-task=1" in argv and "--ntasks=1" in argv
        if i:
            assert f"--dependency=afterany:{100 + i}" in argv
    assert "--array=0-1%96" in calls[0]
    assert "--array=0-0%96" in calls[1]
    assert "--time=08:00:00" in calls[0]
    assert "--time=00:30:00" in calls[2]
    assert not any(a.startswith("--array") for a in calls[2])
    final = json.loads((folder / "receipt.json").read_text())
    assert final["state"] == "submitted"
    assert [j["job_id"] for j in final["jobs"]] == [str(i) for i in range(101, 106)]


def test_submit_failure_retains_accepted_jobs_without_retry(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    folder, receipt = arrays.prepare(args, {"training": ["a", "b", "c"]}, 2, "")
    calls = []

    def fake(argv):
        calls.append(argv)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(1, argv, stderr="MaxSubmitJobs limit")
        return SimpleNamespace(stdout="123\n")

    monkeypatch.setattr(arrays, "command", fake)
    with pytest.raises(RuntimeError, match="Already accepted jobs continue"):
        arrays.submit_plan(folder, receipt, args.runs)
    final = json.loads((folder / "receipt.json").read_text())
    assert final["state"] == "rejected" and final["jobs"][0]["job_id"] == "123"
    assert len(calls) == 2


def test_ambiguous_submission_blocks_retry(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    folder, receipt = arrays.prepare(args, {"training": ["a"]}, 2, "")
    monkeypatch.setattr(arrays, "command", lambda argv: SimpleNamespace(stdout="unexpected response"))
    with pytest.raises(RuntimeError, match="Unrecognized sbatch"):
        arrays.submit_plan(folder, receipt, args.runs)
    with pytest.raises(ValueError, match="Unconfirmed submission"):
        arrays.ensure_idle(args.runs)


def test_active_prior_submission_blocks_duplicate(tmp_path, monkeypatch):
    path = tmp_path / "arrays" / "old" / "receipt.json"
    path.parent.mkdir(parents=True)
    arrays.write_json(path, {"state": "submitted", "jobs": [{"job_id": "123"}]})
    monkeypatch.setattr(arrays, "command", lambda argv: SimpleNamespace(stdout="122\n123\n123\n"))
    with pytest.raises(ValueError, match="still active: 123"):
        arrays.ensure_idle(tmp_path)
    monkeypatch.setattr(arrays, "command", lambda argv: SimpleNamespace(stdout="122\n"))
    arrays.ensure_idle(tmp_path)


def test_resume_validates_existing_results(tmp_path):
    design = {"many_count": 1, "precise_count": 1, "train": [{"id": "p0"}], "inclinations": [0, 90]}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(design))
    task = pilot.tasks(design, "training")[0]
    result = tmp_path / "tasks" / pilot.task_id(task) / "result.npz"
    result.parent.mkdir(parents=True)
    metadata = {"manifest_sha256": pilot.sha(manifest), "task": pilot.task_id(task)}
    np.savez(result, metadata=json.dumps(metadata), transmission=[1.0, 1.0])
    pending, done = arrays.unfinished(design, manifest, tmp_path, "training")
    assert done == 1 and len(pending) == 3 and pilot.task_id(task) not in pending
    np.savez(result, metadata=json.dumps(metadata), transmission=[1.0])
    with pytest.raises(ValueError, match="inclination"):
        arrays.unfinished(design, manifest, tmp_path, "training")
    metadata["manifest_sha256"] = "stale"
    np.savez(result, metadata=json.dumps(metadata), transmission=[1.0, 1.0])
    with pytest.raises(ValueError, match="stale"):
        arrays.unfinished(design, manifest, tmp_path, "training")


def test_dispatch_exact_task_and_bounds(tmp_path, monkeypatch):
    mapping = tmp_path / "map.json"
    mapping.write_text(json.dumps({"command": ["python with spaces", "worker.py"], "tasks": ["a", "b"]}))
    seen = []
    monkeypatch.setattr(arrays.os, "execvpe", lambda exe, argv, env: seen.append((exe, argv)))
    arrays.dispatch(mapping, 1)
    assert seen == [("python with spaces", ["python with spaces", "worker.py", "--task", "b"])]
    for index in (-1, 2):
        with pytest.raises(ValueError, match="outside frozen mapping"):
            arrays.dispatch(mapping, index)


def test_generated_array_script_executes_mapping_with_spaces(tmp_path):
    # Exercise actual Bash -> array dispatcher -> worker, without Slurm or Hyperion.
    args = arguments(tmp_path)
    folder, _ = arrays.prepare(args, {"training": ["first", "second"]}, 10, "")
    worker = folder / "dummy worker.py"
    worker.write_text("import sys\nprint(sys.argv[1:])\n")
    mapping = folder / "training-000.json"
    mapping.write_text(json.dumps({"command": [sys.executable, str(worker)], "tasks": ["first", "second"]}))
    import os

    result = subprocess.run(
        ["bash", str(folder / "training-000.sh")],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "SLURM_ARRAY_TASK_ID": "1"},
    )
    assert result.stdout.strip() == "['--task', 'second']"


def test_no_missing_tasks_still_queues_export(tmp_path):
    args = arguments(tmp_path)
    _, receipt = arrays.prepare(args, {"training": []}, 10, "")
    assert len(receipt["plan"]) == 1
    assert receipt["plan"][0]["label"] == "training-export"


@pytest.mark.hyperion
def test_generated_scripts_run_real_analytic_workers_and_export(tmp_path):
    """End-to-end dry dust-free control: real worker/config/result/export, no RT solver."""
    import os

    args = arguments(tmp_path)
    design = json.loads((HERE / "pilot.json").read_text())
    design["train"] = [next(p for p in design["train"] if p["x"][1:3] == [0.0, 0.0])]
    design["many_count"] = design["precise_count"] = 1
    # The analytic worker hashes this file, but does not load grain optical data.
    args.grains.write_bytes(b"analytic test grain resource")
    design["grain_sha256"] = pilot.sha(args.grains)
    args.manifest.write_text(json.dumps(design))
    pending, _ = arrays.unfinished(design, args.manifest, args.runs, "training")
    folder, receipt = arrays.prepare(args, {"training": pending}, 2, "MaxArraySize = 2")
    for item in receipt["plan"]:
        for index in range(item["count"] or 1):
            subprocess.run(
                ["bash", item["script"]],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "SLURM_ARRAY_TASK_ID": str(index)},
            )
    with np.load(args.runs / "compact/training.npz", allow_pickle=False) as result:
        np.testing.assert_array_equal(result["transmission"], np.ones((4, 19)))
        metadata = json.loads(str(result["metadata"]))
    assert metadata["manifest_sha256"] == pilot.sha(args.manifest)
    assert all(r["worker_sha256"] == pilot.sha(HERE / "pilot.py") for r in metadata["records"])
    assert arrays.unfinished(design, args.manifest, args.runs, "training") == ([], 4)
    assert (folder / "pilot.py").read_bytes() == (HERE / "pilot.py").read_bytes()
