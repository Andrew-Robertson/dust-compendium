"""Extension preserves the pilot, skips old tasks and pins geometry sources."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    here = Path(__file__).resolve().parents[1] / "studies/tied-emulator-budget"
    monkeypatch.syspath_prepend(str(here))
    return tuple(importlib.import_module(s) for s in ("extend", "arrays", "audit_geometry"))


def designs(ext, tmp_path):
    rng = np.random.default_rng(20)
    pool = np.exp(rng.uniform(-3, 1, (100, 3)))
    info = {"catalog_sha256": "fake-catalog"}
    old = ext.sampling.make_design(
        pool, np.arange(100), info, lambda w: 0.55 / w, "fake-grain", many=64, precise=32
    )
    new = ext.sampling.make_design(
        pool, np.arange(100), info, lambda w: 0.55 / w, "fake-grain", many=128, precise=32
    )
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(old))
    extra = ext.extension(old, new, ext.pilot.sha(path), 128)
    ep = tmp_path / "extra.json"
    ep.write_text(json.dumps(extra))
    return old, new, extra, path, ep


def test_extension_keeps_parent_and_seed_identity(modules, tmp_path):
    ext, _, _ = modules
    old, new, extra, path, _ = designs(ext, tmp_path)
    assert old == json.loads(path.read_text())
    assert old["train"] + extra["train"] == new["train"]
    assert extra["many_count"] == 64 and extra["precise_count"] == 0 and extra["validation"] == []
    old_ids = {ext.pilot.task_id(t) for t in ext.pilot.tasks(old, "training")}
    new_tasks = ext.pilot.tasks(extra, "training")
    assert len(new_tasks) == 128
    assert not old_ids & {ext.pilot.task_id(t) for t in new_tasks}
    for task in new_tasks[:2]:
        assert ext.pilot.campaign_config(extra, task, Path("x")) == ext.pilot.campaign_config(
            new, task, Path("x")
        )
    with pytest.raises(ValueError, match="training only"):
        ext.pilot.tasks(extra, "validation")
    new["train"][0]["x"][0] *= 1.01
    with pytest.raises(ValueError, match="prefix coordinates"):
        ext.extension(old, new, ext.pilot.sha(path), 128)


def test_submission_snapshots_and_checks_imported_sources(modules, tmp_path, monkeypatch):
    ext, arrays, _ = modules
    _, _, extra, _, ep = designs(ext, tmp_path)
    args = SimpleNamespace(
        manifest=ep,
        runs=tmp_path / "runs",
        grains=Path("dust"),
        concurrency=96,
        partition="obs",
        walltime="08:00:00",
    )
    folder, receipt = arrays.prepare(args, {"training": ["one", "two"]}, 10001, "")
    assert (folder / "source/dustcompendium/geometry/spheroid.py").exists()
    for item in receipt["plan"]:
        assert "export PYTHONPATH=" in Path(item["script"]).read_text()
        assert str(folder / "source") in Path(item["script"]).read_text()
    ext.pilot.verify_sources(extra)
    monkeypatch.setattr(ext.pilot, "source_fingerprints", lambda: {"changed": "source"})
    with pytest.raises(ValueError, match="source differs"):
        ext.pilot.verify_sources(extra)


def export_fake(ext, path, d, manifest, stage):
    records = []
    for task in ext.pilot.tasks(d, stage):
        records.append(
            dict(
                task=ext.pilot.task_id(task),
                point=task[1],
                allocation=task[0],
                emitter=task[2],
                replicate=task[3],
                manifest_sha256=ext.pilot.sha(manifest),
                runtime_source_sha256=d.get("runtime_source_sha256"),
            )
        )
    np.savez_compressed(
        path,
        metadata=json.dumps(
            dict(design=d, manifest_sha256=ext.pilot.sha(manifest), stage=stage, records=records)
        ),
        transmission=np.ones((len(records), 19)),
        compute_seconds=np.ones(len(records)),
        worker_wall_seconds=np.ones(len(records)),
    )


def test_merge_preserves_execution_provenance_and_rejects_wrong_exports(modules, tmp_path):
    ext, _, _ = modules
    old, _, extra, parent, ep = designs(ext, tmp_path)
    original = tmp_path / "original"
    original.mkdir()
    for stage in ("training", "validation"):
        export_fake(ext, original / f"{stage}.npz", old, parent, stage)
    more = tmp_path / "additional.npz"
    export_fake(ext, more, extra, ep, "training")
    args = SimpleNamespace(
        parent=parent, manifest=ep, original=original, extra=more, output=tmp_path / "merged"
    )
    ext.merge(args)
    with np.load(args.output / "training.npz") as f:
        m = json.loads(str(f["metadata"]))
        assert m["design"]["many_count"] == 128
        assert f["transmission"].shape == ((128 + 32) * 2, 19)
        assert len({r["task"] for r in m["records"]}) == (128 + 32) * 2
        assert {r["manifest_sha256"] for r in m["records"]} == {ext.pilot.sha(parent), ext.pilot.sha(ep)}
    with pytest.raises(ValueError, match="provenance"):
        ext.load_archive(more, old, ext.pilot.sha(parent), "training")


def test_pilot_geometry_avoids_v1_defects(modules):
    ext, _, audit = modules
    result = audit.audit(json.loads((ext.pilot.HERE / "pilot.json").read_text()))
    assert len(result["checks"]) == 6
    assert max(r["maximum_spheroid_cell_fraction"] for r in result["checks"]) < 0.01


def test_raw_cleanup_only_for_opted_in_extension_after_audit_sample(modules, tmp_path):
    ext, _, _ = modules
    old, _, extra, _, _ = designs(ext, tmp_path)
    folder = tmp_path / "task"
    folder.mkdir()
    names = (
        "dusty.hdf5",
        "dusty.rtout",
        "clear.hdf5",
        "clear.rtout",
        "result.npz",
        "request.json",
        "dusty.log",
    )
    for name in names:
        (folder / name).write_text("keep until verified")
    task = ext.pilot.tasks(extra, "training")[0]
    ext.pilot.cleanup_raw(extra, task, folder)
    assert all((folder / name).exists() for name in names)
    later = ext.pilot.tasks(extra, "training")[40]
    ext.pilot.cleanup_raw(old, later, folder)
    assert all((folder / name).exists() for name in names)
    ext.pilot.cleanup_raw(extra, later, folder)
    assert not any((folder / name).exists() for name in names[:4])
    assert all((folder / name).exists() for name in names[4:])
