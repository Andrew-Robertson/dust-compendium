"""Frozen sampling, job provenance, budget matching, and end-to-end software checks."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "studies/tied-emulator-budget"


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


design_module = module("design")
pilot = module("pilot")
emulate = module("emulate")


def design():
    rng = np.random.default_rng(15)
    pool = np.exp(rng.uniform(-3, 1, (100, 3)))
    d = design_module.make_design(pool, np.arange(100), {}, lambda w: 0.55 / w, "test", many=64, precise=32)
    d["validation"] = d["validation"][:6]
    return d


def test_nested_reproducible_design_and_disjoint_validation():
    a, b = design(), design()
    assert a == b
    train = np.array([p["x"] for p in a["train"]])
    assert len(set(map(tuple, train))) == len(train)
    assert not set(map(tuple, train)) & {tuple(p["x"]) for p in a["validation"]}
    assert len({tuple(p["x"]) for p in a["train"][:32]}) == 32
    assert np.sum(np.all(train[:, 1:3] == 0, axis=1)) == 4
    assert np.ptp(train, axis=0).min() > 0
    assert len(pilot.tasks(a, "timing")) == 64
    assert len(pilot.tasks(a, "training")) == 192
    assert len(pilot.tasks(a, "validation")) == 36
    assert {pilot.task_id(t) for t in pilot.tasks(a, "timing")} <= {
        pilot.task_id(t) for t in pilot.tasks(a, "training")
    }


def test_point_configuration_no_cartesian_expansion_and_seed_isolation():
    d = design()
    seeds = []
    for task in pilot.tasks(d, "training")[:4]:
        campaign = pilot.Campaign(pilot.campaign_config(d, task, Path("dust.hdf5")))
        assert len(campaign) == 2
        assert campaign.wavelengths.tolist() == [task[1]["x"][0]]
        assert campaign.inclinations.size == 19
        for run in campaign.runs():
            assert run.spec.optical_depths == {"disk": task[1]["x"][1], "spheroid": task[1]["x"][2]}
        seeds.append(next(r.spec.seed for r in campaign.runs() if r.emitter == task[2]))
    assert len(set(seeds)) == len(seeds)
    task = pilot.tasks(d, "training")[0]
    c = pilot.campaign_config(d, task, Path("dust.hdf5"))
    raw = c.model_dump(mode="json")
    for component in raw["geometry"]["components"].values():
        component["dust"]["optical_depth"] = 0
    clear = pilot.Campaign(pilot.CampaignConfig.model_validate(raw))
    assert all(set(r.spec.optical_depths.values()) == {0.0} for r in clear.runs())


def test_matched_budget_not_photon_ratio():
    n, r = emulate.matched_count(np.ones(64), np.full(32, 1.5), 32)
    assert n == 48 and r == 1
    with pytest.raises(ValueError, match="no matched"):
        emulate.matched_count(np.ones(64), np.full(32, 10), 32)


def test_features_keep_wavelength_and_zero_depths():
    p = [{"x": [0.2, 0, 1, 0.1], "opacity_ratio": 2}, {"x": [0.4, 0, 2, 0.1], "opacity_ratio": 1}]
    f = emulate.features(p)
    assert f[0, 0] != f[1, 0]
    np.testing.assert_allclose(f[0, 1:], f[1, 1:])
    assert np.isfinite(f).all()


def test_signed_metric_and_negative_prediction_accounting():
    ref = np.array([[0.5, 0.5]])
    m = emulate.diagnostics(ref * 0.99, ref, np.zeros_like(ref), np.ones(2) / 2)
    assert m["E_A"]["mean"] == pytest.approx(0.01 * emulate.FACTOR)
    assert m["delta_A_Tref_gt_0p01"]["mean"] > 0
    bad = emulate.diagnostics(-ref, ref, np.zeros_like(ref), np.ones(2) / 2)
    assert bad["negative_predictions"] == 2
    assert bad["invalid_mag_at_bright_reference"] == 2
    weights = emulate.angle_weights(np.linspace(0, 90, 19))
    assert weights.sum() == pytest.approx(1)
    assert (weights > 0).all()


def synthetic_export(tmp_path, d, stage):
    records, t, cpu = [], [], []
    angles = np.deg2rad(d["inclinations"])
    for allocation, point, emitter, rep in pilot.tasks(d, stage):
        task = (allocation, point, emitter, rep)
        wave, td, ts, _q = point["x"]
        transmission = 0.15 + 0.8 * np.exp(-0.1 * (td + ts) / (wave + 1) * (1 + 0.2 * np.sin(angles)))
        t.append(transmission * (1 + (rep - 1) * 0.001))
        cpu.append(1.5 if allocation == "high" else 1)
        records.append(
            {
                "allocation": allocation,
                "point": point,
                "emitter": emitter,
                "replicate": rep,
                "task": pilot.task_id(task),
            }
        )
    path = tmp_path / f"{stage}.npz"
    np.savez(
        path,
        metadata=json.dumps({"stage": stage, "design": d, "records": records, "manifest_sha256": "test"}),
        transmission=t,
        compute_seconds=cpu,
    )
    return path


def test_full_synthetic_learning_pipeline_and_portable_weights(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("matplotlib")
    from threadpoolctl import threadpool_limits

    d = design()
    train = synthetic_export(tmp_path, d, "training")
    valid = synthetic_export(tmp_path, d, "validation")
    with threadpool_limits(limits=1):
        emulate.run(train, valid, tmp_path / "fit", max_iter=2)
    result = json.loads((tmp_path / "fit/results.json").read_text())
    assert result["matched_low_count"] == 48
    assert len(result["results"]) == 18
    with np.load(tmp_path / "fit/disk_high_precise_seed13.npz", allow_pickle=False) as f:
        model = dict(f)
    prediction = emulate.predict(model, emulate.features(d["validation"]))
    with np.load(tmp_path / "fit/disk_predictions.npz") as f:
        np.testing.assert_allclose(prediction, f["predictions"][6])
    meta, _t, _cpu = emulate.archive(train, "training")
    meta["records"][0] = meta["records"][1]
    with pytest.raises(ValueError, match="duplicate"):
        emulate.validate_records(meta, "training")


@pytest.mark.hyperion
@pytest.mark.solver
def test_real_worker_export_and_resume(tmp_path, solver_path):
    from dustcompendium.dust import ferrara

    grains = tmp_path / "dust.hdf5"
    ferrara.build("milkyWay").write(str(grains))
    d = design()
    d["grain_sha256"] = pilot.sha(grains)
    d["train"] = d["train"][:1]
    d["train"][0]["x"] = [0.55, 0.3, 0.1, 1.0]
    d["timing_count"] = 1
    d["geometry"]["cells"] = 12
    for budget in d["budgets"].values():
        budget.update(imaging=100, direct=100)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(d))
    pilot.submit(
        d,
        manifest,
        SimpleNamespace(
            stage="timing",
            runs=tmp_path / "runs",
            grains=grains,
            partition=None,
            walltime="00:01:00",
            execute=True,
            scheduler="local",
            concurrency=2,
        ),
    )
    for task in pilot.tasks(d, "timing"):
        result = tmp_path / "runs/tasks" / pilot.task_id(task) / "result.npz"
        before = result.stat().st_mtime_ns
        pilot.worker(d, manifest, task, tmp_path / "runs", grains)
        assert result.stat().st_mtime_ns == before
    output = tmp_path / "timing.npz"
    pilot.export(d, manifest, tmp_path / "runs", "timing", output)
    _meta, t, cpu = emulate.archive(output, "timing")
    assert t.shape == (4, 19)
    assert np.isfinite(t).all() and (t >= 0).all()
    assert (cpu > 0).all()
    with pytest.raises(ValueError, match="stale"):
        pilot.read_result(result, "different", task)
