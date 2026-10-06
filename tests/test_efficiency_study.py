"""Regression checks for the study tools, without requiring HPC outputs."""

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from dustcompendium.campaign import Campaign
from dustcompendium.config import TabulationConfig, load_campaign

STUDY = Path(__file__).resolve().parents[1] / "studies/tied-two-component-azimuths"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, STUDY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


efficiency = load_module("efficiency")
generator = load_module("make_efficiency_configs")


def test_stage_counts_propagate_and_old_default_is_preserved():
    config = load_campaign(str(STUDY.parents[1] / "configs/tied-two-component-smoke.yaml"))
    default = next(Campaign(config).runs()).spec
    assert default.photons == 10000
    assert default.imaging_photons is default.raytracing_photons is None
    updated = config.model_copy(
        update={"tabulation": TabulationConfig(imagingPhotons=12, raytracingPhotons=34)}
    )
    run = next(Campaign(updated).runs()).spec
    assert (run.imaging_photons, run.raytracing_photons) == (12, 34)


@pytest.mark.parametrize("field", ["imagingPhotons", "raytracingPhotons"])
def test_zero_stage_counts_rejected(field):
    with pytest.raises(ValueError):
        TabulationConfig(**{field: 0})


def test_efficiency_configs_match_the_completed_seed_families():
    base = yaml.safe_load((STUDY.parents[1] / "configs/tied-two-component-smoke.yaml").read_text())
    configs = list(generator.configurations(base))
    assert len(configs) == 15
    assert len({name for name, _ in configs}) == 15
    for name, config in configs:
        seed = int(name.rsplit("seed", 1)[1])
        assert config["tabulation"]["seed"] == -653 - 10000 * seed
        assert config["geometry"] == base["geometry"]
        assert config["tabulation"]["azimuths"] == [90.0]
    assert "imagingPhotons" not in base["tabulation"]


def test_cpu_time_is_not_missing_as_zero(tmp_path):
    path = tmp_path / "test.log"
    assert np.isnan(efficiency.cpu_seconds(path))
    path.write_text("Total CPU time elapsed: 3.14E+01\n")
    assert efficiency.cpu_seconds(path) == 31.4


def test_ratio_of_means_not_mean_of_ratios():
    flux = np.array([[[[[1.0]], [[3.0]]], [[[0.5]], [[0.3]]]]])
    np.testing.assert_allclose(efficiency.transmission(flux, [0, 0]).ravel(), [1, 0.2])


def test_correlated_azimuths_do_not_reduce_noise():
    rng = np.random.default_rng(731)
    flux = np.ones((2000, 2, 4, 1, 1))
    flux[:, 1] = 0.5 + rng.normal(0, 0.01, (2000, 1, 1, 1))
    s = efficiency.statistics(flux, 0.01 * np.ones_like(flux), [0, 0])
    assert s["azimuth_covariance_factor"][1, 0, 0] == pytest.approx(4)
    assert s["sigma_a"][1, 0, 0] == pytest.approx(efficiency.FACTOR * 0.01 / 0.5, rel=0.05)
    assert s["sigma_t"][0, 0, 0] == 0


def test_independent_azimuths_reduce_noise():
    rng = np.random.default_rng(732)
    flux = np.ones((10000, 2, 4, 1, 1))
    flux[:, 1] = 0.5 + rng.normal(0, 0.01, (10000, 4, 1, 1))
    s = efficiency.statistics(flux, 0.01 * np.ones_like(flux), [0, 0])
    assert s["azimuth_covariance_factor"][1, 0, 0] == pytest.approx(1, rel=0.05)


def test_report_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(733)
    paths = []
    for name, azimuths in (("az1", [90]), ("az2", [90, 270])):
        meta = {
            "variant": name,
            "geometry": {},
            "dust": {},
            "models": [
                {"emitter": e, "parameters": {"tau": tau}} for e in ("disk", "spheroid") for tau in (0, 1)
            ],
            "references": [0, 0, 2, 2],
            "inclinations": [0, 90],
            "wavelengths": [0.1, 0.55],
            "azimuths": azimuths,
        }
        flux = np.ones((5, 4, len(azimuths), 2, 2))
        flux[:, [1, 3]] = 0.5 + rng.normal(0, 0.01, flux[:, [1, 3]].shape)
        path = tmp_path / f"{name}.npz"
        np.savez(
            path,
            metadata=json.dumps(meta),
            flux=flux,
            reported=0.01 * np.ones_like(flux),
            cpu_seconds=np.ones((5, 4)) * len(azimuths),
            seeds=np.arange(5),
        )
        paths.append(path)
    efficiency.report(paths, tmp_path / "report", "az1")
    summary = json.loads((tmp_path / "report/summary.json").read_text())
    assert summary["variants"]["az1"]["sigma_A_mag"]["cells"] == 8
    assert summary["variants"]["az1"]["cost_variance_ratio_to_baseline"]["median"] == 1
    assert (tmp_path / "report/efficiency_summary.png").exists()


@pytest.mark.hyperion
@pytest.mark.solver
def test_small_real_solver_export(tmp_path, solver_path):
    """Tiny software check, not a scientific convergence measurement."""
    from dustcompendium.dust import ferrara
    from dustcompendium.hyperion_model import write_model

    configs = tmp_path / "configs"
    configs.mkdir()
    base = yaml.safe_load((STUDY.parents[1] / "configs/tied-two-component-smoke.yaml").read_text())
    base["dust"] = {"ferrara": "milkyWay"}
    base["geometry"].update(radialCells=10, verticalCells=10)
    base["geometry"]["components"]["disk"]["dust"]["opticalDepth"] = [0, 0.3]
    spheroid = base["geometry"]["components"]["spheroid"]
    spheroid["stellar"]["scaleRadial"] = spheroid["dust"]["scaleRadial"] = 1
    spheroid["dust"]["opticalDepth"] = 0
    base["tabulation"].update(
        wavelengths=[0.55],
        inclinations=[0, 60],
        azimuths=[90],
        photons=100,
        imagingPhotons=100,
        raytracingPhotons=150,
    )
    dust = ferrara.build("milkyWay")
    for seed in range(3):
        base["label"] = f"testEfficiency{seed}"
        base["tabulation"]["seed"] = -653 - 10000 * seed
        path = configs / f"az1_seed{seed}.yaml"
        path.write_text(yaml.safe_dump(base))
        campaign = Campaign(load_campaign(str(path)))
        output = tmp_path / "runs" / path.stem / "output"
        output.mkdir(parents=True)
        (output / "logs").mkdir()
        for run in campaign.runs():
            source = output / f"{run.file_stem}.hdf5"
            result = output / f"{run.file_stem}.rtout"
            write_model(run.spec, dust, str(source), sampling="average", **campaign.grid_options())
            process = subprocess.run(
                [solver_path, "-f", str(source), str(result)], capture_output=True, text=True, check=True
            )
            (output / "logs" / f"{run.file_stem}.log").write_text(process.stdout)
    efficiency.export(configs, tmp_path / "runs", tmp_path / "compact")
    with np.load(tmp_path / "compact/az1.npz", allow_pickle=False) as archive:
        assert archive["flux"].shape == (3, 4, 1, 2, 1)
        assert np.isfinite(archive["cpu_seconds"]).all()
        assert (archive["cpu_seconds"] >= 0).all()
        metadata = json.loads(str(archive["metadata"]))
        assert metadata["imaging_photons"] == 100
        assert metadata["raytracing_photons"] == 150
