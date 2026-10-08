"""Spatial-convergence manifests, matched-seed statistics and compact reports."""

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from dustcompendium.campaign import Campaign
from dustcompendium.config import CampaignConfig
from dustcompendium.grid import CylindricalGrid

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "resolution", ROOT / "studies/tied-two-component-resolution/study.py"
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def base_config():
    return yaml.safe_load((ROOT / "configs/tied-two-component-smoke.yaml").read_text())


def test_configs_refine_all_spatial_scales_and_preserve_physics():
    base = base_config()
    configs = list(study.configurations(base))
    assert len(configs) == 9
    assert base["geometry"]["components"]["disk"]["dust"]["opticalDepth"] == [0, 0.3, 3]
    originals = Campaign(CampaignConfig.model_validate(base)).grid_options()
    assert originals["inner_radius_fraction"] == 0.01
    for name, config in configs:
        campaign = Campaign(CampaignConfig.model_validate(config))
        assert len(campaign) == 24
        opts = campaign.grid_options()
        n = opts["radial_cells"]
        assert opts["inner_radius_fraction"] == pytest.approx(1 / n)
        assert config["tabulation"]["seed"] == -653 - 10000 * int(name[-1])
        assert campaign.azimuths.tolist() == [90]
        run = next(campaign.runs())
        grid = CylindricalGrid.for_galaxy(run.spec.galaxy, 10, **opts)
        assert grid.shape == (1, n, n)
        assert grid.radial_walls[1] == pytest.approx(run.spec.galaxy.radial_scales[0] / n)
        assert grid.radial_walls[-1] == pytest.approx(10 * run.spec.galaxy.extent_radial)


@pytest.mark.parametrize("fraction", [0, -1, np.inf, np.nan, 10, 11])
def test_invalid_inner_radius_rejected(fraction):
    base = base_config()
    base["geometry"]["innerRadiusFraction"] = fraction
    with pytest.raises(ValueError):
        CampaignConfig.model_validate(base)
    run = next(Campaign(CampaignConfig.model_validate(base_config())).runs())
    with pytest.raises(ValueError):
        CylindricalGrid.for_galaxy(run.spec.galaxy, 10, inner_radius_fraction=fraction)


def test_paired_uncertainty_retains_covariance_and_sign():
    fine = np.array([0.4, 0.5, 0.6])[:, None]
    coarse = fine * 0.9
    d = study.difference(coarse, fine)
    np.testing.assert_allclose(d["delta_A"], -2.5 * np.log10(0.9))
    np.testing.assert_allclose(d["se_A"], 0, atol=1e-15)
    np.testing.assert_allclose(d["delta_T"], -0.05)
    np.testing.assert_allclose(d["se_T"], 0.01 / np.sqrt(3))
    z = study.difference(np.zeros((3, 1)), np.zeros((3, 1)))
    assert np.isnan(z["delta_A"]).all()
    assert z["delta_T"].item() == 0


def synthetic_archives(tmp_path):
    paths = []
    for n in study.LEVELS:
        raw = next(c for name, c in study.configurations(base_config()) if name == f"grid{n}_seed0")
        campaign = Campaign(CampaignConfig.model_validate(raw))
        runs, records, references = study.exporter().model_records(campaign)
        meta = {
            "geometry": campaign.config.geometry.model_dump(mode="json"),
            "dust": {},
            "models": records,
            "references": references,
            "inclinations": [0, 90],
            "wavelengths": [0.1, 0.55],
            "azimuths": [90],
            "imaging_photons": 40000,
            "raytracing_photons": 40000,
            "seed_policy": "geometry",
            "provenance": [{"model_seeds": [r.spec.seed - s * 10000 for r in runs]} for s in range(3)],
        }
        flux = np.ones((3, len(runs), 1, 2, 2))
        dusty = np.arange(len(runs)) != np.array(references)
        flux[:, dusty] = 0.5 * np.exp(-1 / n)
        flux[:, dusty] *= np.array([0.99, 1, 1.01])[:, None, None, None, None]
        path = tmp_path / f"grid{n}.npz"
        np.savez(
            path, metadata=json.dumps(meta), flux=flux, seeds=[0, 1, 2], cpu_seconds=np.ones((3, len(runs)))
        )
        paths.append(path)
    return paths


def test_compact_report_and_missing_level(tmp_path):
    pytest.importorskip("matplotlib")
    paths = synthetic_archives(tmp_path)
    study.report(paths, tmp_path / "report")
    summary = json.loads((tmp_path / "report/summary.json").read_text())
    assert summary["comparisons"]["200-400"]["absolute_delta_A"]["median"] == pytest.approx(
        study.FACTOR * (1 / 200 - 1 / 400)
    )
    assert (tmp_path / "report/resolution_curves.png").exists()
    with pytest.raises(ValueError, match="all three"):
        study.load_matched(paths[:2])


@pytest.mark.parametrize("change", ["seed", "budget", "physical", "inner"])
def test_mismatched_inputs_rejected(tmp_path, change):
    paths = synthetic_archives(tmp_path)
    with np.load(paths[0]) as f:
        data = dict(f)
    meta = json.loads(str(data["metadata"]))
    if change == "seed":
        meta["provenance"][0]["model_seeds"][0] -= 1
    elif change == "budget":
        meta["imaging_photons"] = 10000
    elif change == "physical":
        meta["geometry"]["cut_off"] = 20
    else:
        meta["geometry"]["inner_radius_fraction"] = 0.03
    data["metadata"] = json.dumps(meta)
    np.savez(paths[0], **data)
    with pytest.raises(ValueError):
        study.load_matched(paths)


@pytest.mark.hyperion
@pytest.mark.solver
def test_real_solver_accepts_refined_grids(tmp_path, solver_path):
    """Low-photon software smoke test, not a scientific convergence result."""
    import h5py

    from dustcompendium.dust import ferrara
    from dustcompendium.hyperion_model import write_model
    from dustcompendium.postprocess import read_sed_views

    dust = ferrara.build("milkyWay")
    for name, config in study.configurations(base_config()):
        if not name.endswith("seed0"):
            continue
        config["tabulation"].update(
            wavelengths=[0.55], inclinations=[0, 60], imagingPhotons=100, raytracingPhotons=100
        )
        campaign = Campaign(CampaignConfig.model_validate(config))
        run = next(r for r in campaign.runs() if r.emitter == "spheroid" and r.indices == (1, 1, 0))
        source, result = tmp_path / f"{name}.hdf5", tmp_path / f"{name}.rtout"
        write_model(run.spec, dust, str(source), sampling="average", **campaign.grid_options())
        with h5py.File(source) as f:
            walls = f["Grid/Geometry/walls_1"]["w"]
            assert walls[1] == pytest.approx(
                run.spec.galaxy.radial_scales[0] / campaign.config.geometry.radial_cells
            )
        subprocess.run([solver_path, "-f", str(source), str(result)], capture_output=True, check=True)
        _, flux, _ = read_sed_views(result, azimuth_count=1)
        assert flux.shape == (1, 2, 1)
        assert np.isfinite(flux).all() and np.all(flux >= 0)
