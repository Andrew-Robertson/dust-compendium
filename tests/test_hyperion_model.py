"""Tests for the Hyperion adapter.

Everything here needs Hyperion, and the round trip needs a solver binary too.
Both are skipped when absent; see ``conftest.py``.
"""

import subprocess

import numpy as np
import pytest

from dustcompendium.galaxy import Component, Galaxy
from dustcompendium.geometry import ExponentialDisk, HernquistSpheroid, SechSquaredVertical
from dustcompendium.model import ModelSpec, dust_density, stellar_emission

pytestmark = pytest.mark.hyperion

# Physical scales, as the original used: a disk scale length of 1 kpc, in the
# centimetres Hyperion works in. Results are independent of the choice.
KILOPARSEC = 3.0856776e21
DISK = ExponentialDisk(KILOPARSEC, SechSquaredVertical(0.137 * KILOPARSEC))


def galaxy():
    return Galaxy(
        [
            Component("disk", stellar=DISK, dust=DISK),
            Component("spheroid", stellar=HernquistSpheroid(0.1 * KILOPARSEC)),
        ]
    )


def spec(optical_depth=1.0, **kwargs):
    return ModelSpec(
        galaxy(),
        kwargs.pop("emitter", "disk"),
        {"disk": optical_depth},
        wavelengths=np.array([0.44, 0.55]),
        inclinations=np.array([0.0, 90.0]),
        photons=kwargs.pop("photons", 10000),
        seed=-654,
        **kwargs,
    )


@pytest.fixture(scope="module")
def dust():
    from dustcompendium.dust import ferrara

    return ferrara.build("milkyWay", reference_opacity=250.0)


class TestBuildModel:
    def test_separate_stage_budgets(self, dust):
        from dustcompendium.hyperion_model import build_model

        model = build_model(
            spec(imaging_photons=123, raytracing_photons=456), dust, radial_cells=20, vertical_cells=20
        )
        assert model.n_photons["last_sources"] == 123
        assert model.n_photons["raytracing_sources"] == 456

    def test_the_grid_matches_the_specification(self, dust):
        from dustcompendium.hyperion_model import build_model

        model_spec = spec()
        model = build_model(model_spec, dust, radial_cells=20, vertical_cells=20)
        assert model.grid.shape == model_spec.grid(radial_cells=20, vertical_cells=20).shape

    def test_the_dust_density_matches_what_we_computed(self, dust):
        from dustcompendium.dust import opacity_to_extinction
        from dustcompendium.hyperion_model import build_model

        model_spec = spec()
        grid = model_spec.grid(radial_cells=20, vertical_cells=20)
        model = build_model(model_spec, dust, radial_cells=20, vertical_cells=20)
        expected = dust_density(
            model_spec.galaxy, grid, model_spec.optical_depths, opacity_to_extinction(dust)
        )
        np.testing.assert_allclose(model.grid["density"][0].array, expected, rtol=1.0e-12)

    def test_the_source_map_matches_the_stellar_mass(self, dust):
        from dustcompendium.hyperion_model import build_model

        model_spec = spec()
        grid = model_spec.grid(radial_cells=20, vertical_cells=20)
        model = build_model(model_spec, dust, radial_cells=20, vertical_cells=20)
        expected = stellar_emission(model_spec.galaxy, grid, "disk")
        np.testing.assert_allclose(np.asarray(model.sources[0].map), expected, rtol=1.0e-12)

    def test_an_explicit_opacity_overrides_the_dust(self, dust):
        from dustcompendium.hyperion_model import build_model

        model_spec = spec()
        doubled = build_model(model_spec, dust, opacity=500.0, radial_cells=20, vertical_cells=20)
        halved = build_model(model_spec, dust, opacity=250.0, radial_cells=20, vertical_cells=20)
        np.testing.assert_allclose(
            doubled.grid["density"][0].array * 2.0,
            halved.grid["density"][0].array,
            rtol=1.0e-12,
        )

    def test_emission_can_come_from_the_spheroid(self, dust):
        from dustcompendium.hyperion_model import build_model

        model = build_model(spec(emitter="spheroid"), dust, radial_cells=20, vertical_cells=20)
        assert np.asarray(model.sources[0].map).sum() > 0.0


@pytest.mark.solver
@pytest.mark.slow
class TestSolverRoundTrip:
    def solve(self, tmp_path, solver_path, dust, optical_depth):
        from hyperion.model import ModelOutput

        from dustcompendium.hyperion_model import write_model

        source = str(tmp_path / f"input_{optical_depth}.hdf5")
        result = str(tmp_path / f"output_{optical_depth}.hdf5")
        write_model(spec(optical_depth), dust, source, radial_cells=30, vertical_cells=30)
        completed = subprocess.run(
            [solver_path, "-f", source, result], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
        sed = ModelOutput(result).get_sed()
        # Peel-off directions are the two configured azimuths of each
        # inclination. They share photon histories, so this average is not
        # assumed to halve the variance.
        values = np.array(sed.val)[:, 0, :]
        return (values[:2] + values[2:]) / 2.0, np.array(sed.wav)

    def test_attenuation_increases_towards_edge_on(self, tmp_path, solver_path, dust):
        attenuated, _ = self.solve(tmp_path, solver_path, dust, 1.0)
        face_on, edge_on = attenuated
        assert np.all(face_on > edge_on)

    def test_a_dust_free_model_is_brighter_at_every_inclination(self, tmp_path, solver_path, dust):
        transparent, _ = self.solve(tmp_path, solver_path, dust, 0.0)
        attenuated, _ = self.solve(tmp_path, solver_path, dust, 1.0)
        assert np.all(attenuated <= transparent * 1.001)

    def test_a_dust_free_model_is_isotropic(self, tmp_path, solver_path, dust):
        """With no dust the galaxy looks the same from every direction."""
        transparent, _ = self.solve(tmp_path, solver_path, dust, 0.0)
        face_on, edge_on = transparent
        np.testing.assert_allclose(face_on, edge_on, rtol=0.05)

    def test_extinction_is_greater_in_the_blue(self, tmp_path, solver_path, dust):
        """Reddening: the 0.44 micron light is attenuated more than 0.55."""
        transparent, wavelengths = self.solve(tmp_path, solver_path, dust, 0.0)
        attenuated, _ = self.solve(tmp_path, solver_path, dust, 5.0)
        assert np.allclose(wavelengths, [0.44, 0.55])
        transmission = attenuated / transparent
        blue, visual = transmission[1]
        assert blue < visual

    def test_configured_azimuths_survive_the_solver_round_trip(self, tmp_path, solver_path, dust):
        from dustcompendium.hyperion_model import write_model
        from dustcompendium.postprocess import read_sed_views

        source = str(tmp_path / "three_azimuths.hdf5")
        result = str(tmp_path / "three_azimuths.rtout")
        write_model(
            spec(1.0, azimuths=np.array([0.0, 120.0, 240.0]), photons=2000),
            dust,
            source,
            radial_cells=20,
            vertical_cells=20,
        )
        completed = subprocess.run(
            [solver_path, "-f", source, result], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
        wavelengths, values, uncertainties = read_sed_views(result, azimuth_count=3)
        assert wavelengths.size == 2
        assert values.shape == uncertainties.shape == (3, 2, 2)
