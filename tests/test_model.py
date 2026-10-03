"""Tests for assembling the arrays a model is built from."""

import numpy as np
import pytest

from dustcompendium.galaxy import Component, Galaxy
from dustcompendium.geometry import (
    ExponentialDisk,
    HernquistSpheroid,
    JaffeSpheroid,
    SechSquaredVertical,
)
from dustcompendium.model import (
    ModelSpec,
    dust_density,
    flat_spectrum,
    stellar_emission,
    viewing_angles,
)

DISK = ExponentialDisk(1.0, SechSquaredVertical(0.137))


def galaxy(spheroid_dust=None, spheroid_scale=1.0):
    return Galaxy(
        [
            Component("disk", stellar=DISK, dust=DISK),
            Component(
                "spheroid",
                stellar=HernquistSpheroid(spheroid_scale),
                dust=spheroid_dust,
            ),
        ]
    )


def grid_for(galaxy_, **kwargs):
    from dustcompendium.grid import CylindricalGrid

    return CylindricalGrid.for_galaxy(galaxy_, 10.0, **kwargs)


class TestDustDensity:
    def test_zero_optical_depth_gives_no_dust(self):
        one = galaxy()
        grid = grid_for(one)
        density = dust_density(one, grid, {"disk": 0.0}, opacity=1.0)
        assert np.all(density == 0.0)

    def test_density_scales_with_optical_depth(self):
        one = galaxy()
        grid = grid_for(one)
        single = dust_density(one, grid, {"disk": 1.0}, opacity=1.0)
        double = dust_density(one, grid, {"disk": 2.0}, opacity=1.0)
        np.testing.assert_allclose(double, 2.0 * single, rtol=1.0e-12)

    def test_density_scales_inversely_with_opacity(self):
        one = galaxy()
        grid = grid_for(one)
        thin = dust_density(one, grid, {"disk": 1.0}, opacity=1.0)
        thick = dust_density(one, grid, {"disk": 1.0}, opacity=2.0)
        np.testing.assert_allclose(thick, thin / 2.0, rtol=1.0e-12)

    def test_components_add(self):
        """Dust in two components is the sum, which is the spheroid extension."""
        both = galaxy(spheroid_dust=HernquistSpheroid(1.0))
        grid = grid_for(both)
        disk_only = dust_density(both, grid, {"disk": 1.0, "spheroid": 0.0}, 1.0)
        spheroid_only = dust_density(both, grid, {"disk": 0.0, "spheroid": 1.0}, 1.0)
        together = dust_density(both, grid, {"disk": 1.0, "spheroid": 1.0}, 1.0)
        np.testing.assert_allclose(together, disk_only + spheroid_only, rtol=1.0e-12)

    @pytest.mark.parametrize(
        "profile", [DISK, HernquistSpheroid(1.0), JaffeSpheroid(1.0)], ids=["disk", "hernquist", "jaffe"]
    )
    def test_averaged_sampling_conserves_dust_mass(self, profile):
        """Point sampling does not, which matters for the cuspy profiles."""
        one = Galaxy([Component("c", stellar=DISK, dust=profile)])
        grid = grid_for(one)
        exact = profile.density_normalization(1.0, 1.0) * profile.cell_mass(*grid.bounds).sum()
        averaged = dust_density(one, grid, {"c": 1.0}, 1.0, sampling="average")
        assert float((averaged * grid.cell_volumes).sum()) == pytest.approx(exact, rel=1.0e-10)

    def test_centre_sampling_reproduces_a_direct_evaluation(self):
        """The default must remain what the original did: a point sample."""
        one = galaxy()
        grid = grid_for(one)
        radius, height = grid.centres
        expected = DISK.density_normalization(1.0, 1.0) * DISK.density(radius, height)
        np.testing.assert_allclose(dust_density(one, grid, {"disk": 1.0}, 1.0), expected, rtol=1.0e-13)

    def test_missing_optical_depth_is_rejected(self):
        both = galaxy(spheroid_dust=HernquistSpheroid(1.0))
        with pytest.raises(KeyError, match="missing: spheroid"):
            dust_density(both, grid_for(both), {"disk": 1.0}, 1.0)

    def test_optical_depth_for_a_dustless_component_is_rejected(self):
        one = galaxy()
        with pytest.raises(KeyError, match="unexpected: spheroid"):
            dust_density(one, grid_for(one), {"disk": 1.0, "spheroid": 1.0}, 1.0)

    def test_unknown_sampling_is_rejected(self):
        one = galaxy()
        with pytest.raises(ValueError, match="sampling must be"):
            dust_density(one, grid_for(one), {"disk": 1.0}, 1.0, sampling="middle")

    def test_a_large_spheroid_empties_the_disk_dust_when_point_sampled(self):
        """Recorded behaviour, not endorsed: see the note in dust_density.

        The vertical extent is set by the largest component, so a spheroid much
        bigger than the disk scale height leaves cells many dust scale heights
        thick. Point sampling then finds essentially no dust anywhere, and the
        model has none of the optical depth it was asked for. Averaging over the
        cell keeps the mass.
        """
        big = galaxy(spheroid_scale=10.0)
        grid = grid_for(big)
        exact = DISK.density_normalization(1.0, 1.0) * DISK.cell_mass(*grid.bounds).sum()
        sampled = dust_density(big, grid, {"disk": 1.0}, 1.0, sampling="centre")
        averaged = dust_density(big, grid, {"disk": 1.0}, 1.0, sampling="average")
        assert float((sampled * grid.cell_volumes).sum()) < 1.0e-3 * exact
        assert float((averaged * grid.cell_volumes).sum()) == pytest.approx(exact, rel=1.0e-10)


class TestStellarEmission:
    def test_emission_is_the_cell_mass(self):
        one = galaxy()
        grid = grid_for(one)
        np.testing.assert_allclose(
            stellar_emission(one, grid, "disk"), DISK.cell_mass(*grid.bounds), rtol=1.0e-13
        )

    def test_emission_is_non_negative_and_finite(self):
        one = galaxy()
        grid = grid_for(one)
        for name in ("disk", "spheroid"):
            emission = stellar_emission(one, grid, name)
            assert np.all(np.isfinite(emission)) and np.all(emission >= 0.0)

    def test_emission_is_symmetric_about_the_midplane(self):
        one = galaxy()
        grid = grid_for(one)
        emission = stellar_emission(one, grid, "spheroid")
        np.testing.assert_allclose(emission, emission[:, ::-1], rtol=1.0e-6, atol=1.0e-11 * emission.sum())

    def test_a_dustless_component_can_still_emit(self):
        one = galaxy()
        assert stellar_emission(one, grid_for(one), "spheroid").sum() > 0.0

    def test_a_starless_component_cannot_emit(self):
        one = Galaxy([Component("disk", stellar=DISK), Component("halo", dust=DISK)])
        with pytest.raises(ValueError, match="has no stars"):
            stellar_emission(one, grid_for(one), "halo")

    def test_an_unknown_component_is_rejected(self):
        one = galaxy()
        with pytest.raises(KeyError, match="no component named"):
            stellar_emission(one, grid_for(one), "bulge")


class TestViewingAngles:
    def test_default_reproduces_two_opposite_azimuths(self):
        inclinations = np.array([0.0, 45.0, 90.0])
        polar, azimuthal = viewing_angles(inclinations)
        assert polar.size == azimuthal.size == 2 * inclinations.size
        np.testing.assert_allclose(polar[:3], inclinations)
        np.testing.assert_allclose(polar[3:], inclinations)
        np.testing.assert_allclose(azimuthal[:3], 90.0)
        np.testing.assert_allclose(azimuthal[3:], 270.0)

    def test_each_inclination_is_seen_at_every_requested_azimuth(self):
        inclinations = np.array([0.0, 60.0])
        azimuths = np.array([0.0, 120.0, 240.0])
        polar, azimuthal = viewing_angles(inclinations, azimuths)
        np.testing.assert_allclose(polar, [0.0, 60.0, 0.0, 60.0, 0.0, 60.0])
        np.testing.assert_allclose(azimuthal, [0.0, 0.0, 120.0, 120.0, 240.0, 240.0])

    @pytest.mark.parametrize("bad", [np.array([-1.0]), np.array([91.0])])
    def test_inclinations_outside_the_quadrant_are_rejected(self, bad):
        with pytest.raises(ValueError, match="between 0 and 90"):
            viewing_angles(bad)

    def test_an_empty_set_is_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            viewing_angles(np.array([]))

    @pytest.mark.parametrize(
        "bad",
        [np.array([]), np.array([-1.0]), np.array([360.0]), np.array([90.0, 90.0])],
    )
    def test_invalid_azimuths_are_rejected(self, bad):
        with pytest.raises(ValueError, match="azimuths"):
            viewing_angles(np.array([45.0]), bad)


class TestFlatSpectrum:
    def test_flux_is_flat_and_frequency_ascending(self):
        frequency, flux = flat_spectrum((1.0e12, 1.0e16))
        assert np.all(np.diff(frequency) > 0.0)
        assert np.all(flux == flux[0])

    def test_the_range_is_spanned_exactly(self):
        frequency, _ = flat_spectrum((1.0e12, 1.0e16))
        assert frequency.min() == pytest.approx(1.0e12, rel=1.0e-12)
        assert frequency.max() == pytest.approx(1.0e16, rel=1.0e-12)

    def test_an_inverted_range_is_rejected(self):
        with pytest.raises(ValueError, match="lowest < highest"):
            flat_spectrum((1.0e16, 1.0e12))

    @pytest.mark.hyperion
    def test_the_spectrum_stays_inside_the_dust_frequency_range(self):
        """A photon outside it aborts the solve, and Hyperion still exits zero.

        Regression: the default range matched the dust's extrapolation range
        exactly, and rounding put its top frequency 1.5e12 Hz above the dust's.
        Whether a photon was drawn there was a matter of chance, so a fraction of
        a campaign's models aborted and were recorded as successes.
        """
        from dustcompendium.dust import ferrara
        from dustcompendium.hyperion_model import spectrum_for

        dust = ferrara.build("milkyWay", 250.0)
        frequency, _ = spectrum_for(dust)
        available = dust.optical_properties.nu
        assert frequency.min() > available.min()
        assert frequency.max() < available.max()


class TestModelSpec:
    def test_defaults_give_a_usable_grid(self):
        spec = ModelSpec(galaxy(), "disk", {"disk": 1.0})
        assert spec.grid().shape == (1, 100, 100)

    def test_scalars_are_promoted_to_arrays(self):
        spec = ModelSpec(
            galaxy(), "disk", {"disk": 1.0}, wavelengths=0.55, inclinations=90.0, azimuths=45.0
        )
        assert spec.wavelengths.shape == (1,) and spec.inclinations.shape == (1,)
        assert spec.azimuths.shape == (1,)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"wavelengths": np.array([-1.0])}, "wavelengths must be positive"),
            ({"photons": 0}, "photons must be positive"),
            ({"inclinations": np.array([100.0])}, "between 0 and 90"),
        ],
    )
    def test_invalid_specifications_are_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ModelSpec(galaxy(), "disk", {"disk": 1.0}, **kwargs)


@pytest.mark.parametrize("spheroid_scale", [1.0, 10.0, 100.0])
def test_nested_spacing_keeps_the_disk_dust_whatever_the_spheroid(spheroid_scale):
    """The fix for the large-spheroid failure recorded above.

    On the published grid a spheroid ten times the disk scale length leaves the
    model with no disk dust at all. Resolving the smallest scale in each
    direction restores the requested optical depth, at every spheroid size, with
    the original point sampling untouched.
    """
    from dustcompendium.grid import CylindricalGrid

    one = galaxy(spheroid_scale=spheroid_scale)
    grid = CylindricalGrid.for_galaxy(one, 10.0, spacing="nested")
    exact = DISK.density_normalization(1.0, 1.0) * DISK.cell_mass(*grid.bounds).sum()
    density = dust_density(one, grid, {"disk": 1.0}, 1.0, sampling="centre")
    assert float((density * grid.cell_volumes).sum()) == pytest.approx(exact, rel=0.02)
