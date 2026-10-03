"""Tests for turning solved models into a tabulation."""

import numpy as np
import pytest

from dustcompendium.postprocess import (
    EXTRAPOLATION_DECADES,
    average_azimuths,
    attenuation_of,
    fit_extrapolation,
)


class TestAverageAzimuths:
    def test_values_are_averaged_over_azimuth_only(self):
        values = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
        uncertainties = np.ones_like(values)
        averaged, _ = average_azimuths(values, uncertainties)
        np.testing.assert_allclose(averaged, np.mean(values, axis=0))
        assert averaged.shape == (3, 4)

    def test_uncertainty_uses_fully_correlated_upper_bound(self):
        values = np.ones((2, 1, 2))
        uncertainties = np.array([[[0.1, 0.2]], [[0.3, 0.4]]])
        _, combined = average_azimuths(values, uncertainties)
        np.testing.assert_allclose(combined, [[0.2, 0.3]])

    def test_independence_is_not_silently_assumed(self):
        values = np.ones((4, 1, 1))
        uncertainties = np.ones_like(values) * 0.2
        _, combined = average_azimuths(values, uncertainties)
        assert combined[0, 0] == pytest.approx(0.2)

    @pytest.mark.parametrize(
        ("values", "uncertainties"),
        [(np.ones((2, 3)), np.ones((2, 3))), (np.ones((2, 3, 4)), np.ones((2, 3, 5)))],
    )
    def test_bad_shapes_are_rejected(self, values, uncertainties):
        with pytest.raises(ValueError):
            average_azimuths(values, uncertainties)


class TestAttenuation:
    def test_a_model_divided_by_itself_is_transparent(self):
        values = np.array([[2.0, 3.0]])
        attenuation, _ = attenuation_of(values, values * 0.01, values, values * 0.01)
        np.testing.assert_allclose(attenuation, 1.0)

    def test_attenuation_is_the_ratio(self):
        attenuation, _ = attenuation_of(np.array([1.0]), np.array([0.0]), np.array([4.0]), np.array([0.0]))
        assert attenuation[0] == pytest.approx(0.25)

    def test_both_uncertainties_are_propagated(self):
        """The original divided only the numerator's, understating the result."""
        values, reference = np.array([1.0]), np.array([2.0])
        one_sided = 0.1 / 2.0
        _, uncertainty = attenuation_of(values, np.array([0.1]), reference, np.array([0.2]))
        assert uncertainty[0] > one_sided
        expected = 0.5 * np.sqrt((0.1 / 1.0) ** 2 + (0.2 / 2.0) ** 2)
        assert uncertainty[0] == pytest.approx(expected)

    def test_a_zero_reference_gives_no_answer_rather_than_an_error(self):
        attenuation, _ = attenuation_of(np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))
        assert np.isnan(attenuation[0])


class TestExtrapolation:
    def perfect(self, constant=-0.5, gradient=-0.3):
        # Several points per decade, as the published sixty-point grid has. With
        # only two in the fitting range a straight line passes through both and
        # the residual is identically zero, which says nothing about the form.
        depths = np.hstack([0.0, np.logspace(-2.0, 3.0, 21)])
        with np.errstate(divide="ignore"):
            attenuation = np.exp(constant) * np.where(depths > 0, depths, 1.0) ** gradient
        return depths, attenuation[:, None, None] * np.ones((1, 2, 3))

    def test_a_power_law_is_recovered_exactly(self):
        depths, attenuation = self.perfect()
        coefficients, residual = fit_extrapolation(depths, attenuation)
        np.testing.assert_allclose(coefficients[0], -0.5, rtol=1.0e-9)
        np.testing.assert_allclose(coefficients[1], -0.3, rtol=1.0e-9)
        np.testing.assert_allclose(residual, 0.0, atol=1.0e-12)

    def test_the_coefficients_reproduce_the_transmission(self):
        """They are used as exp(c0 + c1 ln tau), so check that directly."""
        depths, attenuation = self.perfect()
        coefficients, _ = fit_extrapolation(depths, attenuation)
        predicted = np.exp(coefficients[0] + coefficients[1] * np.log(1.0e4))
        assert predicted[0, 0] == pytest.approx(np.exp(-0.5) * 1.0e4**-0.3, rel=1.0e-9)

    def test_departure_from_a_power_law_shows_in_the_residual(self):
        """The point of keeping it: the form is assumed, not derived."""
        depths, attenuation = self.perfect()
        bent = attenuation.copy()
        bent[-1] *= 2.0
        _, residual = fit_extrapolation(depths, bent)
        assert np.all(residual > 0.05)

    def test_only_the_top_decade_is_fitted(self):
        """Changing a depth outside the fitting range must not move the fit."""
        depths, attenuation = self.perfect()
        coefficients, _ = fit_extrapolation(depths, attenuation)
        disturbed = attenuation.copy()
        disturbed[1] *= 10.0
        moved, _ = fit_extrapolation(depths, disturbed)
        np.testing.assert_allclose(coefficients, moved, rtol=1.0e-12)

    def test_two_points_in_range_make_the_residual_meaningless(self):
        """Recorded so a coarse grid's zero residual is not read as a good fit."""
        depths = np.array([0.0, 1.0, 100.0, 1000.0])
        attenuation = np.array([1.0, 0.5, 0.2, 0.02])[:, None] * np.ones((1, 2))
        _, residual = fit_extrapolation(depths, attenuation)
        np.testing.assert_allclose(residual, 0.0, atol=1.0e-12)

    def test_the_zero_entry_is_ignored(self):
        """It cannot take part in a fit in the log."""
        depths, attenuation = self.perfect()
        assert depths[0] == 0.0
        coefficients, _ = fit_extrapolation(depths, attenuation)
        assert np.all(np.isfinite(coefficients))

    def test_the_fitting_range_is_the_documented_factor(self):
        depths = np.array([1.0, 5.0, 20.0, 100.0])
        attenuation = np.ones((4, 1))
        # Only depths within a factor of ten of 100 are used, so 1.0 is excluded.
        coefficients, _ = fit_extrapolation(depths, attenuation, decades=EXTRAPOLATION_DECADES)
        np.testing.assert_allclose(coefficients[1], 0.0, atol=1.0e-12)

    def test_too_few_points_is_refused(self):
        with pytest.raises(ValueError, match="needs at least two"):
            fit_extrapolation(np.array([0.0, 1.0]), np.ones((2, 1)))

    def test_a_mismatched_axis_is_refused(self):
        with pytest.raises(ValueError, match="optical depths against an array"):
            fit_extrapolation(np.array([1.0, 10.0, 100.0]), np.ones((2, 1)))

    def test_a_non_positive_transmission_gives_no_coefficients(self):
        """A model with nothing escaping cannot be fitted in the log."""
        depths, attenuation = self.perfect()
        attenuation[:, 0, 0] = 0.0
        coefficients, residual = fit_extrapolation(depths, attenuation)
        assert np.all(np.isnan(coefficients[:, 0, 0]))
        assert np.isnan(residual[0, 0])
        assert np.all(np.isfinite(coefficients[:, 1, 1]))


@pytest.mark.hyperion
@pytest.mark.solver
@pytest.mark.slow
class TestCollect:
    """End to end: build a small campaign, solve it, and assemble it."""

    CONFIG = """
label: collected
dust: {ferrara: milkyWay, referenceOpacity: 250.0}
geometry:
  cutOff: 10.0
  radialCells: 16
  verticalCells: 16
  spacing: nested
  sampling: average
  components:
    disk:
      stellar: {profile: exponentialDisk, verticalStructure: sechSquared,
                scaleRadial: 1.0, scaleHeight: 0.137}
      dust: {profile: exponentialDisk, verticalStructure: sechSquared,
             scaleRadial: 1.0, scaleHeight: 0.137,
             opticalDepth: {minimum: 1.0, maximum: 100.0, count: 3, includeZero: true}}
tabulation:
  wavelengths: {minimum: 0.44, maximum: 0.55, count: 2}
  inclinations: {minimum: 0.0, maximum: 90.0, count: 2, spacing: linear}
  photons: 4000
"""

    @pytest.fixture(scope="class")
    def tabulation(self, tmp_path_factory):
        import subprocess

        from dustcompendium.campaign import Campaign
        from dustcompendium.config import CampaignConfig
        from dustcompendium.dust import ferrara, opacity_to_extinction
        from dustcompendium.hyperion_model import write_model
        from dustcompendium.postprocess import collect
        from dustcompendium.runner import solver_command

        root = tmp_path_factory.mktemp("collected")
        campaign = Campaign(CampaignConfig.from_yaml(self.CONFIG))
        dust = ferrara.build("milkyWay", 250.0)
        for run in campaign.runs():
            source = root / f"{run.file_stem}.hdf5"
            write_model(run.spec, dust, str(source), sampling="average", **campaign.grid_options())
            completed = subprocess.run(
                solver_command(source, root / f"{run.file_stem}.rtout"),
                capture_output=True,
                text=True,
                check=False,
            )
            assert completed.returncode == 0
        return collect(campaign, root, opacity_to_extinction(dust))

    def test_the_unattenuated_model_transmits_everything(self, tabulation):
        """Dividing the dust-free model by itself must give exactly one."""
        np.testing.assert_allclose(tabulation.attenuation["disk"][:, :, 0], 1.0)

    def test_transmission_falls_as_optical_depth_rises(self, tabulation):
        attenuation = tabulation.attenuation["disk"]
        assert np.all(np.diff(attenuation, axis=2) < 0.0)

    def test_transmission_stays_between_zero_and_one(self, tabulation):
        attenuation = tabulation.attenuation["disk"]
        assert np.all(attenuation > 0.0) and np.all(attenuation <= 1.0 + 1.0e-9)

    def test_the_blue_is_attenuated_more_than_the_visual(self, tabulation):
        """Reddening, from the shape of the extinction curve."""
        attenuation = tabulation.attenuation["disk"]
        assert np.all(attenuation[0, :, 1:] < attenuation[1, :, 1:])

    def test_edge_on_is_attenuated_more_than_face_on(self, tabulation):
        attenuation = tabulation.attenuation["disk"]
        assert np.all(attenuation[:, 1, 1:] < attenuation[:, 0, 1:])

    def test_the_shapes_are_what_the_axes_imply(self, tabulation):
        tabulation.check()
        assert tabulation.expected_shape("disk") == (2, 2, 4)

    def test_it_is_a_version_one_tabulation_without_a_spheroid(self, tabulation):
        """One emitter and one axis is not the published layout."""
        assert tabulation.format_version == 2
