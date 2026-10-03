"""Tests for expanding a configuration into the models it describes."""

import numpy as np
import pytest

from dustcompendium.campaign import Campaign
from dustcompendium.config import CampaignConfig

PUBLISHED = """
label: compendium
dust: {ferrara: milkyWay, referenceOpacity: 250.0}
geometry:
  components:
    disk:
      stellar: {profile: exponentialDisk, verticalStructure: sechSquared,
                scaleRadial: 1.0, scaleHeight: 0.137}
      dust: {profile: exponentialDisk, verticalStructure: sechSquared,
             scaleRadial: 1.0, scaleHeight: 0.137,
             opticalDepth: {minimum: 0.01, maximum: 1.0e4, count: 60, includeZero: true}}
    spheroid:
      stellar: {profile: hernquist, truncation: 10.0,
                scaleRadial: {minimum: 1.0e-3, maximum: 1.0e2, count: 51}}
"""

SPHEROID_DUST = (
    PUBLISHED
    + """      dust: {profile: hernquist, truncation: 10.0,
             scaleRadial: {minimum: 1.0e-3, maximum: 1.0e2, count: 51},
             opticalDepth: {minimum: 0.01, maximum: 1.0e4, count: 6, includeZero: true}}
"""
)

INDEPENDENT_DUST = (
    PUBLISHED
    + """      dust: {profile: hernquist, truncation: 10.0,
             scaleRadial: {minimum: 0.1, maximum: 10.0, count: 4},
             opticalDepth: {minimum: 0.01, maximum: 1.0e4, count: 6, includeZero: true}}
"""
)


def campaign(text=PUBLISHED):
    return Campaign(CampaignConfig.from_yaml(text))


class TestAxes:
    def test_the_published_campaign_has_the_axes_the_original_had(self):
        one = campaign()
        assert [axis.name for axis in one.axes] == [
            "opticalDepth:disk",
            "scaleRadial:spheroid",
        ]
        assert one.shape == (61, 51)

    def test_a_scale_which_does_not_vary_is_not_an_axis(self):
        """The disk's scale radius is fixed, so it is not tabulated against."""
        assert "scaleRadial:disk" not in [axis.name for axis in campaign().axes]

    def test_stellar_axes_only_affect_their_own_emitter(self):
        """Attenuation depends on all the dust, but only on the observed stars."""
        one = campaign()
        assert [axis.name for axis in one.axes_for("disk")] == ["opticalDepth:disk"]
        assert [axis.name for axis in one.axes_for("spheroid")] == [
            "opticalDepth:disk",
            "scaleRadial:spheroid",
        ]

    def test_dust_axes_affect_every_emitter(self):
        one = campaign(SPHEROID_DUST)
        for emitter in one.emitters:
            names = [axis.name for axis in one.axes_for(emitter)]
            assert "opticalDepth:disk" in names
            assert "opticalDepth:spheroid" in names

    def test_a_shared_scale_is_one_axis_not_two(self):
        """Stars and dust of the same size must not be a grid of a size against itself."""
        one = campaign(SPHEROID_DUST)
        radial = [axis for axis in one.axes if axis.kind == "scaleRadial"]
        assert len(radial) == 1
        assert radial[0].roles == frozenset({"stellar", "dust"})
        assert radial[0].name == "scaleRadial:spheroid"

    def test_independent_scales_stay_separate_and_are_named_apart(self):
        """The original already gave the disk different stellar and dust scale heights."""
        one = campaign(INDEPENDENT_DUST)
        radial = sorted(axis.name for axis in one.axes if axis.kind == "scaleRadial")
        assert radial == ["scaleRadial:spheroid:dust", "scaleRadial:spheroid:stellar"]


class TestRuns:
    def test_the_model_count_matches_the_axes(self):
        one = campaign()
        assert len(one) == 61 + 61 * 51
        assert len(list(one.runs())) == len(one)

    def test_pruning_irrelevant_axes_saves_models(self):
        """Tabulating the disk against the spheroid size would repeat it 51 times."""
        one = campaign()
        naive = 61 * 51 * 2
        assert len(one) < naive

    def test_every_run_has_a_distinct_label(self):
        labels = [run.label for run in campaign().runs()]
        assert len(set(labels)) == len(labels)

    def test_file_stems_carry_no_awkward_characters(self):
        one = campaign(PUBLISHED.replace("label: compendium", "label: a:b/c d"))
        for run in one.runs():
            assert not set(run.file_stem) & set(":/ ")

    def test_seeds_are_distinct(self):
        """The original decremented a seed per model so each got its own realization."""
        seeds = [run.spec.seed for run in campaign().runs()]
        assert len(set(seeds)) == len(seeds)

    def test_geometry_seed_policy_pairs_optical_depth_variants(self):
        text = (
            SPHEROID_DUST.replace(
                "{minimum: 0.01, maximum: 1.0e4, count: 60, includeZero: true}",
                "[0.0, 1.0]",
            )
            .replace(
                "{minimum: 0.01, maximum: 1.0e4, count: 6, includeZero: true}",
                "[0.0, 1.0]",
            )
            .replace(
                "{minimum: 1.0e-3, maximum: 1.0e2, count: 51}",
                "[0.5, 2.0]",
            )
        )
        one = campaign(text + "tabulation: {seedPolicy: geometry}\n")
        runs = list(one.runs())
        grouped = {}
        for run in runs:
            axes = one.axes_for(run.emitter)
            geometry = tuple(
                index for axis, index in zip(axes, run.indices, strict=True) if axis.kind != "opticalDepth"
            )
            grouped.setdefault((run.emitter, geometry), set()).add(run.spec.seed)
        assert grouped
        assert all(len(seeds) == 1 for seeds in grouped.values())
        assert len({next(iter(seeds)) for seeds in grouped.values()}) == len(grouped)

    def test_optical_depths_track_their_axis(self):
        one = campaign()
        depths = one.axes[0].values
        for run in one.runs():
            if run.emitter == "disk":
                assert run.spec.optical_depths["disk"] == pytest.approx(depths[run.indices[0]])

    def test_the_spheroid_scale_tracks_its_axis(self):
        one = campaign()
        radii = one.axes[1].values
        for run in one.runs():
            if run.emitter == "spheroid":
                scale = run.spec.galaxy["spheroid"].stellar.scale_radial
                assert scale == pytest.approx(radii[run.indices[1]])

    def test_an_irrelevant_component_is_left_out(self):
        """A dustless spheroid cannot change the disk's light, and must not size the grid."""
        for run in campaign().runs():
            if run.emitter == "disk":
                assert run.spec.galaxy.names == ("disk",)

    def test_a_dusty_spheroid_is_kept_even_when_the_disk_emits(self):
        for run in campaign(SPHEROID_DUST).runs():
            if run.emitter == "disk":
                assert set(run.spec.galaxy.names) == {"disk", "spheroid"}
                assert run.spec.galaxy["spheroid"].stellar is None
                assert run.spec.galaxy["spheroid"].dust is not None

    def test_dropping_the_spheroid_keeps_the_disk_grid_small(self):
        """Otherwise a large spheroid would set the grid for a disk-only model."""
        one = campaign()
        for run in one.runs():
            if run.emitter == "disk":
                assert run.spec.galaxy.extent_vertical == pytest.approx(0.137)

    def test_wavelengths_and_viewing_angles_reach_every_run(self):
        one = campaign()
        for run in list(one.runs())[:5]:
            np.testing.assert_allclose(run.spec.wavelengths, one.wavelengths)
            np.testing.assert_allclose(run.spec.inclinations, one.inclinations)
            np.testing.assert_allclose(run.spec.azimuths, one.azimuths)

    def test_grid_options_come_from_the_configuration(self):
        one = campaign(PUBLISHED.replace("  components:", "  spacing: nested\n  components:"))
        assert one.grid_options()["spacing"] == "nested"

    def test_repr_summarises_the_campaign(self):
        assert "3172 models" in repr(campaign())


CUSTOM_INCLINATIONS = (
    PUBLISHED.replace(
        "label: compendium",
        "label: compendium",
    )
    + """tabulation:
  inclinations: [9.3, 22.9, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
"""
)


def test_custom_inclinations_are_the_ones_recorded():
    """Regression against a defect in the published Ferrara tabulations.

    ``runModels.pl`` passed a model's own inclinations to the model builder, so
    the models were solved at them, but then built the axis it recorded as a
    uniform grid from zero to ninety regardless (lines 400-406). The published
    ``Ferrara1999_*`` datasets therefore hold attenuations computed at 9.3, 22.9,
    30 ... degrees while their ``inclination`` dataset says 0, 11.25, 22.5 ...,
    so anything interpolating them reads the wrong angle. Comparing the
    published file against Ferrara's own atlas shows it: index for index they
    agree to a median of 0.003 in transmission, while interpolating the atlas
    onto the uniform axis is nearly three times worse.
    """
    one = campaign(CUSTOM_INCLINATIONS)
    expected = [9.3, 22.9, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    np.testing.assert_allclose(one.inclinations, expected)
    for run in list(one.runs())[:3]:
        np.testing.assert_allclose(run.spec.inclinations, expected)
    assert not np.allclose(one.inclinations, np.linspace(0.0, 90.0, len(expected)))
