"""Tests for the campaign configuration schema."""

import numpy as np
import pytest
from pydantic import ValidationError

from dustcompendium.config import CampaignConfig, RangeConfig, load_campaign, values_of
from dustcompendium.geometry import ExponentialDisk, HernquistSpheroid

MINIMAL = """
label: minimal
dust: {ferrara: milkyWay}
geometry:
  components:
    disk:
      stellar: {profile: exponentialDisk, verticalStructure: sechSquared,
                scaleRadial: 1.0, scaleHeight: 0.137}
      dust: {profile: exponentialDisk, verticalStructure: sechSquared,
             scaleRadial: 1.0, scaleHeight: 0.137, opticalDepth: 1.0}
"""


class TestRangeConfig:
    def test_logarithmic_spacing(self):
        values = RangeConfig(minimum=0.01, maximum=100.0, count=5).values()
        np.testing.assert_allclose(values, [0.01, 0.1, 1.0, 10.0, 100.0])

    def test_linear_spacing(self):
        values = RangeConfig(minimum=0.0, maximum=90.0, count=3, spacing="linear").values()
        np.testing.assert_allclose(values, [0.0, 45.0, 90.0])

    def test_zero_is_prepended_when_asked(self):
        """The optical depth axes carry an exact zero for the unattenuated case."""
        values = RangeConfig(minimum=1.0, maximum=100.0, count=3, include_zero=True).values()
        np.testing.assert_allclose(values, [0.0, 1.0, 10.0, 100.0])

    def test_zero_is_not_duplicated(self):
        values = RangeConfig(minimum=0.0, maximum=2.0, count=3, spacing="linear", include_zero=True).values()
        np.testing.assert_allclose(values, [0.0, 1.0, 2.0])

    def test_a_single_point_is_the_minimum(self):
        np.testing.assert_allclose(RangeConfig(minimum=7.0, maximum=9.0, count=1).values(), [7.0])

    def test_logarithmic_spacing_needs_a_positive_minimum(self):
        with pytest.raises(ValidationError, match="positive minimum"):
            RangeConfig(minimum=0.0, maximum=1.0, count=3)

    def test_an_inverted_range_is_rejected(self):
        with pytest.raises(ValidationError, match="below minimum"):
            RangeConfig(minimum=10.0, maximum=1.0, count=3)


class TestValuesOf:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [(1.5, [1.5]), ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])],
    )
    def test_scalars_and_lists_pass_through(self, given, expected):
        np.testing.assert_allclose(values_of(given), expected)

    def test_ranges_are_expanded(self):
        assert values_of(RangeConfig(minimum=1.0, maximum=10.0, count=2)).size == 2


class TestCampaignConfig:
    def test_a_minimal_configuration_parses(self):
        config = CampaignConfig.from_yaml(MINIMAL)
        assert config.label == "minimal"
        assert config.dust.ferrara == "milkyWay"
        assert config.geometry.spacing == "published"
        assert config.tabulation.seed_policy == "model"

    def test_geometry_seed_policy_parses_from_camel_case(self):
        config = CampaignConfig.from_yaml(MINIMAL + "tabulation: {seedPolicy: geometry}\n")
        assert config.tabulation.seed_policy == "geometry"

    def test_camel_case_keys_are_used_in_yaml(self):
        config = CampaignConfig.from_yaml(MINIMAL)
        assert config.geometry.cut_off == 10.0
        assert config.geometry.components["disk"].dust.vertical_structure == "sechSquared"

    def test_unknown_keys_are_rejected(self):
        """A typo in a configuration should not be silently ignored."""
        with pytest.raises(ValidationError):
            CampaignConfig.from_yaml(MINIMAL + "  unexpectedKey: 1\n")

    def test_a_galaxy_with_no_stars_is_rejected(self):
        text = MINIMAL.replace(
            "      stellar: {profile: exponentialDisk, verticalStructure: sechSquared,\n"
            "                scaleRadial: 1.0, scaleHeight: 0.137}\n",
            "",
        )
        with pytest.raises(ValidationError, match="nothing would emit"):
            CampaignConfig.from_yaml(text)

    def test_a_galaxy_with_no_dust_is_rejected(self):
        text = MINIMAL.replace(
            "      dust: {profile: exponentialDisk, verticalStructure: sechSquared,\n"
            "             scaleRadial: 1.0, scaleHeight: 0.137, opticalDepth: 1.0}\n",
            "",
        )
        with pytest.raises(ValidationError, match="nothing to attenuate"):
            CampaignConfig.from_yaml(text)

    def test_dust_needs_exactly_one_source_of_grains(self):
        with pytest.raises(ValidationError, match="exactly one"):
            CampaignConfig.from_yaml(MINIMAL.replace("{ferrara: milkyWay}", "{}"))
        with pytest.raises(ValidationError, match="exactly one"):
            CampaignConfig.from_yaml(
                MINIMAL.replace("{ferrara: milkyWay}", "{ferrara: milkyWay, file: a.hdf5}")
            )

    def test_a_dust_profile_needs_an_optical_depth(self):
        with pytest.raises(ValidationError, match="needs an opticalDepth"):
            CampaignConfig.from_yaml(MINIMAL.replace(", opticalDepth: 1.0", ""))

    def test_a_stellar_profile_may_not_carry_an_optical_depth(self):
        with pytest.raises(ValidationError, match="belongs to a dust profile"):
            CampaignConfig.from_yaml(
                MINIMAL.replace(
                    "scaleRadial: 1.0, scaleHeight: 0.137}",
                    "scaleRadial: 1.0, scaleHeight: 0.137, opticalDepth: 1.0}",
                    1,
                )
            )

    def test_yaml_which_is_not_a_mapping_is_rejected(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            CampaignConfig.from_yaml("- just\n- a list\n")


class TestProfileConfig:
    def test_a_disk_is_built_with_its_vertical_structure(self):
        config = CampaignConfig.from_yaml(MINIMAL)
        profile = config.geometry.components["disk"].stellar.build()
        assert isinstance(profile, ExponentialDisk)
        assert profile.vertical.scale_height == pytest.approx(0.137)

    def test_a_spheroid_is_built_with_its_truncation(self):
        text = (
            MINIMAL
            + """    spheroid:
      stellar: {profile: hernquist, scaleRadial: 2.0, truncation: 5.8}
"""
        )
        config = CampaignConfig.from_yaml(text)
        profile = config.geometry.components["spheroid"].stellar.build()
        assert isinstance(profile, HernquistSpheroid)
        assert profile.truncation == pytest.approx(5.8)

    def test_a_disk_needs_a_scale_height(self):
        with pytest.raises(ValidationError, match="needs a scaleHeight"):
            CampaignConfig.from_yaml(MINIMAL.replace(", scaleHeight: 0.137}", "}", 1))

    def test_a_disk_may_not_be_truncated(self):
        with pytest.raises(ValidationError, match="truncation applies to the spheroids"):
            CampaignConfig.from_yaml(
                MINIMAL.replace("scaleHeight: 0.137}", "scaleHeight: 0.137, truncation: 5.0}", 1)
            )

    def test_a_spheroid_may_not_take_a_scale_height(self):
        text = (
            MINIMAL
            + """    spheroid:
      stellar: {profile: hernquist, scaleRadial: 2.0, scaleHeight: 1.0}
"""
        )
        with pytest.raises(ValidationError, match="spherical"):
            CampaignConfig.from_yaml(text)

    def test_building_a_varying_parameter_without_a_choice_is_refused(self):
        text = MINIMAL.replace(
            "      stellar: {profile: exponentialDisk, verticalStructure: sechSquared,\n"
            "                scaleRadial: 1.0, scaleHeight: 0.137}",
            "      stellar: {profile: exponentialDisk, verticalStructure: sechSquared,\n"
            "                scaleRadial: {minimum: 1.0, maximum: 4.0, count: 3}, scaleHeight: 0.137}",
        )
        config = CampaignConfig.from_yaml(text)
        with pytest.raises(ValueError, match="varies across the campaign"):
            config.geometry.components["disk"].stellar.build()


class TestShippedConfigurations:
    """The configurations in ``configs/`` must stay valid."""

    @pytest.mark.parametrize(
        "name",
        [
            "compendium-d03-rv3.1.yaml",
            "compendium-spheroid-dust.yaml",
            "tied-two-component-smoke.yaml",
        ],
    )
    def test_they_load(self, name):
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "configs" / name
        config = load_campaign(str(path))
        assert config.label
        assert config.geometry.components

    def test_the_published_configuration_reproduces_the_original_grids(self):
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "configs" / "compendium-d03-rv3.1.yaml"
        config = load_campaign(str(path))
        depths = values_of(config.geometry.components["disk"].dust.optical_depth)
        radii = values_of(config.geometry.components["spheroid"].stellar.scale_radial)
        assert depths.size == 61 and depths[0] == 0.0 and depths[-1] == pytest.approx(1.0e4)
        assert radii.size == 51
        np.testing.assert_allclose(radii[[0, -1]], [1.0e-3, 1.0e2])
        assert values_of(config.tabulation.wavelengths).size == 250
        assert values_of(config.tabulation.inclinations).size == 46
