r"""Declarative configuration for a campaign of models.

The original carried its configuration in the driver itself, as Perl data
structures with the variants that were not currently wanted commented out --
``runModels.pl`` has three such blocks, and the state they were left in is why
it no longer reproduces the published datasets. Here a campaign is a YAML file.

Lengths are in units of the disk scale length, as in the original, which chose a
physical disk scale radius of a millionth of a megaparsec and noted that the
results do not depend on it. Optical depths and wavelengths fix everything that
matters, so the absolute scale cancels.

Values which vary across the campaign are written as a range rather than a
number, and become an axis of the output tabulation:

.. code-block:: yaml

    opticalDepth: {minimum: 0.01, maximum: 1.0e4, count: 60, includeZero: true}
"""

from typing import Annotated, Any, Literal

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .geometry import (
    ExponentialDisk,
    Profile,
    spheroid_profile,
    vertical_structure,
)

__all__ = [
    "CampaignConfig",
    "ComponentConfig",
    "DustConfig",
    "GeometryConfig",
    "ProfileConfig",
    "RangeConfig",
    "TabulationConfig",
    "load_campaign",
    "values_of",
]


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(word.capitalize() for word in rest)


class _Base(BaseModel):
    model_config = ConfigDict(alias_generator=_camel, populate_by_name=True, extra="forbid", frozen=True)


class RangeConfig(_Base):
    """A parameter which varies across the campaign, becoming an output axis."""

    minimum: float
    maximum: float
    count: int = Field(gt=0)
    spacing: Literal["log", "linear"] = "log"
    include_zero: bool = False
    """Prepend an exact zero, as the optical depth axes do for the unattenuated case."""

    @model_validator(mode="after")
    def _check(self) -> "RangeConfig":
        if self.maximum < self.minimum:
            raise ValueError(f"maximum {self.maximum} is below minimum {self.minimum}")
        if self.spacing == "log" and self.minimum <= 0.0:
            raise ValueError("logarithmic spacing needs a positive minimum")
        return self

    def values(self) -> NDArray[np.float64]:
        """The tabulation points, ascending."""
        if self.count == 1:
            points = np.array([self.minimum])
        elif self.spacing == "log":
            points = np.logspace(np.log10(self.minimum), np.log10(self.maximum), self.count)
        else:
            points = np.linspace(self.minimum, self.maximum, self.count)
        if self.include_zero and not np.any(points == 0.0):
            points = np.hstack([0.0, points])
        return points


#: A parameter given as a single value, an explicit list, or a range.
Parameter = Annotated[float | list[float] | RangeConfig, Field(union_mode="left_to_right")]


def values_of(parameter: Parameter) -> NDArray[np.float64]:
    """The values a parameter takes, as an array, whichever form it was given in."""
    if isinstance(parameter, RangeConfig):
        return parameter.values()
    return np.atleast_1d(np.asarray(parameter, dtype=float))


class ProfileConfig(_Base):
    """One density profile, whose varying parameters become output axes."""

    profile: Literal["exponentialDisk", "hernquist", "jaffe"]
    scale_radial: Parameter = 1.0
    scale_height: Parameter | None = None
    """Only for ``exponentialDisk``; the vertical scale height."""
    vertical_structure: Literal["exponential", "sechSquared"] | None = None
    """Only for ``exponentialDisk``."""
    truncation: float | None = None
    """Only for the spheroids; the radius, in scale radii, beyond which there is nothing."""
    optical_depth: Parameter | None = None
    """Only meaningful for a dust profile, where it is what the profile is normalized to."""

    @model_validator(mode="after")
    def _check(self) -> "ProfileConfig":
        disk = self.profile == "exponentialDisk"
        if disk and self.scale_height is None:
            raise ValueError("exponentialDisk needs a scaleHeight")
        if disk and self.truncation is not None:
            raise ValueError("truncation applies to the spheroids, not to a disk")
        if not disk and (self.scale_height is not None or self.vertical_structure is not None):
            raise ValueError(
                f"{self.profile} is spherical, so takes neither scaleHeight nor verticalStructure"
            )
        return self

    def build(self, **chosen: float) -> Profile:
        """Build the profile, taking one value for each varying parameter.

        Parameters
        ----------
        **chosen
            A value for ``scale_radial`` and, for a disk, ``scale_height``.
            Anything not given falls back to the configured value, which must
            then be a single number.
        """

        def one(name: str) -> float:
            if name in chosen:
                return chosen[name]
            values = values_of(getattr(self, name))
            if values.size != 1:
                raise ValueError(f"{name} varies across the campaign, so a value must be chosen")
            return float(values[0])

        if self.profile == "exponentialDisk":
            structure = vertical_structure(self.vertical_structure or "sechSquared", one("scale_height"))
            return ExponentialDisk(one("scale_radial"), structure)
        return spheroid_profile(self.profile, one("scale_radial"), self.truncation)


class ComponentConfig(_Base):
    """A named component, with an optional stellar and an optional dust profile."""

    stellar: ProfileConfig | None = None
    dust: ProfileConfig | None = None

    @model_validator(mode="after")
    def _check(self) -> "ComponentConfig":
        if self.stellar is None and self.dust is None:
            raise ValueError("a component needs a stellar profile, a dust profile, or both")
        if self.stellar is not None and self.stellar.optical_depth is not None:
            raise ValueError("opticalDepth belongs to a dust profile, not a stellar one")
        if self.dust is not None and self.dust.optical_depth is None:
            raise ValueError("a dust profile needs an opticalDepth to normalize it")
        return self


class DustConfig(_Base):
    """Which grain properties to use."""

    file: str | None = None
    """Path to a Hyperion dust file, such as those built by the ``hyperion-dust`` package."""
    ferrara: Literal["milkyWay", "smallMagellanicCloud"] | None = None
    """Instead of a file, build the Gordon et al. (1997) grains used by Ferrara et al. (1999)."""
    reference_opacity: float = 1.0
    """The V band opacity to put a Ferrara extinction curve on; ignored with a file."""
    description: str = ""
    """Recorded in the output, as the original did."""

    @model_validator(mode="after")
    def _check(self) -> "DustConfig":
        if (self.file is None) == (self.ferrara is None):
            raise ValueError("give exactly one of a dust file or a ferrara grain type")
        return self


class GeometryConfig(_Base):
    """The galaxy, and the grid it is solved on."""

    components: dict[str, ComponentConfig] = Field(min_length=1)
    cut_off: float = Field(default=10.0, gt=0.0)
    spacing: Literal["published", "nested"] = "published"
    sampling: Literal["centre", "average"] = "centre"
    radial_cells: int = Field(default=100, ge=2)
    vertical_cells: int = Field(default=100, ge=1)
    inner_radius_fraction: float = Field(default=0.01, gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _check_inner_radius(self) -> "GeometryConfig":
        if self.inner_radius_fraction >= self.cut_off:
            raise ValueError("innerRadiusFraction must be below cutOff")
        return self


class TabulationConfig(_Base):
    """Where the attenuation is tabulated, and how hard the models are run."""

    wavelengths: Parameter = Field(default_factory=lambda: RangeConfig(minimum=0.01, maximum=3.0, count=250))
    inclinations: Parameter = Field(
        default_factory=lambda: RangeConfig(minimum=0.0, maximum=90.0, count=46, spacing="linear")
    )
    azimuths: Parameter = Field(default_factory=lambda: [90.0, 270.0])
    """Explicit peel-off azimuths, in degrees.

    The two opposite directions reproduce the Benson calculations. They are
    separate estimators made from shared photon histories, not independent
    radiative-transfer runs; convergence studies should vary this list rather
    than assuming their average gains a factor of square root two.
    """
    photons: int = Field(default=100000, gt=0)
    imaging_photons: int | None = Field(default=None, gt=0)
    """Scattered-light trajectories per wavelength; defaults to ``photons``."""
    raytracing_photons: int | None = Field(default=None, gt=0)
    """Direct-light source samples shared across wavelengths; defaults to ``photons``."""
    seed: int = -653
    """The first random seed assigned by the campaign."""
    seed_policy: Literal["model", "geometry"] = "model"
    """How seeds are shared.

    ``model`` reproduces the original scripts by assigning every radiative
    transfer model a distinct seed. ``geometry`` assigns one seed to all
    optical-depth variants of the same emitter and geometry. The latter makes
    the dusty models and their dust-free normalization share photon histories,
    reducing Monte Carlo noise in their ratio.
    """

    @model_validator(mode="after")
    def _check_angles(self) -> "TabulationConfig":
        inclinations = values_of(self.inclinations)
        if inclinations.size == 0 or np.any((inclinations < 0.0) | (inclinations > 90.0)):
            raise ValueError("inclinations must be non-empty and lie between 0 and 90 degrees")
        azimuths = values_of(self.azimuths)
        if azimuths.size == 0 or np.any((azimuths < 0.0) | (azimuths >= 360.0)):
            raise ValueError("azimuths must be non-empty and lie in [0, 360) degrees")
        if np.unique(azimuths).size != azimuths.size:
            raise ValueError("azimuths must not contain duplicate directions")
        return self


class CampaignConfig(_Base):
    """A complete description of a set of models and the tabulation they produce."""

    label: str
    description: str = ""
    dust: DustConfig
    geometry: GeometryConfig
    tabulation: TabulationConfig = Field(default_factory=TabulationConfig)

    @model_validator(mode="after")
    def _check(self) -> "CampaignConfig":
        if not any(component.stellar is not None for component in self.geometry.components.values()):
            raise ValueError("no component has stars, so nothing would emit any light")
        if not any(component.dust is not None for component in self.geometry.components.values()):
            raise ValueError("no component has dust, so there would be nothing to attenuate")
        return self

    @classmethod
    def from_yaml(cls, text: str) -> "CampaignConfig":
        """Parse a campaign from YAML text."""
        parsed: Any = yaml.safe_load(text)
        if not isinstance(parsed, dict):
            raise ValueError("a campaign configuration must be a mapping")
        return cls.model_validate(parsed)


def load_campaign(path: str) -> CampaignConfig:
    """Read and validate a campaign configuration from a YAML file."""
    with open(path, encoding="utf-8") as stream:
        return CampaignConfig.from_yaml(stream.read())
