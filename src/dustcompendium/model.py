r"""Assembling the arrays a radiative transfer model is built from.

Everything here is plain NumPy and needs no Hyperion, so the physics can be
tested without a radiative transfer install. :mod:`dustcompendium.hyperion_model`
turns the result into a Hyperion model file.

A model is one galaxy geometry, one set of optical depths, and one emitting
component. The dust distribution is the sum over every component carrying dust,
each normalized to its own optical depth, so putting dust in the spheroid as
well as the disk needs nothing here beyond another entry in the mapping.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .galaxy import Galaxy
from .grid import CylindricalGrid

__all__ = [
    "ModelSpec",
    "dust_density",
    "flat_spectrum",
    "stellar_emission",
    "viewing_angles",
]


def dust_density(
    galaxy: Galaxy,
    grid: CylindricalGrid,
    optical_depths: Mapping[str, float],
    opacity: float,
    sampling: str = "centre",
) -> NDArray[np.float64]:
    r"""Dust density in every cell, summed over the components carrying dust.

    Each component's profile is normalized so that its own optical depth along
    its own reference ray is the value given, then all are added. Because the
    dust files give opacities per unit dust mass, the result is a dust density.

    Sampled at cell centres by default, as the original did. That is a point
    sample rather than a cell average, so the dust mass on the grid is not quite
    the mass the profile actually has, and the model does not quite have the
    optical depth it was asked for. Measured on the published grid, against the
    exactly integrated mass:

    ======================  ==================  ================
    dust profile            total dust mass     worst cell
    ======================  ==================  ================
    disk, sech^2            -0.08%              -1.6%
    Hernquist, r_s = 1      +0.3%               -59%
    Jaffe, r_s = 1          +7.1%               -94.5%
    ======================  ==================  ================

    Negligible for a disk, which is all the published tabulations use, and the
    default reproduces them. It is not negligible for dust in a spheroid, whose
    cusp the innermost cells badly under-sample, so ``sampling="average"`` gives
    each cell the density that puts the profile's true mass in it.

    Parameters
    ----------
    galaxy
        Supplies the components and their dust profiles.
    grid
        Where to evaluate.
    optical_depths
        Optical depth for each dust-bearing component, by name. Every such
        component must appear, and no others.
    opacity
        Opacity to extinction per unit dust mass in the V band.
    sampling
        ``centre`` to evaluate the profile at each cell centre, reproducing the
        original, or ``average`` to give each cell the mean density over its
        volume, which conserves dust mass.

    Raises
    ------
    KeyError
        If the mapping does not name exactly the dust-bearing components.
    ValueError
        If ``sampling`` is not one of the two accepted values.
    """
    if sampling not in ("centre", "average"):
        raise ValueError(f"sampling must be 'centre' or 'average', got {sampling!r}")
    expected = {component.name for component in galaxy.attenuating}
    given = set(optical_depths)
    if given != expected:
        missing = ", ".join(sorted(expected - given)) or "none"
        extra = ", ".join(sorted(given - expected)) or "none"
        raise KeyError(
            f"optical depths must name exactly the dust-bearing components; "
            f"missing: {missing}; unexpected: {extra}"
        )

    radius, height = grid.centres
    density = np.zeros(grid.shape)
    for component in galaxy.attenuating:
        profile = component.dust
        assert profile is not None  # guaranteed by Galaxy.attenuating
        depth = optical_depths[component.name]
        if depth == 0.0:
            continue
        normalization = profile.density_normalization(depth, opacity)
        if sampling == "centre":
            shape = profile.density(radius, height)
        else:
            shape = profile.cell_mass(*grid.bounds) / grid.cell_volumes
        density = density + normalization * shape
    return density


def stellar_emission(galaxy: Galaxy, grid: CylindricalGrid, emitter: str) -> NDArray[np.float64]:
    """Relative probability of a photon being emitted from each cell.

    This is the stellar mass in the cell, integrated over the cell rather than
    sampled at its centre, so that the total is right however coarse the grid.
    Hyperion normalizes the map itself, so only the relative values matter.

    Raises
    ------
    KeyError
        If there is no such component.
    ValueError
        If the component has no stars.
    """
    component = galaxy[emitter]
    if component.stellar is None:
        emitting = ", ".join(other.name for other in galaxy.emitting) or "none"
        raise ValueError(
            f"component {emitter!r} has no stars, so cannot emit; components with stars: {emitting}"
        )
    return component.stellar.cell_mass(*grid.bounds)


def viewing_angles(
    inclinations: NDArray[np.float64],
    azimuths: NDArray[np.float64] | None = None,
) -> tuple[NDArray, NDArray]:
    r"""Peel-off directions for every inclination--azimuth combination.

    The default azimuths of 90 and 270 degrees reproduce the original Benson
    calculations. In an axisymmetric model all requested azimuths have the same
    expectation, but estimates from them share photon histories and need not be
    statistically independent. Requesting more directions also adds peel-off
    work, so the useful number is a convergence question rather than an exact
    symmetry argument.

    Returns
    -------
    Polar and azimuthal angles in degrees, ordered in one inclination block per
    azimuth. This order lets post-processing reshape the viewing-direction axis
    to ``(azimuth, inclination)``.
    """
    inclinations = np.asarray(inclinations, dtype=float)
    if azimuths is None:
        azimuths = np.array([90.0, 270.0])
    azimuths = np.asarray(azimuths, dtype=float)
    if inclinations.ndim != 1 or inclinations.size == 0:
        raise ValueError("inclinations must be a non-empty one-dimensional array")
    if np.any(inclinations < 0.0) or np.any(inclinations > 90.0):
        raise ValueError("inclinations must lie between 0 and 90 degrees")
    if azimuths.ndim != 1 or azimuths.size == 0:
        raise ValueError("azimuths must be a non-empty one-dimensional array")
    if np.any(azimuths < 0.0) or np.any(azimuths >= 360.0):
        raise ValueError("azimuths must lie in [0, 360) degrees")
    if np.unique(azimuths).size != azimuths.size:
        raise ValueError("azimuths must not contain duplicate directions")
    polar = np.tile(inclinations, azimuths.size)
    azimuthal = np.repeat(azimuths, inclinations.size)
    return polar, azimuthal


def flat_spectrum(
    frequency_range: tuple[float, float], points: int = 100
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""A flat :math:`F_\nu` source spectrum, as frequency and flux.

    The radiative transfer is monochromatic and only ratios of emergent to
    emitted luminosity are ever used, so the shape of the source spectrum
    cancels and any spectrum will do. The original read one from a
    ``spectrum.txt`` which was never version controlled and is now lost;
    generating it removes that dependency.

    It must lie inside the frequency range the dust's optical properties are
    defined over. A photon drawn outside it aborts the solve -- and Hyperion
    still exits zero when it does, so the failure is silent. Callers should take
    the range from the dust rather than assume one; see
    :func:`~dustcompendium.hyperion_model.spectrum_for`.

    Parameters
    ----------
    frequency_range
        Lowest and highest frequency to cover, in Hz.
    points
        How many frequencies to sample.
    """
    lowest, highest = frequency_range
    if not 0.0 < lowest < highest:
        raise ValueError(f"need 0 < lowest < highest, got {frequency_range}")
    frequency = np.logspace(np.log10(lowest), np.log10(highest), points)
    return frequency, np.ones_like(frequency)


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to build one radiative transfer model.

    Parameters
    ----------
    galaxy
        The geometry.
    emitter
        Name of the component whose stars are the source of light. One model is
        run per emitting component, since the attenuation is tabulated
        separately for each.
    optical_depths
        Optical depth of each dust-bearing component, by name.
    wavelengths
        Wavelengths to solve at, in microns.
    inclinations
        Viewing inclinations in degrees, from face-on at 0 to edge-on at 90.
    azimuths
        Peel-off azimuths in degrees. The default reproduces the two opposite
        directions in the original Benson calculations.
    cut_off
        How many scale lengths out to extend the grid.
    photons
        Default sampling budget for both passes. Imaging uses this number per
        wavelength; direct raytracing shares its samples across wavelengths.
    imaging_photons, raytracing_photons
        Optional overrides for the two passes, leaving ``photons`` as the
        backwards-compatible default.
    seed
        Random seed. The original decremented a seed per model so that each got
        an independent realization; do the same when building a grid of models.
    """

    galaxy: Galaxy
    emitter: str
    optical_depths: Mapping[str, float] = field(default_factory=dict)
    wavelengths: NDArray[np.float64] = field(default_factory=lambda: np.array([0.55]))
    inclinations: NDArray[np.float64] = field(default_factory=lambda: np.array([90.0]))
    azimuths: NDArray[np.float64] = field(default_factory=lambda: np.array([90.0, 270.0]))
    cut_off: float = 10.0
    photons: int = 100000
    seed: int = -1
    imaging_photons: int | None = None
    raytracing_photons: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "wavelengths", np.atleast_1d(np.asarray(self.wavelengths, float)))
        object.__setattr__(self, "inclinations", np.atleast_1d(np.asarray(self.inclinations, float)))
        object.__setattr__(self, "azimuths", np.atleast_1d(np.asarray(self.azimuths, float)))
        if np.any(self.wavelengths <= 0.0):
            raise ValueError("wavelengths must be positive")
        if self.photons <= 0:
            raise ValueError(f"photons must be positive, got {self.photons}")
        for name in ("imaging_photons", "raytracing_photons"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        # Validates the inclinations, and the emitter through stellar_emission's
        # own checks when the model is built.
        viewing_angles(self.inclinations, self.azimuths)

    def grid(self, **options: Any) -> CylindricalGrid:
        """The grid this model is solved on.

        Options are passed to :meth:`~dustcompendium.grid.CylindricalGrid.for_galaxy`,
        notably ``spacing`` and the cell counts.
        """
        return CylindricalGrid.for_galaxy(self.galaxy, self.cut_off, **options)
