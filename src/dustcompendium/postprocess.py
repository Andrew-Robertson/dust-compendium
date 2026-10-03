r"""Turning solved models into a tabulation.

Three steps, each of which the original did inline in its driver.

Peel-off directions at the same inclination are averaged over the configured
azimuths. For an axisymmetric model they have the same expectation, but they
share photon histories and are not assumed independent. The tabulated
uncertainty therefore uses the conservative fully-correlated limit; raw views
remain available through :func:`read_sed_views` for convergence studies.

The result is divided by the model with no dust at all, which turns a luminosity
into the fraction of light escaping. Taking the ratio against a model of the same
geometry and the same random seed cancels much of the remaining noise.

Beyond the largest tabulated optical depth the transmission is extrapolated as
``exp(c0 + c1 ln tau)``, fitted over the top decade. The published files store
the two coefficients and nothing else, so nothing records how well that form
actually fits. Here the residual of the fit is kept alongside them, because the
power law is motivated by a disk -- at high optical depth only a surface layer is
seen -- and it is not obvious it holds for a spheroid whose central optical depth
formally diverges.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "EXTRAPOLATION_DECADES",
    "average_azimuths",
    "attenuation_of",
    "collect",
    "fit_extrapolation",
    "read_sed",
    "read_sed_views",
]

#: How far below the largest optical depth the extrapolation is fitted over, as
#: a factor. Ten is the top decade, as in the original.
EXTRAPOLATION_DECADES = 10.0


def average_azimuths(
    values: NDArray,
    uncertainties: NDArray,
) -> tuple[NDArray, NDArray]:
    r"""Average an ``(azimuth, inclination, wavelength)`` array.

    The individual Hyperion uncertainties do not describe covariance between
    peel-off directions computed from the same photon histories. In the absence
    of that covariance, combining them in quadrature would assume independence.
    Instead return their arithmetic mean, corresponding to the upper bound in
    which their errors are perfectly positively correlated. Repeated complete
    runs with different seeds are needed to measure the uncertainty of the
    azimuthal mean more sharply.
    """
    values = np.asarray(values)
    uncertainties = np.asarray(uncertainties)
    if values.ndim != 3:
        raise ValueError("values must be indexed by azimuth, inclination and wavelength")
    if uncertainties.shape != values.shape:
        raise ValueError(
            f"uncertainties have shape {uncertainties.shape}, expected {values.shape}"
        )
    return np.mean(values, axis=0), np.mean(np.abs(uncertainties), axis=0)


def read_sed_views(
    path: str | Path,
    azimuth_count: int = 2,
) -> tuple[NDArray, NDArray, NDArray]:
    r"""Read every peel-off direction from a solved model.

    Returns wavelengths and luminosity/uncertainty arrays indexed by azimuth,
    inclination and wavelength. ``azimuth_count=2`` preserves compatibility
    with the original Benson configuration.
    """
    if azimuth_count <= 0:
        raise ValueError(f"azimuth_count must be positive, got {azimuth_count}")
    try:
        from hyperion.model import ModelOutput
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "reading a solved model needs Hyperion, which is not installed; "
            "see the installation notes in the README"
        ) from error

    sed = ModelOutput(str(path)).get_sed(uncertainties=True)
    # The aperture axis is present but singular: these are SEDs, not images.
    values = np.asarray(sed.val)[:, 0, :]
    uncertainties = np.asarray(sed.unc)[:, 0, :]
    if values.shape[0] % azimuth_count:
        raise ValueError(
            f"{path} has {values.shape[0]} viewing directions, which cannot be "
            f"divided among {azimuth_count} azimuths"
        )
    inclinations = values.shape[0] // azimuth_count
    shape = (azimuth_count, inclinations, values.shape[-1])
    return np.asarray(sed.wav), values.reshape(shape), uncertainties.reshape(shape)


def read_sed(
    path: str | Path,
    azimuth_count: int = 2,
) -> tuple[NDArray, NDArray, NDArray]:
    r"""Read a solved model, averaging each inclination over azimuth.

    Parameters
    ----------
    path
        A model solved by one of the Hyperion solvers.

    Returns
    -------
    Wavelengths in microns, luminosities indexed by inclination and wavelength,
    and conservative Monte Carlo uncertainties. The default of two azimuths
    reproduces the original model layout.

    Raises
    ------
    ImportError
        If Hyperion is not installed.
    ValueError
        If the model's viewing directions cannot be divided among the requested
        number of azimuths.
    """
    wavelengths, values, uncertainties = read_sed_views(path, azimuth_count=azimuth_count)
    averaged, conservative = average_azimuths(values, uncertainties)
    return wavelengths, averaged, conservative


def attenuation_of(
    values: NDArray,
    uncertainties: NDArray,
    reference: NDArray,
    reference_uncertainty: NDArray,
) -> tuple[NDArray, NDArray]:
    r"""The fraction of light escaping, and its uncertainty.

    Parameters
    ----------
    values, uncertainties
        The attenuated model.
    reference, reference_uncertainty
        The same model with no dust.

    Notes
    -----
    Both uncertainties are propagated. The original divided only the attenuated
    model's uncertainty by the reference, ignoring the reference's own, which
    understates the result near zero optical depth where the two are comparable.
    Galacticus does not read the uncertainties, so this changes nothing
    downstream; it makes them mean what they say.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        attenuation = np.where(reference > 0.0, values / reference, np.nan)
        relative = np.sqrt(
            np.where(values > 0.0, (uncertainties / values) ** 2, 0.0)
            + np.where(reference > 0.0, (reference_uncertainty / reference) ** 2, 0.0)
        )
    return attenuation, np.abs(attenuation) * relative


def fit_extrapolation(
    optical_depths: Sequence[float] | NDArray,
    attenuation: NDArray,
    decades: float = EXTRAPOLATION_DECADES,
) -> tuple[NDArray, NDArray]:
    r"""Fit ``exp(c0 + c1 ln tau)`` to the high optical depth end.

    Parameters
    ----------
    optical_depths
        The tabulated depths, ascending, along the first axis of ``attenuation``.
        A zero entry is ignored: it cannot take part in a fit in the log.
    attenuation
        Transmission, with optical depth along its first axis.
    decades
        Fit over depths within this factor of the largest.

    Returns
    -------
    Coefficients with the two of them along the first axis, and the root mean
    square residual of the fit in the log, over the remaining axes.

    The residual only says something when more than two depths fall in the
    fitting range: a straight line through two points passes through both, and
    reports a residual of zero whatever the underlying form. The published
    sixty-point grid spans six decades, so its top decade holds about ten.

    Raises
    ------
    ValueError
        If fewer than two usable depths fall in the fitting range.
    """
    depths = np.asarray(optical_depths, dtype=float)
    attenuation = np.asarray(attenuation, dtype=float)
    if depths.size != attenuation.shape[0]:
        raise ValueError(f"{depths.size} optical depths against an array of {attenuation.shape[0]}")
    usable = depths > 0.0
    within = usable & (depths * decades >= depths[usable].max())
    if int(np.count_nonzero(within)) < 2:
        raise ValueError(
            f"only {int(np.count_nonzero(within))} optical depths lie within a factor of "
            f"{decades:g} of the largest; the extrapolation needs at least two"
        )

    selected = attenuation[within]
    trailing = selected.shape[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        logarithms = np.log(selected.reshape(selected.shape[0], -1))
    abscissa = np.log(depths[within])
    finite = np.isfinite(logarithms).all(axis=0)
    coefficients = np.full((2, logarithms.shape[1]), np.nan)
    residual = np.full(logarithms.shape[1], np.nan)
    if np.any(finite):
        gradient, intercept = np.polyfit(abscissa, logarithms[:, finite], 1)
        coefficients[0, finite] = intercept
        coefficients[1, finite] = gradient
        modelled = intercept + np.outer(abscissa, gradient)
        residual[finite] = np.sqrt(np.mean((logarithms[:, finite] - modelled) ** 2, axis=0))
    return coefficients.reshape(2, *trailing), residual.reshape(trailing)


def _zero_index(values: NDArray, name: str) -> int:
    """Where an optical depth axis holds its zero entry."""
    found = np.flatnonzero(values == 0.0)
    if found.size != 1:
        raise ValueError(
            f"axis {name!r} must contain exactly one zero, for the model with no dust "
            f"that everything is normalized against; it has {found.size}. Set "
            "`includeZero: true` on its range."
        )
    return int(found[0])


def collect(
    campaign,
    output: str | Path,
    opacity: float,
    metadata: dict | None = None,
    suffix: str = ".rtout",
):
    """Assemble a campaign's solved models into a tabulation.

    Parameters
    ----------
    campaign
        The campaign whose models were run.
    output
        Directory holding the solved models.
    opacity
        The V band opacity per unit dust mass the models were built with.
        Recorded in the file, and what Galacticus uses to recognise it.
    metadata
        Extra attributes to record, such as the description of the grains.
    suffix
        Extension of the solved models.

    Raises
    ------
    FileNotFoundError
        If any model is missing, saying how many and naming the first.
    ValueError
        If an optical depth axis has no zero entry, or the wavelengths a model
        was solved at do not match the campaign.

    Notes
    -----
    Every model is held in memory as it is assembled. The published campaign is
    250 wavelengths by 46 inclinations by 61 optical depths by 51 spheroid
    sizes, which is some 285 MB per array.
    """
    from .tabulation import Tabulation

    output = Path(output)
    runs = list(campaign.runs())
    missing = [path for path in (output / f"{run.file_stem}{suffix}" for run in runs) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(runs)} solved models are missing, the first being "
            f"{missing[0]}. Run `dust-compendium run` first."
        )

    axis_values = {axis.name: axis.values for axis in campaign.axes}
    emitter_axes = {
        emitter: tuple(axis.name for axis in campaign.axes_for(emitter)) for emitter in campaign.emitters
    }
    depth_axes = {
        axis.name: _zero_index(axis.values, axis.name)
        for axis in campaign.axes
        if axis.kind == "opticalDepth"
    }

    wavelengths = None
    luminosity: dict[str, NDArray] = {}
    noise: dict[str, NDArray] = {}
    for run in runs:
        found, values, uncertainties = read_sed(
            output / f"{run.file_stem}{suffix}",
            azimuth_count=campaign.azimuths.size,
        )
        if wavelengths is None:
            wavelengths = found
            if not np.allclose(wavelengths, campaign.wavelengths, rtol=1.0e-6):
                raise ValueError(
                    "the models were solved at different wavelengths from the campaign's; "
                    "the configuration has changed since they were built"
                )
        shape = (wavelengths.size, values.shape[0], *campaign.shape_for(run.emitter))
        if run.emitter not in luminosity:
            luminosity[run.emitter] = np.full(shape, np.nan)
            noise[run.emitter] = np.full(shape, np.nan)
        index = (slice(None), slice(None), *run.indices)
        luminosity[run.emitter][index] = values.T
        noise[run.emitter][index] = uncertainties.T

    assert wavelengths is not None  # runs is non-empty, or `missing` would have fired
    inclinations = campaign.inclinations

    attenuation, uncertainty, extrapolation, residual = {}, {}, {}, {}
    for emitter in campaign.emitters:
        axes = emitter_axes[emitter]
        # The unattenuated model sits where every optical depth is zero, and at
        # the same geometry, so that the ratio cancels what noise it can.
        reference_index: list = [slice(None), slice(None)]
        reference_index += [depth_axes[name] if name in depth_axes else slice(None) for name in axes]
        reference = luminosity[emitter][tuple(reference_index)]
        reference_noise = noise[emitter][tuple(reference_index)]
        # Broadcast the reference back over the optical depth axes it was taken at.
        for position, name in enumerate(axes):
            if name in depth_axes:
                reference = np.expand_dims(reference, 2 + position)
                reference_noise = np.expand_dims(reference_noise, 2 + position)
        attenuation[emitter], uncertainty[emitter] = attenuation_of(
            luminosity[emitter], noise[emitter], reference, reference_noise
        )

        along = campaign.axes[0].name if campaign.axes else None
        position = axes.index(along) if along in axes else None
        if position is None:
            raise ValueError(
                f"emitter {emitter!r} is not tabulated against {along!r}, so there is "
                "nothing to extrapolate along"
            )
        moved = np.moveaxis(attenuation[emitter], 2 + position, 0)
        coefficients, scatter = fit_extrapolation(axis_values[along], moved)
        extrapolation[emitter] = coefficients
        residual[emitter] = scatter

    return Tabulation(
        label=campaign.config.label,
        wavelengths=wavelengths,
        inclinations=inclinations,
        axis_names=tuple(axis.name for axis in campaign.axes),
        axis_values=axis_values,
        emitter_axes=emitter_axes,
        attenuation=attenuation,
        uncertainty=uncertainty,
        extrapolation=extrapolation,
        residual=residual,
        opacity=opacity,
        metadata=metadata or {},
    )
