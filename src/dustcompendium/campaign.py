r"""Expanding a configuration into the individual models that have to be run.

A campaign tabulates attenuation over a grid of parameters. Every parameter
written as a range in the configuration becomes an *axis*: the optical depth of
each dust-bearing component, and any geometric scale that varies. One model is
run for each point on that grid and for each component with stars, since the
attenuation of each is tabulated separately.

The published compendium is 60 optical depths by 51 spheroid sizes by two
emitting components, which is 6,120 models -- why this wants a cluster.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray

from .config import CampaignConfig, ProfileConfig, values_of
from .galaxy import Component, Galaxy
from .model import ModelSpec

__all__ = ["Axis", "Campaign", "Run"]

#: Geometric parameters which may vary, and so become axes.
GEOMETRIC = ("scale_radial", "scale_height")


@dataclass(frozen=True)
class Axis:
    """One tabulated parameter, and the values it takes."""

    kind: str
    """``opticalDepth``, ``scaleRadial`` or ``scaleHeight``."""
    component: str
    """Which component it belongs to."""
    roles: frozenset[str]
    """Which of ``stellar`` and ``dust`` this axis sets.

    A component's stars and its dust may have different scale lengths -- the
    original already gave the disk separate stellar and dust scale heights -- so
    the two are separate parameters. When they are configured over the same
    values they are one axis, driving both, rather than a square grid of a scale
    against itself.
    """
    values: NDArray[np.float64]
    qualified: bool = False
    """Whether the name needs a role to tell it from another axis of the same parameter."""

    @property
    def name(self) -> str:
        """A stable name, used for labels and for output dataset names."""
        if self.qualified:
            return f"{self.kind}:{self.component}:{sorted(self.roles)[0]}"
        return f"{self.kind}:{self.component}"

    def affects(self, emitter: str) -> bool:
        """Whether this axis can change what the given emitter's light looks like.

        Attenuation depends on every dust distribution, but only on the stellar
        distribution of the component being observed. So a geometric axis
        belonging to another component's *stars* changes nothing, and tabulating
        against it would repeat the same model. The original arranged this by
        hand, pinning the spheroid scale to a single value when the disk was
        emitting, which is why ``attenuationDisk`` carries no spheroid axis.
        """
        return "dust" in self.roles or self.component == emitter

    def __len__(self) -> int:
        return int(self.values.size)


@dataclass(frozen=True)
class Run:
    """One model: a point on the parameter grid, seen from one emitting component."""

    emitter: str
    indices: tuple[int, ...]
    """Index into each of :attr:`Campaign.axes`, in order."""
    spec: ModelSpec
    label: str

    @property
    def file_stem(self) -> str:
        """A file name stem, with the characters a path would rather not carry."""
        return self.label.replace(":", "_").replace("/", "_").replace(" ", "_")


class Campaign:
    """The set of models a configuration describes.

    Parameters
    ----------
    config
        The parsed configuration.
    """

    def __init__(self, config: CampaignConfig) -> None:
        self.config = config
        self.axes = tuple(self._axes())
        self.wavelengths = values_of(config.tabulation.wavelengths)
        self.inclinations = values_of(config.tabulation.inclinations)
        self.azimuths = values_of(config.tabulation.azimuths)

    def _axes(self) -> Iterator[Axis]:
        for name, component in self.config.geometry.components.items():
            roles = (("stellar", component.stellar), ("dust", component.dust))
            for role, profile in roles:
                if profile is None or role != "dust":
                    continue
                if profile.optical_depth is not None:
                    yield Axis("opticalDepth", name, frozenset({role}), values_of(profile.optical_depth))
            for parameter in GEOMETRIC:
                kind = parameter.replace("_r", "R").replace("_h", "H")
                # Group the roles which vary this parameter over identical
                # values; each distinct set of values is one axis.
                groups: list[tuple[NDArray[np.float64], set[str]]] = []
                for role, profile in roles:
                    if profile is None or getattr(profile, parameter, None) is None:
                        continue
                    values = values_of(getattr(profile, parameter))
                    if values.size <= 1:
                        continue
                    for existing, members in groups:
                        if existing.shape == values.shape and np.array_equal(existing, values):
                            members.add(role)
                            break
                    else:
                        groups.append((values, {role}))
                for values, members in groups:
                    yield Axis(kind, name, frozenset(members), values, qualified=len(groups) > 1)

    @property
    def emitters(self) -> tuple[str, ...]:
        """Components with stars, each of which is tabulated separately."""
        return tuple(
            name
            for name, component in self.config.geometry.components.items()
            if component.stellar is not None
        )

    def axes_for(self, emitter: str) -> tuple[Axis, ...]:
        """The axes the given emitter is actually tabulated against."""
        return tuple(axis for axis in self.axes if axis.affects(emitter))

    def shape_for(self, emitter: str) -> tuple[int, ...]:
        """Length of each axis the given emitter is tabulated against."""
        return tuple(len(axis) for axis in self.axes_for(emitter))

    @property
    def shape(self) -> tuple[int, ...]:
        """Length of each axis, in order."""
        return tuple(len(axis) for axis in self.axes)

    def __len__(self) -> int:
        """How many models the campaign contains."""
        return sum(int(np.prod(self.shape_for(emitter), dtype=int)) for emitter in self.emitters)

    def _chosen(
        self, indices: Sequence[int], axes: Sequence[Axis] | None = None
    ) -> dict[tuple[str, str, str], float]:
        """The value each axis takes at a point, keyed by component, role and parameter."""
        chosen = {}
        for axis, index in zip(axes if axes is not None else self.axes, indices, strict=True):
            parameter = {"scaleRadial": "scale_radial", "scaleHeight": "scale_height"}.get(
                axis.kind, "optical_depth"
            )
            for role in axis.roles:
                chosen[(axis.component, role, parameter)] = float(axis.values[index])
        return chosen

    def galaxy(
        self, indices: Sequence[int], axes: Sequence[Axis] | None = None, emitter: str | None = None
    ) -> Galaxy:
        """The galaxy at a point on the parameter grid, as one emitter sees it.

        A component which neither emits for this run nor carries dust cannot
        change the answer, and is left out. That is not merely an economy: a
        component included needlessly still sizes the grid, and a large spheroid
        sizing the grid for a disk-only model is what leaves the disk
        unresolved. The original arranged the same thing by hand, pinning the
        spheroid scale to zero whenever the disk was emitting.
        """
        chosen = self._chosen(indices, axes)

        def build(name: str, role: str, profile: ProfileConfig | None):
            if profile is None:
                return None
            overrides = {
                parameter: value
                for (component, kind, parameter), value in chosen.items()
                if component == name and kind == role and parameter in GEOMETRIC
            }
            return profile.build(**overrides)

        components = []
        for name, component in self.config.geometry.components.items():
            emits = emitter is None or name == emitter
            if not emits and component.dust is None:
                continue
            components.append(
                Component(
                    name,
                    stellar=build(name, "stellar", component.stellar) if emits else None,
                    dust=build(name, "dust", component.dust),
                )
            )
        return Galaxy(components)

    def optical_depths(self, indices: Sequence[int], axes: Sequence[Axis] | None = None) -> dict[str, float]:
        """The optical depth of each dust-bearing component at a point."""
        chosen = self._chosen(indices, axes)
        return {
            component: value
            for (component, _role, parameter), value in chosen.items()
            if parameter == "optical_depth"
        }

    def runs(self) -> Iterator[Run]:
        """Every model in the campaign, in a stable order.

        With the default ``model`` seed policy, the seed is decremented per
        model as the original did. With ``geometry``, every optical-depth
        variant of one emitter and geometry shares a seed. This makes each
        attenuated model share photon histories with the dust-free model used
        to normalize it.
        """
        tabulation = self.config.tabulation
        seed = tabulation.seed
        geometry_seeds: dict[tuple[str, tuple[tuple[str, int], ...]], int] = {}
        for emitter in self.emitters:
            axes = self.axes_for(emitter)
            for indices in product(*(range(len(axis)) for axis in axes)):
                if tabulation.seed_policy == "model":
                    seed -= 1
                    run_seed = seed
                else:
                    geometry = tuple(
                        (axis.name, index)
                        for axis, index in zip(axes, indices, strict=True)
                        if axis.kind != "opticalDepth"
                    )
                    key = (emitter, geometry)
                    if key not in geometry_seeds:
                        seed -= 1
                        geometry_seeds[key] = seed
                    run_seed = geometry_seeds[key]
                suffix = "_".join(str(index) for index in indices)
                yield Run(
                    emitter=emitter,
                    indices=indices,
                    label=f"{self.config.label}_{emitter}" + (f"_{suffix}" if suffix else ""),
                    spec=ModelSpec(
                        galaxy=self.galaxy(indices, axes, emitter),
                        emitter=emitter,
                        optical_depths=self.optical_depths(indices, axes),
                        wavelengths=self.wavelengths,
                        inclinations=self.inclinations,
                        azimuths=self.azimuths,
                        cut_off=self.config.geometry.cut_off,
                        photons=tabulation.photons,
                        imaging_photons=tabulation.imaging_photons,
                        raytracing_photons=tabulation.raytracing_photons,
                        seed=run_seed,
                    ),
                )

    def grid_options(self) -> Mapping[str, object]:
        """Grid options from the configuration, to pass when building a model."""
        geometry = self.config.geometry
        return {
            "spacing": geometry.spacing,
            "radial_cells": geometry.radial_cells,
            "vertical_cells": geometry.vertical_cells,
            "inner_radius_fraction": geometry.inner_radius_fraction,
        }

    def __repr__(self) -> str:
        axes = ", ".join(f"{axis.name}[{len(axis)}]" for axis in self.axes)
        return f"Campaign({self.config.label!r}, {len(self)} models, axes: {axes})"
