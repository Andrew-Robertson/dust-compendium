r"""The cylindrical grid the models are built on.

Reproduces the grid of the original ``hyperionBuildModel.py``: logarithmic in
radius with an extra wall on the axis, linear and symmetric in height, and a
single azimuthal cell, the models having cylindrical symmetry.

Cell centres follow Hyperion's own convention, because that is where the dust
density is sampled and a different convention would quietly change the optical
depth of every cell. Radially the centre is the *geometric* mean of the two
walls, except in the innermost cell when the axis is a wall, where the
logarithm is undefined and Hyperion uses half the outer wall instead.
Vertically it is the arithmetic mean. :func:`CylindricalGrid.shape` and the
array layout also match Hyperion's, which indexes cells as
``(azimuth, height, radius)``.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from .galaxy import Galaxy

__all__ = ["CylindricalGrid"]

#: First positive radial wall as a fraction of the selected radial scale
#: (not the cut-off radius). From ``hyperionBuildModel.py`` line 101.
INNER_RADIUS_FRACTION = 1.0e-2


def _nested_walls(inner: float, outer: float, cells: int) -> NDArray[np.float64]:
    """Vertical walls uniform within ``inner`` and logarithmic out to ``outer``.

    Symmetric about the midplane, which is always a wall, so no cell straddles
    it. Half the cells on each side, and those split evenly between the uniform
    core and the logarithmic wings.
    """
    half = max(cells // 2, 2)
    core = max(half // 2, 1)
    wings = half - core
    positive = np.linspace(0.0, inner, core + 1)
    if wings > 0:
        positive = np.hstack([positive, np.logspace(np.log10(inner), np.log10(outer), wings + 1)[1:]])
    return np.hstack([-positive[::-1], positive[1:]])


@dataclass(frozen=True)
class CylindricalGrid:
    """Cell walls of a cylindrical polar grid, and the quantities derived from them.

    Parameters
    ----------
    radial_walls
        Increasing radii, starting at zero.
    vertical_walls
        Increasing heights, symmetric about zero.
    azimuthal_walls
        Increasing azimuths, spanning a full turn.
    """

    radial_walls: NDArray[np.float64]
    vertical_walls: NDArray[np.float64]
    azimuthal_walls: NDArray[np.float64]

    def __post_init__(self) -> None:
        for name in ("radial_walls", "vertical_walls", "azimuthal_walls"):
            walls = np.asarray(getattr(self, name), dtype=float)
            object.__setattr__(self, name, walls)
            if walls.ndim != 1 or walls.size < 2:
                raise ValueError(f"{name} must be a one-dimensional array of at least two walls")
            if not np.all(np.diff(walls) > 0.0):
                raise ValueError(f"{name} must be strictly increasing")
        if self.radial_walls[0] < 0.0:
            raise ValueError("radial walls must be non-negative")

    @classmethod
    def for_galaxy(
        cls,
        galaxy: Galaxy,
        cut_off: float,
        radial_cells: int = 100,
        vertical_cells: int = 100,
        spacing: str = "published",
        inner_radius_fraction: float = INNER_RADIUS_FRACTION,
    ) -> "CylindricalGrid":
        r"""Build a grid large enough to hold every component of a galaxy.

        The extent is the largest scale among the components in each direction,
        multiplied by ``cut_off``, as in the original. Note that a spheroid
        contributes its scale radius vertically as well as radially, so a
        spheroid larger than the disk's scale height sets the vertical extent.

        Parameters
        ----------
        galaxy
            Supplies the radial and vertical extents.
        cut_off
            How many scale lengths out to truncate, positive.
        radial_cells
            Number of radial cells. One of them is the innermost cell reaching
            the axis, so there are ``radial_cells - 1`` logarithmic walls beyond
            the two innermost.
        vertical_cells
            Number of vertical cells. An *even* number puts a wall on the
            midplane, which is what the published grids do and what keeps any
            cell from straddling it.
        spacing
            ``published`` reproduces the original grid exactly: vertical walls
            spaced uniformly, and the innermost radial wall a hundredth of the
            *largest* radial scale. ``nested`` instead resolves the smallest
            scale in each direction, placing the innermost radial wall a
            hundredth of the smallest radial scale, and spacing vertical walls
            uniformly out to the cut off of the smallest vertical scale and
            logarithmically beyond. The two agree when every component is the
            same size.

            The published scheme cannot resolve a thin disk inside a large
            spheroid: at a spheroid ten times the disk scale length it leaves
            cells fifteen dust scale heights thick, and the dust all but
            vanishes from the model. See
            :func:`~dustcompendium.model.dust_density`.

        inner_radius_fraction
            First positive radial wall, in units of the radial scale selected
            by ``spacing``. Defaults to 0.01 for backwards compatibility.
            Refine this as well as the cell counts to test cusp resolution.

        Raises
        ------
        ValueError
            If the cut off or the cell counts are not positive, or if
            ``spacing`` is not one of the two schemes.
        """
        if spacing not in ("published", "nested"):
            raise ValueError(f"spacing must be 'published' or 'nested', got {spacing!r}")
        if cut_off <= 0.0:
            raise ValueError(f"cut off must be positive, got {cut_off}")
        if radial_cells < 2 or vertical_cells < 1:
            raise ValueError("need at least two radial cells and one vertical cell")
        if not np.isfinite(inner_radius_fraction) or not 0 < inner_radius_fraction < cut_off:
            raise ValueError("inner_radius_fraction must be finite, positive and below cut_off")
        extent_radial = cut_off * galaxy.extent_radial
        extent_vertical = cut_off * galaxy.extent_vertical
        smallest_radial = galaxy.radial_scales[0] if spacing == "nested" else galaxy.extent_radial
        radial = np.hstack(
            [
                0.0,
                np.logspace(
                    np.log10(inner_radius_fraction * smallest_radial),
                    np.log10(extent_radial),
                    radial_cells,
                ),
            ]
        )
        scales = galaxy.vertical_scales
        if spacing == "nested" and scales[0] < scales[-1]:
            walls = _nested_walls(cut_off * scales[0], extent_vertical, vertical_cells)
        else:
            walls = np.linspace(-extent_vertical, extent_vertical, vertical_cells + 1)
        azimuthal = np.linspace(0.0, 2.0 * np.pi, 2)
        return cls(radial, walls, azimuthal)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Cell counts as ``(azimuth, height, radius)``, matching Hyperion's layout."""
        return (
            self.azimuthal_walls.size - 1,
            self.vertical_walls.size - 1,
            self.radial_walls.size - 1,
        )

    @cached_property
    def radial_centres(self) -> NDArray[np.float64]:
        """Geometric mean of the walls, or half the outer wall against the axis."""
        walls = self.radial_walls
        if walls[0] == 0.0:
            centres = np.empty(walls.size - 1)
            centres[0] = walls[1] / 2.0
            centres[1:] = 10.0 ** ((np.log10(walls[1:-1]) + np.log10(walls[2:])) / 2.0)
            return centres
        return 10.0 ** ((np.log10(walls[:-1]) + np.log10(walls[1:])) / 2.0)

    @cached_property
    def vertical_centres(self) -> NDArray[np.float64]:
        """Arithmetic mean of the walls."""
        return (self.vertical_walls[:-1] + self.vertical_walls[1:]) / 2.0

    @cached_property
    def azimuthal_centres(self) -> NDArray[np.float64]:
        """Arithmetic mean of the walls."""
        return (self.azimuthal_walls[:-1] + self.azimuthal_walls[1:]) / 2.0

    @cached_property
    def centres(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Radius and height at every cell centre, each with :attr:`shape`."""
        radius = np.broadcast_to(self.radial_centres[None, None, :], self.shape)
        height = np.broadcast_to(self.vertical_centres[None, :, None], self.shape)
        return radius, height

    @cached_property
    def bounds(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Lower and upper radius and height of every cell, each with :attr:`shape`.

        Ready to hand to :meth:`~dustcompendium.geometry.Profile.cell_mass`.
        """
        return np.broadcast_arrays(
            self.radial_walls[None, None, :-1],
            self.radial_walls[None, None, 1:],
            self.vertical_walls[None, :-1, None],
            self.vertical_walls[None, 1:, None],
        )

    @cached_property
    def cell_volumes(self) -> NDArray[np.float64]:
        r"""Volume of every cell, :math:`\tfrac{1}{2}\Delta\phi (R_+^2 - R_-^2) \Delta z`."""
        radius_lower, radius_upper, height_lower, height_upper = self.bounds
        azimuth = np.broadcast_to(np.diff(self.azimuthal_walls)[:, None, None], self.shape)
        return 0.5 * azimuth * (radius_upper**2 - radius_lower**2) * (height_upper - height_lower)

    @property
    def midplane_is_a_wall(self) -> bool:
        """Whether no cell straddles the midplane, which the published grids ensure."""
        return bool(np.any(self.vertical_walls == 0.0))
