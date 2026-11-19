"""Study-only central dust modifications; no change to production profiles."""

import numpy as np
from scipy.integrate import quad

from dustcompendium.galaxy import Component, Galaxy
from dustcompendium.geometry import ExponentialDisk, HernquistSpheroid, SechSquaredVertical
from dustcompendium.grid import CylindricalGrid


class InnerHernquist(HernquistSpheroid):
    """Flatten/empty only spheroid dust inside epsilon*a; conserve total dust mass.

    Outer cutoff is 10a. Input tau retains the ORIGINAL Hernquist mass calibration;
    actual b=a depth changes by the tiny mass-renormalization factor. Stellar
    emission remains the unmodified Hernquist distribution.
    """

    def __init__(self, a, kind="cusp", epsilon=0.0):
        super().__init__(a, truncation=10.0)
        if kind not in ("cusp", "core", "cavity"):
            raise ValueError(kind)
        if kind != "cusp" and not 0 < epsilon < 1:
            raise ValueError("need 0 < epsilon < 1")
        self.kind, self.epsilon = kind, float(epsilon)
        f = (10 / 11) ** 2
        removed = 0.0 if kind == "cusp" else (epsilon / (1 + epsilon)) ** 2
        if kind == "core":
            removed -= 2 * epsilon**2 / (3 * (1 + epsilon) ** 3)
        self.mass_factor = f / (f - removed)

    def density(self, radius, height):
        value = super().density(radius, height)
        if self.kind != "cusp":
            r = np.hypot(radius, height) / self.scale_radial
            value = np.where(
                r < self.epsilon, 0.0 if self.kind == "cavity" else self._shape(self.epsilon), value
            )
        return self.mass_factor * value

    def cell_mass(self, rl, ru, zl, zu):
        original = super().cell_mass(rl, ru, zl, zu)
        if self.kind == "cusp":
            return original
        a = self.scale_radial
        e = self.epsilon
        bounds = np.broadcast_arrays(
            np.asarray(rl) / a, np.asarray(ru) / a, np.asarray(zl) / a, np.asarray(zu) / a
        )
        xl, _xu, ul, uu = bounds
        znear = np.where((ul <= 0) & (uu >= 0), 0, np.minimum(abs(ul), abs(uu)))
        touches = np.hypot(xl, znear) < e
        result = np.array(original, copy=True)
        for idx in zip(*np.nonzero(np.atleast_1d(touches)), strict=True):
            # Inputs here are arrays with the grid shape; scalar inputs also work.
            flat = np.ravel_multi_index(idx, np.atleast_1d(touches).shape)
            lower_r, upper_r, b, t = [float(v.flat[flat]) for v in bounds]
            lo, hi = max(b, -e), min(t, e)

            def removed(u, lower_r=lower_r, upper_r=upper_r):
                inner = min(np.hypot(lower_r, u), e)
                outer = min(np.hypot(upper_r, u), e)
                old = self._enclosed_annulus(outer) - self._enclosed_annulus(inner)
                new = 0 if self.kind == "cavity" else self._shape(e) * (outer**2 - inner**2) / 2
                return old - new

            points = [0.0] if lo < 0 < hi else []
            for wall in (lower_r, upper_r):
                if wall < e:
                    cross = np.sqrt(e * e - wall * wall)
                    points.extend(v for v in (-cross, cross) if lo < v < hi)
            correction = (
                2
                * np.pi
                * a**3
                * quad(
                    removed, lo, hi, points=sorted(set(points)) or None, epsabs=1e-15, epsrel=1e-10, limit=100
                )[0]
            )
            # Exact zero in cells entirely inside a cavity avoids subtraction noise.
            far = np.hypot(upper_r, max(abs(b), abs(t)))
            result.flat[flat] = (
                0.0 if self.kind == "cavity" and far <= e else original.flat[flat] - correction
            )
        if np.any(result < -1e-11 * np.max(original)):
            raise ValueError("negative modified cell mass")
        return np.maximum(result, 0) * self.mass_factor

    def central_column(self):
        """One-sided analytic shape column; multiply by original density normalization."""
        if self.kind == "cusp":
            return np.inf
        e = self.epsilon

        def primitive(x):
            return np.log1p(1 / x) - 1 / (1 + x) - 1 / (2 * (1 + x) ** 2)

        value = primitive(e) - primitive(10.0)
        if self.kind == "core":
            value += 1 / (1 + e) ** 3
        return self.scale_radial * self.mass_factor * value


def galaxy(q, kind="cusp", epsilon=0):
    disk = ExponentialDisk(1.0, SechSquaredVertical(0.137))
    return Galaxy(
        [
            Component("disk", stellar=disk, dust=disk),
            Component(
                "spheroid",
                stellar=HernquistSpheroid(q, truncation=10.0),
                dust=InnerHernquist(q, kind, epsilon),
            ),
        ]
    )


def grid_for(g, level):
    base = CylindricalGrid.for_galaxy(
        g, 10.0, radial_cells=100, vertical_cells=100, spacing="nested", inner_radius_fraction=0.01
    )
    if level == "legacy":
        return base
    resolution = {"inner8": 8, "inner16": 16}[level]
    a = g["spheroid"].dust.scale_radial

    # Identical walls across physical profiles. Keep EVERY legacy wall. Resolve
    # the smallest cavity in R and z, plus logarithmic transition to legacy cells.
    def extra(first):
        end = max(first, 0.03 * a)
        start = 0.003 * a / resolution
        count = int(np.ceil(np.log(end / start) / np.log(1 + 1 / resolution))) + 1
        return np.unique(np.r_[np.geomspace(start, end, count), 0.003 * a, 0.01 * a])

    radial = np.unique(np.r_[base.radial_walls, extra(base.radial_walls[1])])
    pos = base.vertical_walls[base.vertical_walls > 0]
    z = extra(pos[0])
    vertical = np.unique(np.r_[base.vertical_walls, z, -z])
    return CylindricalGrid(radial, vertical, base.azimuthal_walls)


def central_grid_column(density, grid, angle):
    """Exact integral through piecewise constant cells along a central ray."""
    s, c = np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))
    tmax = min(
        grid.radial_walls[-1] / s if s > 1e-14 else np.inf,
        grid.vertical_walls[-1] / c if c > 1e-14 else np.inf,
    )
    times = np.r_[
        0,
        tmax,
        grid.radial_walls / s if s > 1e-14 else [],
        grid.vertical_walls[grid.vertical_walls >= 0] / c if c > 1e-14 else [],
    ]
    times = np.unique(times[(times >= 0) & (times <= tmax)])
    mid = (times[:-1] + times[1:]) / 2
    ir = np.searchsorted(grid.radial_walls, mid * s, side="right") - 1
    iz = np.searchsorted(grid.vertical_walls, mid * c, side="right") - 1
    return np.sum(np.diff(times) * density[0, iz, ir])
