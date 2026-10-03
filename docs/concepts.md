# Concepts and conventions

## A galaxy is a set of components

Each component may carry a stellar profile, a dust profile, both, or neither.
Attenuation is tabulated separately for each component with stars, since it is
the fraction of *that* component's light which escapes.

This is why putting dust in the spheroid as well as the disk is a change to a
configuration rather than to any code: give the spheroid component a `dust`
block, and it gains an optical depth axis which every emitter is then tabulated
against.

A component's stars and its dust are separate profiles, and need not have the
same shape or size. Configured over the same values they become a single axis;
over different values they stay separate.

## Optical depth is defined along a ray, and which ray matters

Models are normalized by optical depth rather than by mass, so each profile has
to say what its optical depth *means*.

For a **disk** it is the depth through the centre viewed face-on, from
{math}`-\infty` to {math}`+\infty`. Both vertical structures give a central
column of {math}`2h_z`, so the normalization depends on the scale height but not
on which structure is used — a coincidence of those two profiles, not a rule.

For a **spheroid** the density diverges at the centre, so the central column is
infinite and cannot be used. The optical depth is instead measured along a ray
at the **scale radius**. A spheroid optical depth is therefore not a central
optical depth, and the two are not interchangeable.

Truncating a spheroid changes this: the ray leaves the truncation sphere at
{math}`|z| = r_\mathrm{s}\sqrt{t^2-1}`, so the column is over a finite range and
no longer the untruncated constant.

## The grid has to resolve the smallest scale and reach the largest

The models are axisymmetric, so they are solved on a cylindrical grid:
logarithmic in radius, symmetric in height, one azimuthal cell.

`spacing: published` reproduces the original: vertical walls uniformly spaced,
and the innermost radial wall a hundredth of the *largest* radial scale. That
cannot resolve a thin disk inside a large spheroid. With a spheroid ten times
the disk scale length it leaves vertical cells fifteen dust scale heights thick,
and a model asked for an optical depth of one ends up with one of zero — the
dust falls between the cell centres.

`spacing: nested` spaces the vertical walls uniformly out to the cut off of the
*smallest* vertical scale and logarithmically beyond, and starts the radial grid
from the smallest radial scale. It holds the requested optical depth at every
spheroid size tested, out to a hundred disk scale lengths.

Hyperion's AMR grid was not used for this: it is Cartesian, so it would cost the
cylindrical symmetry these models have and the exact cell mass integrals that go
with it. The cylindrical grid already accepts arbitrary wall positions.

## Dust density is sampled, and how matters for cuspy profiles

`sampling: centre` evaluates the profile at each cell centre, as the original
did. That is a point sample, not a cell average, so the grid does not hold the
mass the profile has. For a disk this costs 0.08%, which is why it has never
mattered. For a Jaffe spheroid it costs 7%, and up to 95% in the innermost cell.

`sampling: average` gives each cell the density which puts the profile's true
mass in it, using the exact cell mass integrals.

## Extrapolating beyond the tabulated optical depth

Beyond the largest tabulated depth the transmission is extrapolated as
{math}`\exp(c_0 + c_1 \ln\tau)`, fitted over the top decade. With dust in more
than one component the fit is made along the *disk's* optical depth at each
value of the others, so a spheroid's own optical depth stays bounded by the
table rather than being extrapolated through.

The root mean square residual of that fit is stored beside the coefficients, so
the assumption can be checked rather than trusted. It is worth checking: the
power law is motivated by a disk, where at high optical depth only a surface
layer is seen, and how well it holds depends on {math}`\tau_\lambda` rather than
{math}`\tau_\mathrm{V}`. On a grid reaching {math}`\tau_\mathrm{V}=50` the fit
is good for a disk but poor for a small spheroid in the near infrared, where the
actual optical depth is an order of magnitude smaller and nowhere near
asymptotic. On the compendium grid, which reaches {math}`10^4`, it is on much
firmer ground.

The residual says nothing when only two depths fall in the fitting range: a
straight line through two points passes through both.

## Noise, and the two azimuths

The original Benson calculations requested every inclination twice, from
opposite azimuths. This package keeps those directions as its default for
reproducibility, and permits an explicit list under ``tabulation.azimuths``.

For an axisymmetric model the directions have the same physical expectation,
but Hyperion estimates both from the same photon histories. They are therefore
correlated, and each extra direction also requires another peel-off calculation.
It is not justified to assert that averaging two directions reduces the noise by
{math}`\sqrt{2}` without measuring their covariance. Tabulation averages use the
fully-correlated (conservative) uncertainty limit. Use the individual directions
and repeated random seeds to test whether one, two or more azimuths are the most
efficient choice at fixed computational cost.

Dividing by the model with no dust — same geometry, same seed — cancels much of
what remains.

One quantity to treat carefully: the reddening
{math}`R_\mathrm{V} = A_V/(A_B-A_V)` has a denominator which is a difference of
two nearly equal extinctions. Where the dust is thin that difference is
comparable to the noise on either, and the ratio scatters wildly — it can come
out negative. A reddening plotted from the low optical depth end says more about
the photon count than about the dust.
