# Changelog

## 0.1.0 (unreleased)

First release as a package. Replaces the Perl and Python 2 scripts used for
[Benson (2018)](https://ui.adsabs.harvard.edu/abs/2018RNAAS...2..188B), which
are preserved verbatim in the first commit of this repository.

### What it does

- Galaxies are a set of named components, each of which may carry a stellar
  profile, a dust profile, both or neither, so dust in the spheroid as well as
  the disk is a change to a configuration rather than to any code.
- Campaigns are described in YAML and validated on load. Any value written as a
  range becomes an axis of the tabulation. `validate` reports what a campaign
  would run without needing Hyperion.
- `build`, `run` and `collect` each resume from what the last left on disk.
  `run` has local and Slurm backends; PBS is not carried over.
- The output format is versioned, self-describing, and reproduces the layout
  Galacticus reads.
- `plot` replaces nine near-identical plotting scripts, and drops the gnuplot,
  cairolatex and pdflatex chain. `compare` measures agreement against a
  published atlas.
- The core — geometry, optical depths, the reader and writer, post-processing,
  plotting and comparison — needs no Hyperion at all.

### Fixed, relative to the original scripts

Six defects, found by testing the ported code against numerical quadrature and
against Ferrara et al. (1999). Two of them reached published data; see the
documentation for which tabulations are affected.

- The spheroid cell-mass antiderivative carried the wrong sign below the
  midplane at the `R = 0`, `|z| = r_s` corner, which the compendium grids reach.
  For a spheroid the size of the disk this put 54% of its light into two cells
  holding 0.0003% of its mass.
- The recorded inclination axis was always a uniform grid, even when the models
  were solved at other angles, which the Ferrara-matched tabulations were.
- Cell corners were combined through nested absolute values, so a cell
  straddling the midplane was wrong and one symmetric about it came out empty.
- The spheroid antiderivatives were unhandled on the sphere `r = r_s`, returned
  NaN at small non-zero radius, and lost all precision near `R = r_s`.
- `spheroidCutOff` was parsed and never applied; spheroids can now be truncated.
- The source spectrum spanned exactly the dust's frequency range, so a photon
  drawn at its edge aborted the solve. Hyperion exits zero when it aborts, so
  those models were recorded as successes; solved models are now checked for the
  SEDs they were meant to produce.

### Added beyond the original

- `spacing: nested`, which resolves a thin disk inside a large spheroid. The
  published grid cannot: at a spheroid ten times the disk scale length it leaves
  a model asked for an optical depth of one with one of zero.
- `sampling: average`, which puts the profile's true mass in each cell. Point
  sampling loses 7% of a Jaffe dust profile, and up to 95% in the innermost cell.
- The extrapolation fit's residual is stored beside its coefficients, so the
  power-law assumption can be checked rather than trusted.
- Uncertainties propagate from both the attenuated model and the reference.
- A configurable `geometry` seed policy shares photon histories between all
  optical-depth variants of one geometry, reducing noise in attenuation ratios
  while leaving the legacy per-model policy as the default.
- A small D03 R_V=4 tied disk+spheroid smoke campaign and an HPC execution
  guide exercise the complete build, serial Slurm, and collection path.
- Collected files retain the complete serialized campaign configuration, so
  fixed as well as varying geometric assumptions remain recoverable.
- Every Hyperion output is validated for peeled SEDs on its execution node,
  turning Hyperion's misleading zero-status aborts into scheduler failures.
  Successful jobs then receive a bounded submission-node visibility window
  longer than NFS's usual metadata-cache maximum.
