# Dust Compendium

[![CI](https://github.com/galacticusorg/dust-compendium/actions/workflows/ci.yml/badge.svg)](https://github.com/galacticusorg/dust-compendium/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/dust-compendium/badge/?version=latest)](https://dust-compendium.readthedocs.io)

Dust attenuation tables for simple galactic geometries, computed by running the
[Hyperion](http://www.hyperion-rt.org) Monte Carlo radiative transfer code over
a grid of galaxy models.

A tabulation gives the fraction of starlight escaping a galaxy as a function of
wavelength, inclination, the optical depth of each dusty component, and the
sizes of the components themselves. [Galacticus](https://github.com/galacticusorg/galacticus)
reads them through its `dustAttenuationAtlasCompendium` class. The method is
described in [Benson (2018)](https://ui.adsabs.harvard.edu/abs/2018RNAAS...2..188B).

Full documentation is at [dust-compendium.readthedocs.io](https://dust-compendium.readthedocs.io).

## Installation

```bash
pip install dust-compendium
```

That gives the library and the command line tool. Computing *new* tabulations
additionally needs Hyperion, in two parts, neither of which is a dependency of
this package:

- its **Python front-end**, which has to come from source for now — the current
  PyPI release calls `np.string_` when writing a model, which NumPy 2 removed;
- its **Fortran solvers**, which `pip` does not install at all.

See [the installation notes](https://dust-compendium.readthedocs.io/en/latest/installation.html).

## Using it

```bash
dust-compendium validate configs/compendium-d03-rv3.1.yaml   # what would this run?
dust-compendium build    configs/compendium-d03-rv3.1.yaml   # write the model inputs
dust-compendium run      configs/compendium-d03-rv3.1.yaml   # solve them, here or on Slurm
dust-compendium collect  configs/compendium-d03-rv3.1.yaml   # assemble the tabulation
dust-compendium plot     attenuations.hdf5 -q curve -c opticalDepth
dust-compendium compare  attenuations.hdf5 --atlas /path/to/atlas.hdf5
```

A campaign is described by a YAML file. Any value written as a range becomes an
axis of the tabulation; a galaxy is a set of named components, each of which may
carry stars, dust, both or neither. Each step reads what the one before it left
on disk, so any of them can be interrupted and resumed.

`configs/` holds the configuration for the published Draine R_V=3.1 tabulation,
one adding dust to the spheroid, and one matched to Ferrara et al. (1999) for
validation. It also contains a small Draine R_V=4 tied disk+spheroid smoke
campaign for checking a new Hyperion/HPC installation before designing a full
emulator campaign. See the [HPC guide](docs/hpc.md).

## Validation

Reproducing Ferrara et al. (1999) with matched grains, geometry and grids gives
a median absolute difference from their published atlas of **0.0034** in disk
transmission, with 99.9% of entries within 0.02 — as close as the original
scripts got, and near the 0.005 floor set by the atlas being published to two
decimal places.

## Provenance

This replaces the Perl and Python 2 scripts written for the 2018 paper, which
are preserved verbatim in the first commit of this repository. Rewriting them
turned up several defects, two of which reached published data; those are
described under
[Published datasets](https://dust-compendium.readthedocs.io/en/latest/datasets.html).

## License

MIT. See [LICENSE](LICENSE).
