# Dust Compendium

Dust attenuation tables for simple galactic geometries, computed by running the
[Hyperion](http://www.hyperion-rt.org) Monte Carlo radiative transfer code over
a grid of galaxy models.

A tabulation gives the fraction of starlight escaping a galaxy as a function of
wavelength, inclination, the optical depth of each dusty component, and the
sizes of the components themselves. [Galacticus](https://github.com/galacticusorg/galacticus)
reads them through its `dustAttenuationAtlasCompendium` class. The method is
described in [Benson (2018)](https://ui.adsabs.harvard.edu/abs/2018RNAAS...2..188B).

```{toctree}
:maxdepth: 2

installation
usage
hpc
concepts
format
datasets
extending
api
```

## In short

```bash
dust-compendium validate configs/compendium-d03-rv3.1.yaml   # what would this run?
dust-compendium build    configs/compendium-d03-rv3.1.yaml   # write the model inputs
dust-compendium run      configs/compendium-d03-rv3.1.yaml   # solve them
dust-compendium collect  configs/compendium-d03-rv3.1.yaml   # assemble the tabulation
dust-compendium plot     attenuations.hdf5 -q curve -c opticalDepth
```

Each step reads what the one before it left on disk, so any of them can be
interrupted and run again. Only `build` needs Hyperion's Python package; only
`run` needs its solver binaries; the rest need neither.

## Provenance

This package replaces a set of Perl and Python 2 scripts written for the 2018
paper. Those are preserved verbatim in the first commit of this repository, and
the docstrings here refer to them by file and line where behaviour was inherited
from them or deliberately departed from. Several defects were found in them
along the way, some of which reached published data; they are listed under
[Published datasets](datasets.md).
