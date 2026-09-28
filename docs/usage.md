# Running a campaign

A *campaign* is one configuration file and the thousands of radiative transfer
models it expands into. Five commands take it from the file to a tabulation.

## Describing what to compute

Lengths are in units of the disk scale length. The absolute scale cancels:
optical depths and wavelengths fix everything that matters.

```yaml
label: compendium:exp:sech:Hernquist:dustD03Rv3.1
dust:
  file: hyperion-dust-0.1.0/dust_files/d03_3.1_6.0_A.hdf5
geometry:
  cutOff: 10.0
  components:
    disk:
      stellar: {profile: exponentialDisk, verticalStructure: sechSquared,
                scaleRadial: 1.0, scaleHeight: 0.137}
      dust:    {profile: exponentialDisk, verticalStructure: sechSquared,
                scaleRadial: 1.0, scaleHeight: 0.137,
                opticalDepth: {minimum: 0.01, maximum: 1.0e4, count: 60, includeZero: true}}
    spheroid:
      stellar: {profile: hernquist, truncation: 10.0,
                scaleRadial: {minimum: 1.0e-3, maximum: 1.0e2, count: 51}}
tabulation:
  wavelengths:  {minimum: 0.01, maximum: 3.0, count: 250}
  inclinations: {minimum: 0.0, maximum: 90.0, count: 46, spacing: linear}
  photons: 100000
  # Share photon histories between dusty models and their dust-free reference.
  seedPolicy: geometry
```

Any value written as a range, or as a list, becomes an axis of the tabulation.
Anything written as a number is held fixed.

`seedPolicy: model` (the default) gives every model a distinct seed, reproducing
the original driver. `seedPolicy: geometry` gives all optical-depth variants of
one emitter and geometry the same seed. The latter is recommended for new
campaigns because the attenuated and dust-free calculations then share photon
histories, reducing Monte Carlo noise in their ratio.

## Seeing what it would run, before running it

```console
$ dust-compendium validate configs/compendium-d03-rv3.1.yaml
  axes         :
    opticalDepth:disk                        61 values  from 0 to 10000
    scaleRadial:spheroid                     51 values  from 0.001 to 100
  emitters     :
    disk                                     61 models  over opticalDepth:disk
    spheroid                               3111 models  over opticalDepth:disk, scaleRadial:spheroid
  total        : 3172 models
```

This needs no Hyperion, and is the quickest way to find out that a campaign is
ten times the size you meant before committing a cluster to it.

Note that the disk is tabulated against optical depth alone. An axis which
cannot change a component's light — here, the size of a spheroid holding no dust
— is not tabulated against, since doing so would run the same model fifty-one
times.

## Building, running, collecting

```bash
dust-compendium build   config.yaml -o models
dust-compendium run     config.yaml -m models -o output --concurrency 8
dust-compendium collect config.yaml -o output -t attenuations.hdf5
```

`build` writes one Hyperion input per model. `run` solves them. `collect`
divides each by the model with no dust, fits the high optical depth
extrapolation, and writes the tabulation.

Each step skips work already done, so an interrupted campaign is resumed by
running the same command again. `run` checks that a solved model actually
contains the SEDs it was meant to produce rather than trusting the solver's exit
status — Hyperion aborts on some conditions and still exits zero.

## On a cluster

```bash
dust-compendium run config.yaml -m models -o output \
    --scheduler slurm --concurrency 40 \
    --nodes 4 --tasks 64 --partition compute --walltime 02:00:00
```

The Slurm backend writes a batch script per model, submits with `sbatch`, and
watches `squeue` and `sacct`. Only this campaign's jobs count towards
`--concurrency`, so a busy account cannot stall it, and a transient scheduler
failure is never mistaken for jobs having finished.

For a serial, many-model workflow use `--nodes 1 --tasks 1`; `--concurrency`
then controls how many independent Hyperion models are queued or running. See
the [HPC guide](hpc.md) for a staged smoke test and storage guidance.

## Looking at the result

```bash
dust-compendium plot attenuations.hdf5 -e disk -q extinction -a inclination \
    --colour-by opticalDepth -o extinction.pdf
dust-compendium plot attenuations.hdf5 -e spheroid -q curve -c spheroidScaleRadial
dust-compendium compare attenuations.hdf5 --atlas /path/to/atlasFerrara2000/....hdf5
```

`plot` draws any quantity — extinction at any wavelength, or the reddening
{math}`R_\mathrm{V}` — against any axis, optionally as a family of curves.
`compare` measures agreement against a published atlas rather than merely
drawing the two together.
