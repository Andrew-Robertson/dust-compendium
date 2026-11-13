# Extend the tied-model pilot to 32,768 low-photon locations

The new campaign appends 28,672 locations to the original 4,096, with one
disk-source and one spheroid-source task at each location: **57,344 new tasks**.
It uses the same four-dimensional domain, sampling mixture, grain model,
100-by-100 nested grid, cell-averaged densities, one azimuth, 19 inclinations,
and 10,000 direct / 40,000 scattered photon budget. The 512 high-photon
locations and independent three-seed validation reference are reused.

`extension-32768.json` is frozen, SHA-256
`2ffd0df576df9609e4263622ee5a674500f1755ae2325b98d2c8e49b22228c24`.
Its `many_count` is the **additional** count (28,672), not the final count.
Original point indices and RNG seed formulas continue from 4,096. Generation
reproduces the parent sampling prefix before accepting the continuation, and
checks every new location for overlap with training and validation. Parent
records and the parent manifest remain untouched.

At the pilot's measured mean cost, the extension costs approximately
**1,198 additional CPU-hours**, including both emitters. This is an estimate,
not a scheduler walltime promise. No validation or high-photon reruns are needed.

## Audit of the v1 compendium defects

[Galacticus PR #1683](https://github.com/galacticusorg/galacticus/pull/1683)
registers version-2 tabulations and describes three original geometry defects,
plus repaired high-depth extrapolation. Two datasets in our work must be
distinguished:

* The earlier Benson-table emulator experiments used the published **v1 data**.
  Their spheroid targets contain the upstream defects. Those experiments remain
  tests of reproducing that numerical table, not validation of physical
  spheroid attenuation. Keep them for reproducibility; use corrected data for
  a future atlas-based scientific reference.
* The recent tied disk+spheroid pilot computed **new RT solutions** using the
  refactored package, which already contains the geometry corrections.

| Upstream defect | Pilot implementation and evidence |
| --- | --- |
| Wrong Hernquist antiderivative sign at the negative-axis special case | `geometry/spheroid.py` uses `-u/[2(abs(u)+1)]`, giving +1/4 at u=-1. Present in ancestor `34caf44` (2026-09-15), before pilot commit `b681a52`. |
| Thin disk dust disappears when a large spheroid sets the grid extent | Pilot explicitly uses `spacing="nested"` and `sampling="average"`. These resolve the small vertical scale and conserve integrated cell mass. |
| Ignored spheroid truncation | Both stellar and dust spheroids explicitly have truncation 10; `cell_mass` integrates cells crossing the spherical boundary and zeroes cells outside. Implemented in ancestor `fb8bd25` (2026-09-15). |
| Rising/unset high-optical-depth extrapolation coefficients | Not used by this pilot: its targets are dusty/clear RT flux ratios at the actual requested coordinates. Neither the published coefficients nor `fit_extrapolation()` enter the worker or neural fits. |

The installed geometry, density and grid source files are byte-identical to
those at the pilot-introduction commit. The downloaded training records confirm
nested/average settings and truncation 10. All historical worker fingerprints
are `d4c15226f958c46fa8a44942e7093fb7d463c8196918e240107de893298f8169`.
The original exports did **not** fingerprint all imported package files, so
this is a code/branch-history and recorded-configuration audit, not an independent
verification of every archived HPC input or unrecorded local modification.

`audit_geometry.py` exercises the actual pilot configuration/build path at
q = 0.03, 0.137, 1, 3, 10 and 31.6. The last is an out-of-domain regression
control (the pilot stops at q=10). Results are in
`geometry-audit-20261113.json`:

* The spheroid mass agrees with the analytic mass inside 10a to about 6e-15
  relative; reflected cell-mass differences are at most 3e-14 of total mass.
* No individual cell holds more than 0.66% of the spheroid emission.
* The disk mass agrees with the integrated profile. The optical depth through
  the innermost finite-radius cylinder is 0.9934--0.9998 for a requested depth
  of one, including q=31.6; it does not disappear for large spheroids.
* Density and cell mass outside the spheroid truncation are zero.

157 existing geometry tests passed (one skipped). These checks address the
specific v1 defects; they do not replace spatial-grid convergence tests or
claim that our tied geometry and observing grid exactly reproduce the v2 atlas.
The evidence does not call for discarding or rerunning the current pilot.

## Provenance and storage for the extension

The manifest fingerprints every Python source file in the model-building
package. Submission refuses a mismatch. Each submission snapshots the package
and imports that snapshot via PYTHONPATH, so subsequent edits to the checkout
do not change queued jobs. Each result records the source fingerprints.

All tasks retain compact `result.npz` (transmission, raw flux vectors and
reported uncertainties), request/configuration, and solver logs. The first
16 new locations retain raw HDF5 input/output files as an audit sample. For
later locations, the four raw files are removed **only after** an atomic compact
result has been written and reread successfully. Failed tasks keep their raw
files for diagnosis. This retention policy applies only to the new extension;
the original pilot files are untouched. It avoids storing terabytes of
reproducible embedded grain/input copies.

## Run on OBS-HPC

Keep the source checkout in home space and use nobackup for run products.

```bash
cd /home/arobertson/Galacticus/dust-compendium
git pull --ff-only origin feature/tied-two-component-dust
conda activate /home/arobertson/Galacticus/dust-compendium/.conda/hyperion
export DUST_RT_WORK=/carnegie/nobackup/users/arobertson/dust-rt-work

python studies/tied-emulator-budget/audit_geometry.py

python studies/tied-emulator-budget/arrays.py submit \
  --manifest studies/tied-emulator-budget/extension-32768.json \
  --runs "$DUST_RT_WORK/emulator-budget-32768-extension" \
  --stages training \
  --concurrency 96
```

The dry run should report 57,344 remaining tasks in six arrays, followed by an
export job. With MaxArraySize=10001, chunks are 10001, 10001, 10001, 10001,
10001 and 7339. The arrays are chained, so 96 is a global concurrency cap for
this submission rather than 96 per chunk. If the checks pass, submit:

```bash
python studies/tied-emulator-budget/arrays.py submit \
  --manifest studies/tied-emulator-budget/extension-32768.json \
  --runs "$DUST_RT_WORK/emulator-budget-32768-extension" \
  --stages training \
  --concurrency 96 \
  --execute
```

The terminal can disconnect after submission returns. Rerunning the command
after all earlier jobs finish submits only unfinished work. Failed tasks do not
yield a partial export. The output is:

`$DUST_RT_WORK/emulator-budget-32768-extension/compact/training.npz`

## Combine completed results

After the extension finishes, merge it with the original compact archives:

```bash
python studies/tied-emulator-budget/extend.py merge \
  --manifest studies/tied-emulator-budget/extension-32768.json \
  --original "$DUST_RT_WORK/emulator-budget-v1/compact" \
  --extra "$DUST_RT_WORK/emulator-budget-32768-extension/compact/training.npz" \
  --output "$DUST_RT_WORK/emulator-budget-32768-combined/compact"
```

This verifies source archive provenance and task identity, preserves all
original execution metadata, and restores canonical low-then-high ordering.
Expect 66,560 training records (32,768 low + 512 high, times two emitters) and
the same 780 validation records. Each record's manifest hash remains its actual
execution manifest; the top-level hash describes `combined.json`. Learning
curve code should use this dataset-level manifest, not the original pilot's
hard-coded size/hash. The old exploratory training scripts are not automatically
generalized by this extension.

## Regenerating the design locally

Generation needs the original catalog, a verified equivalent grain file and
Hyperion's Python reader. These are not needed on OBS for the frozen manifest.

```bash
python studies/tied-emulator-budget/extend.py build \
  --catalog /path/to/original/romanUNIT-d1_25_16sqDeg.hdf5 \
  --grains /path/to/d03_4.0_4.0_A.hdf5 \
  --total 32768 \
  --output /path/to/check-extension.json
```

An existing different manifest is never overwritten. The catalog checksum,
old sampling prefix and opacity ratios must agree with the frozen parent.
