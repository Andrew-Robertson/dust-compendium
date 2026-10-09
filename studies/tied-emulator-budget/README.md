# Full tied-model emulator pilot: coverage versus photon precision

This is a small-budget version of the **full four-input problem**, not a slice
with one input fixed. It is not the final production domain or calibration.
No files in the earlier azimuth/resolution studies are changed.

## Frozen experiment

`pilot.json` contains the actual continuous locations and their provenance.
Unlike the Benson subsampling studies, these are not snapped to an existing grid.
The order, validation split, grains, seed policy and photon budgets are frozen.
The grain resource is checked against the original byte hash or the explicitly
audited equivalent OBS-HPC build in `grain_compatibility.json`. Their extinction,
albedo, frequency and scattering-angle arrays are identical; their scattering
matrices differ by at most 2.4e-15 fractionally. This is an exact allowlist, not
a general tolerance or a bypass. Unknown builds remain rejected. Each new worker
result records its actual grain hash as well as the frozen reference hash; older
results without the actual-hash field used the original reference (the only file
accepted before this audit). The frozen sampling manifest is unchanged.
The same wavelength/geometry locations are used for disk and spheroid emitters,
which are simulated and learned separately.

| Quantity | Pilot choice |
|---|---|
| Wavelength | 0.048–2.3 microns, supplied explicitly to the emulator |
| Disk optical depth | 0–30, central full face-on V-band optical depth |
| Spheroid optical depth | 0–30, full V-band ray at projected Hernquist radius a |
| q | 0.03–10, spheroid / stellar disk scale radius |
| Geometry | tied dust and stellar sizes, disk h/R=0.137, Hernquist truncation 10a |
| Spatial grid | 100×100, nested, cell-average density, inner radial fraction 0.01 |
| Outputs per RT solve | one wavelength, 19 inclinations (0–90 every 5 deg), one azimuth |
| Low-photon arm | up to 4,096 locations; 10,000 direct + 40,000 scattered photons |
| High-photon arm | first 512 of those locations; 80,000 direct + 320,000 scattered photons |
| Validation | 130 disjoint locations, three independent seeds each, 160,000 direct + 640,000 scattered photons |

Broad proposals use log wavelength, log(1+tau/0.01) for each depth, and log q.
Each block of 16 training locations contains seven broad points, six widened
population points, one disk-only point, one spheroid-only point, and one dust-free
point. Category-specific scrambled Sobol streams avoid conditioning a Sobol
coordinate on even/odd point indices. Both arms retain exactly the same mixture.
Only true dust-free locations get analytical T=1, with no unnecessary RT solve.
Disk-only and spheroid-only limits still require RT for both stellar sources.

The population uses the previously investigated GP-calibrated XMM-LSS catalog,
restricted to z<3, existing dust-attenuated F158<26, physically admissible gas,
metal and stellar masses, and both sizes positive. These requirements leave
10,405 rows; 10,286 lie in this pilot domain. The excluded fraction is recorded,
not clipped into the domain. This is an illustrative population, not a claim
about the future Galacticus calibration or the entire Roman sample.

For training, the joint (tau_d,tau_s,q) distribution is flattened with inverse
square-root occupancy weights on an 8×8×8 transformed-coordinate histogram.
It is widened by a common log dust-normalization shift uniform in +/-0.3 dex
and independent log disk/spheroid radius shifts uniform in +/-0.15 dex. These
change optical depths as mass/size² and q as the radius ratio. Out-of-domain
proposals are rejected, not clipped. Catalog row IDs and the input catalog SHA
are retained. All size-ratio columns are present even though fixed in this pilot.

Wavelengths are log-spaced proposals for all categories, **not** a grism-throughput
or redshift-weighted wavelength distribution. Future photometric/SED tests are
separate. The validation categories are 64 unflattened/unwidened population points,
32 broad points, 16 disk-only and 16 spheroid-only points. Two additional mixed-dust
q=10 points at 0.55 and 1 micron carry the `resolution_watch` label. They are still
100×100 calculations: they flag known grid sensitivity, not a higher-resolution
truth. Validation seeds are independent of all training seeds.

## Run ONLY the timing stage first

This first stage is 16 locations at both photon budgets and both emitters:
64 jobs, four of which are analytical dust-free controls. Each of the other
60 jobs executes a dusty model and its matching clear normalization: **120
serial Hyperion solves**. These outputs are reused by the training stage.
This checks actual monochromatic runtime, not the previous five-wavelength cost.
The two-hour walltime is a starting limit, not a runtime guarantee.

On OBS-HPC:

```bash
cd /home/arobertson/Galacticus/dust-compendium
git pull --ff-only
conda activate /home/arobertson/Galacticus/dust-compendium/.conda/hyperion
export DUST_RT_WORK=/carnegie/nobackup/users/arobertson/dust-rt-work

python studies/tied-emulator-budget/pilot.py run \
  --stage timing \
  --runs "$DUST_RT_WORK/emulator-budget-v1" \
  --execute
```

Defaults: Slurm partition `obs`, concurrency 96, one core and 2 GiB per job,
two-hour walltime. There is no MPI. Omitting `--execute` prints the job count
without submitting. Model construction happens inside the compute jobs, not
on the login node. Each job retains its logs, raw models and compact result in
scratch; source and the manifest stay in home. Do not launch duplicate drivers.
A per-job file lock and request fingerprints protect against accidental overlap
or reusing stale outputs. A solver failure never becomes a zero transmission.

Export after all jobs finish:

```bash
python studies/tied-emulator-budget/pilot.py export \
  --stage timing \
  --runs "$DUST_RT_WORK/emulator-budget-v1" \
  --output "$DUST_RT_WORK/emulator-budget-v1/compact/timing.npz"
```

Copy `timing.npz` back to the Mac. Inspect costs, failures, and the transmission
range before proceeding to the full pilot. The 16-point timing set is deliberately
small and cannot guarantee the absence of slower tail cases in the larger set.
If a resource limit is exceeded, retain logs and resume with more walltime.

## Subsequent stages (after the timing review)

Use `--stage training` and then `--stage validation` in the same run command.
The former skips completed timing jobs and runs the remaining members of both
arms; the latter runs independent validation seeds. Export each complete stage
as `training.npz` and `validation.npz`, respectively, using the same export syntax.
There are 9,216 total training jobs (including 576 analytical dust-free jobs),
and 780 validation jobs. Each non-analytic job includes its clear control.
Large raw outputs remain on scratch; transfer only the compact stage exports.

For a completed matched-budget comparison, the script sums the measured solver
CPU and worker Python CPU, including normalization, across both emitters. It then
chooses the low-photon **prefix in whole blocks of 16** closest to the total cost
of the 512 high-photon locations. Cost matching is independent of all validation
errors. It refuses to call this matched-budget if no prefix from 512 to 4,096 is
within 5%; in that case extend or revise the allocation explicitly. No photon-count
ratio is silently presented as a CPU ratio. Unused low-arm outputs still count
towards total experiment expenditure, but not the compared training-set budget.

The exporter also records wall times, because CPU excludes queue time and does
not fully represent billed/allocated time on every HPC. For resumed raw solves,
solver CPU is retained but earlier Python/I/O overhead cannot be recovered.

## End-to-end learning and validation

The first comparison deliberately learns **T(i)**, not log T. Independent
Monte Carlo fluctuations can average without the Jensen bias produced by
transforming noisy labels to magnitudes. This is not an assertion that T is the
optimal target: errors at small transmission may be harder to learn. The pilot
will expose that, rather than mixing several target representations into the
photon-budget comparison.

Inputs retain all four dimensions:

    log10(lambda), log10(1 + tau_d*K_lambda/0.01),
    log10(1 + tau_s*K_lambda/0.01), log10(q)

where K_lambda = kappa_lambda/kappa_V comes from the exact hashed grain file.
This incorporates the grain extinction feature without discarding wavelength.
This is the extinction-depth coordinate, not the transport-depth approximation.

Each arm fits inclination-weighted PCA of T(i), keeping 12 of 19 components, then
a 3×128 ReLU MLP with fixed settings and three initialization seeds. PCA and
normalizations use training data only. Early stopping reserves 15% of those
training locations, not the external validation set. The outputs include a
held-out PCA reconstruction oracle so compression error is not mistaken for
regression error. The same architecture is a first controlled comparison, not
separate hyperparameter optimization for every arm.

Three arms are reported: low photons at the same 512 locations, high photons at
those 512 locations, and low photons at the matched-budget larger prefix. Report
categories separately; the mixed validation sampling frequencies are not a
population probability distribution. Within a category, panels have equal weight
and inclinations use uniform-cos(i) quadrature.

The reference is the mean of three validation transmissions. Its uncertainty u
is the empirical sample standard deviation divided by sqrt(3), **not** the
single-run Hyperion uncertainty. We retain

    E_A = -0.01*(T_prediction - T_reference)
          / sqrt[u² + (ln(10)/2.5 * 0.01 * T_reference)²].

Also report ordinary delta_A where T_reference>0.01, absolute delta_T everywhere,
negative neural-network outputs and undefined cases. Predictions are not silently
clipped to [0,1]; directional transmission may exceed one. Three seeds give only
a noisy uncertainty estimate, and low-noise validation does not fix grid bias.

On the Mac, with scikit-learn, matplotlib, NumPy and dust-compendium available:

```bash
python studies/tied-emulator-budget/emulate.py \
  --training /path/to/training.npz \
  --validation /path/to/validation.npz \
  --output /path/to/emulator-budget-results
```

This writes per-arm/per-seed metrics, inclination-curve examples, complete held-out
predictions and portable numerical network weights. The NPZ weight files contain
no pickle: inference is an explicit NumPy forward pass, including PCA reconstruction.
They are pilot artifacts, not yet a public API or a calibrated uncertainty model.

## Reproducing the design locally

`design.py --catalog ... --grains ... --output pilot.json` rebuilds the frozen
manifest from the original catalog and D03 grain file. It refuses to overwrite a
different design. OBS-HPC needs neither the Galacticus catalog nor scikit-learn
to run/export the RT stage. A changed design must use a new output root.
