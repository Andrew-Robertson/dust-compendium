# Inner spheroid dust and a central point source

This study asks whether a core/cavity small enough to contain almost no stellar
mass changes attenuation of extended sources, and determines what resolving the
same dust profile requires for a central point source. It is separate from the
production campaign: no existing RT results or model-building defaults change.

## Physics and normalization

Disk stars and dust retain the exponential/sech-squared geometry with h/R=0.137.
Spheroid stars retain a Hernquist profile truncated at 10a. For spheroid dust only,
compare the original cusp with:

- a cavity: density zero for r < epsilon*a, unchanged shape outside;
- a core: density rho_H(epsilon*a) inside, unchanged shape outside;
- epsilon = 0.003 and 0.01 for each.

Each dust profile is renormalized to the SAME total spheroid dust mass inside
10a. The input tau_s labels the original Hernquist mass normalization, defined
on a full sightline at impact parameter a. The actual modified tau(b=a) is that
value times `mass_factor`, recorded in `grid-audit.json`. The largest correction
is only 0.012%. Stars and disk dust are not modified. All three emitters see the
same dust geometry. These are host-spheroid profile experiments, NOT luminosity-
dependent sublimation cavities (which would remove any dust species inside them,
including disk dust).

An untruncated Hernquist profile has M(<epsilon*a)/M = [epsilon/(1+epsilon)]^2:
8.95e-6 or 9.80e-5 at the two radii. The fraction of the truncated stellar light
is larger by (11/10)^2. This motivates expecting small integrated effects, but
is not a bound: photons emitted outside the cavity can traverse it.

The central emitter is an isotropic point at (0,0,0), not a small map source.
All calculations kill photons on absorption; no re-emission or torus model is
included. Total and unscattered direct transmissions are saved separately.
The direct point-source result is checked against exp(-kappa_lambda*Sigma_grid)
on every worker. The original central cusp has NO finite physical column; its
finite grid result is retained only to diagnose discretization dependence.

## Grids and design

The actual pilot production grid is 100x100, nested, with first radial wall
0.01 times the smallest radial scale. At q<=0.1 its first positive vertical
wall is 0.4a. The earlier 400x400 grid reduced that to 0.1a, still much larger
than these cores. Increasing radial resolution alone is insufficient.

All profiles at a given q use the SAME walls. Three grids:

1. `legacy`: exactly the production 100x100 grid.
2. `inner8`: retain every legacy wall, add inner R and z walls reaching
   0.003a/8, logarithmically spaced with adjacent ratio <=1+1/8, and explicitly
   include 0.003a and 0.01a.
3. `inner16`: repeat with 0.003a/16 and ratio <=1+1/16.

Thus outer cells/extent remain fixed. These test INNER refinement, not a new
full-grid convergence campaign. A deterministic ray integral through each grid
is compared with the analytic spherical column before any RT is submitted.

| Case | q | tau_disk | tau_spheroid |
|---|---:|---:|---:|
| compact | 0.01 | 1 | 1 |
| small | 0.1 | 1 | 0.1 |
| matched | 1 | 1 | 1 |
| extended | 10 | 1 | 1 |
| sphere (isotropy control) | 1 | 0 | 0.1 |
| disk_control | 1 | 1 | 0 |

Five physical profiles x three grids x three independent seeds x three emitters
for the first five cases; the last needs only one profile. Total **702 tasks**.
Each task runs its dusty model and a matching dust-free normalization (1,404
solver executions, including inexpensive normalization controls). Five
wavelengths: 0.10, 0.2175, 0.55, 1, 2 microns; inclinations 0,15,...,90 degrees;
one azimuth. 40,000 imaging photons per wavelength and 40,000 direct-ray samples
per run. Seeds match across profile/grid comparisons, but trajectories need not
match after changing the geometry. Only independent seed replicates estimate
uncertainty; inclinations/wavelengths are not independent noise samples.

## OBS-HPC: prepare, then a 12-task pilot

Code stays in home; generated files and frozen provenance snapshots go to scratch.
Preparation freezes Python source, the audited grain resource and the design.
Subsequent git pulls cannot change an already prepared campaign.

```bash
cd /home/arobertson/Galacticus/dust-compendium
git pull --ff-only
conda activate /home/arobertson/Galacticus/dust-compendium/.conda/hyperion
export DUST_RT_WORK=/carnegie/nobackup/users/arobertson/dust-rt-work
export DUST_CORE_RUNS="$DUST_RT_WORK/central-source-v1"

python studies/tied-central-source/study.py prepare \
  --runs "$DUST_CORE_RUNS" \
  --grains hyperion-dust-0.1.0/dust_files/d03_4.0_4.0_A.hdf5

python studies/tied-central-source/study.py submit \
  --runs "$DUST_CORE_RUNS" --pilot --concurrency 12
```

The pilot has nine central-source sphere runs (cusp, 0.003a core/cavity, all
three grids), two extended-source sphere runs, and one disk-only central control.
It is for correctness and timing, NOT a seed-scatter conclusion. The automatic
export writes `compact/pilot.npz`, containing only completed valid cases; check
that it reports 12. Do not repeatedly prepare: use status and resume submission.

The jobs are a real Slurm array, **one CPU per task**, with a dependent export
job. Safe to disconnect. The initial per-task allocation is 4 GiB and two hours;
these are conservative requests, not measured full-study timings. Logs and job
IDs are under `submissions/submit-*/`; preparation itself submits nothing.

```bash
python studies/tied-central-source/study.py status --runs "$DUST_CORE_RUNS"
```

After inspecting the pilot, submit the remaining study tasks with:

```bash
python studies/tied-central-source/study.py submit \
  --runs "$DUST_CORE_RUNS" --concurrency 96
```

Completed results are validated and skipped. The script refuses duplicate live
submissions. If jobs fail/timeout, inspect their logs, then rerun submit (optionally
`--walltime 04:00:00`). An interrupted/ambiguous submission receipt is deliberately
blocked for inspection so a retry cannot silently duplicate jobs. The full export
fails if any result is missing; it never publishes a partial `results.npz`.

Each worker stores fluxes, reported uncertainties, direct/total transmission,
CPU timings and request provenance, validates its compact result, then removes
only its own regenerable `.hdf5`/`.rtout` intermediates. Logs remain. For local
investigation, `worker --index N --keep-raw` retains those files.

## Download and analyse

Download ONLY `central-source-v1/compact/pilot.npz` initially, and subsequently
`central-source-v1/compact/results.npz`. Both contain the design and grid audit;
no raw RT files or whole run directory are needed. Export can be repeated:

```bash
python studies/tied-central-source/study.py export --runs "$DUST_CORE_RUNS"
```

On the Mac, using galacticus-workspace (NumPy and matplotlib; no Hyperion needed):

```bash
python studies/tied-central-source/report.py /path/to/results.npz \
  --output /path/to/central-source-report
```

Outputs: deterministic central-column errors; attenuation curves for all three
emitters at i=45 degrees for each geometry; profile-induced differences on the
SAME fine grid; and legacy/inner8-to-inner16 differences for a FIXED physical
profile. `differences.npz` retains signed changes at every wavelength/inclination.
`summary.json` records full-cell absolute Delta T and magnitude summaries where
both mean transmissions exceed 1e-4 (the cutoff is for interpretable plots, not
an accuracy tolerance or an exclusion from Delta T).

Delta A = -2.5 log10(mean(T_new)/mean(T_reference)); positive means more attenuation.
Uncertainty is the empirical paired-seed SE, including covariance. With three
seeds, a Gaussian two-sided 95% paired-t interval is about +/-4.3 SE, not +/-2 SE.
We do not subtract Hyperion's reported uncertainty or regard inner16 as truth.

We can retain existing extended-source labels only if physical-profile shifts
AND residual grid shifts are negligible for the intended accuracy. The central
source needs a specified finite core/cavity and demonstrated grid convergence
regardless of whether integrated stellar attenuation changes detectably.
