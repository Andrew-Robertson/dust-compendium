# Choosing the next run strategy

This is a numerical convergence study, not a production parameter sweep. The
physical distribution of the next Galacticus calibration remains unsettled.
Keep the existing 54-model smoke grid, five wavelengths and ten inclinations
fixed for these comparisons.

## What the existing summaries suggest

The OBS-HPC results reported in the conversation were:

| Azimuths | Median empirical sigma(A), mag | p95, mag | Seed-0 mean job elapsed time, s |
| --- | ---: | ---: | ---: |
| 1 | 0.003552 | 0.02297 | 40.2 |
| 2 | 0.003108 | 0.02103 | 75.5 |
| 4 | 0.002776 | 0.01812 | 141.5 |

Four azimuths cost about 3.52 times as much for 22% less median scatter. This
suggests spending more photons at one azimuth, but is NOT yet a matched-cell
efficiency measurement: the old aggregates include dust-free self-ratios, and
these are seed-0 elapsed timings, not per-model, repeated-seed solver CPU costs.
Do not infer a production photon count from these three rows.

## 1. Export the completed runs before submitting anything else

On OBS-HPC, in the repository and activated environment:

```bash
cd /home/arobertson/Galacticus/dust-compendium
conda activate /home/arobertson/Galacticus/dust-compendium/.conda/hyperion
export DUST_RT_WORK=/carnegie/nobackup/users/arobertson/dust-rt-work

python studies/tied-two-component-azimuths/efficiency.py export \
  --runs "$DUST_RT_WORK/azimuth-convergence" \
  --output "$DUST_RT_WORK/azimuth-convergence/compact"
```

This reads the existing outputs and logs; it does not submit jobs or change
simulation files. It writes three compressed NumPy files, `az1.npz`, `az2.npz`
and `az4.npz`, containing every seed and azimuth separately, reported flux
uncertainties, model parameters, configuration hashes and solver CPU timings.
The exporter requires every configured output to be solved; missing seeds are
not silently treated as zeros. Missing timing logs become NaN, not zero cost.
The compact exports can be downloaded with VSCode and retained in backed-up
storage; no need to transfer the large Hyperion inputs/outputs.

To make the report on HPC (requires matplotlib), or locally with the three
downloaded files:

```bash
python studies/tied-two-component-azimuths/efficiency.py report \
  "$DUST_RT_WORK/azimuth-convergence/compact/az1.npz" \
  "$DUST_RT_WORK/azimuth-convergence/compact/az2.npz" \
  "$DUST_RT_WORK/azimuth-convergence/compact/az4.npz" \
  --output "$DUST_RT_WORK/azimuth-convergence/efficiency-report"
```

The report writes two self-labelled PNG figures and `summary.json`:

- between-seed precision, matched-cell cost/variance ratios, and the effect of
  azimuth covariance;
- precision by emitter, geometry, and wavelength, showing the worst inclination
  for each case, rather than hiding difficult cases in one median;
- separate disk/spheroid summary statistics and explicit metric definitions.

## Metrics and limitations

For each seed, divide the azimuth-averaged dusty flux by the corresponding
azimuth-averaged dust-free flux. Across seeds, compute the sample standard
deviation `s_T` with `ddof=1`, and its small-error magnitude equivalent
`sigma_A = (2.5 / ln(10)) s_T / mean(T)`. This is a precision measurement, NOT an
error relative to known truth. Report sigma(T) as well; near zero transmission,
the magnitude equivalent is not an accurate description of a Gaussian A error.

For each physical model and output cell compare
`C = mean(solver CPU seconds) * s_T**2`. Lower C means cheaper precision if
variance scales inversely with photon count. Ratios use exactly the same cells
and common seed identities. Dust-free self-ratios are excluded from headline
metrics. There is no galaxy-population weighting in this diagnostic.

The azimuth covariance diagnostic compares measured variance of the azimuthal
mean to `sum(view variances) / N_azimuth**2`. It is 1 for independent views and
N_azimuth for identical views. These view transmissions share an averaged
reference denominator, intentionally retaining that covariance.

Five seeds only give four variance degrees of freedom per cell. Wavelengths,
viewing angles and some physical models share random histories; thousands of
output cells are NOT thousands of independent noise realizations. The plot's
5-95% bars describe variation ACROSS CELLS, not confidence intervals. Repeat
more seeds for a small finalist set if rankings are close. Never rescale all
Hyperion uncertainties by the previously reported median empirical/reported
ratio: our ratio includes a dust-free denominator and conservative azimuth
propagation, not just the uncertainty of a raw Hyperion flux.

## 2. Prepared photon-allocation follow-up

`tabulation.photons` remains backwards compatible. Optional `imagingPhotons`
and `raytracingPhotons` override it independently. Imaging trajectories are
per wavelength; direct-light raytracing samples are shared across wavelengths.

The next three variants all use ONE azimuth:

| Variant | Imaging per wavelength | Direct samples | Purpose |
| --- | ---: | ---: | --- |
| Existing az1 | 10,000 | 10,000 | Reuse baseline |
| az1_i40000_r10000 | 40,000 | 10,000 | Is scattered-light noise limiting? |
| az1_i10000_r40000 | 10,000 | 40,000 | Is direct-light noise limiting? |
| az1_i40000_r40000 | 40,000 | 40,000 | Check approximate inverse-N variance scaling |

Three variants times five seeds times 54 models = 810 NEW serial jobs. The new
campaigns have different labels and output roots: existing inputs/results are
not overwritten. They retain the original geometries and seed families. This
does not guarantee identical random trajectories when photon counts change.

Generate configurations, then run seed 0 only as a timing/storage pilot:

```bash
python studies/tied-two-component-azimuths/make_efficiency_configs.py

for name in az1_i40000_r10000_seed0 az1_i10000_r40000_seed0 az1_i40000_r40000_seed0; do
  config="studies/tied-two-component-azimuths/generated-configs/efficiency/${name}.yaml"
  root="$DUST_RT_WORK/photon-efficiency/$name"
  dust-compendium build "$config" --output "$root/models" || break
  dust-compendium run "$config" \
    --models "$root/models" --output "$root/output" \
    --scheduler slurm --concurrency 96 --tasks 1 --nodes 1 \
    --partition obs --walltime 01:00:00 --memory-per-cpu 2048 || break
done
```

The one-hour request is a cautious pilot allowance, not a measured requirement.
One campaign contains 54 jobs, so concurrency 96 is an upper bound here.
Check free scratch space and pilot runtimes before repeating for seeds 1-4.
No MPI needed. The existing result suggests an upper-order budget of about
36 core-hours for these 810 jobs if all new variants cost as much as
four times the original az1 mean; this is a rough extrapolation, NOT a bound.

After the pilot is satisfactory, this resumes seed0 and runs the remaining
seeds (the runner skips solved models):

```bash
for config in studies/tied-two-component-azimuths/generated-configs/efficiency/*_seed*.yaml; do
  name=$(basename "$config" .yaml)
  root="$DUST_RT_WORK/photon-efficiency/$name"
  dust-compendium build "$config" --output "$root/models" || break
  dust-compendium run "$config" \
    --models "$root/models" --output "$root/output" \
    --scheduler slurm --concurrency 96 --tasks 1 --nodes 1 \
    --partition obs --walltime 01:00:00 --memory-per-cpu 2048 || break
done

python studies/tied-two-component-azimuths/efficiency.py export \
  --configs studies/tied-two-component-azimuths/generated-configs/efficiency \
  --runs "$DUST_RT_WORK/photon-efficiency" \
  --output "$DUST_RT_WORK/photon-efficiency/compact"

python studies/tied-two-component-azimuths/efficiency.py report \
  "$DUST_RT_WORK/azimuth-convergence/compact/"*.npz \
  "$DUST_RT_WORK/photon-efficiency/compact/"*.npz \
  --output "$DUST_RT_WORK/photon-efficiency/efficiency-report"
```

At fixed geometry we can then test the approximation
`Var(T) = a/N_imaging + b/N_direct`. Comparing the single-budget increases
identifies which term dominates; the joint increase checks scaling. Choose
allocation based on measured CPU cost too, not photon counts alone. Wavelength
count changes that balance because only imaging repeats the trajectories for
each wavelength. Do not extrapolate five-wavelength efficiency blindly to a
single-wavelength training campaign.

## 3. Before choosing production settings

- Repeat finalists with one wavelength per run, as intended for a Sobol training
  design, and include UV, bump, optical and near-IR cases. Do this after the
  cheap five-wavelength comparison identifies plausible allocations.
- Test the 100x100 spatial grid against finer grids at representative compact,
  mixed-dust and high-opacity cases with enough photons to resolve systematic
  shifts. Monte Carlo seed scatter cannot establish spatial accuracy.
- Recheck the final optical-depth/size domain after the Galacticus recalibration.
  Do not claim convergence in high-tau corners absent from this smoke grid.

No production parameter-space campaign is being submitted by these tools.
