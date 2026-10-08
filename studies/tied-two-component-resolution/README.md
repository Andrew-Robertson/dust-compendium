# Quick spatial-grid convergence pilot

This checks for spatial discretization bias before the end-to-end emulator pilot.
It does not choose a final production parameter range, or test outer truncation,
alternative profile shapes, or all possible optical depths.

## Design

| Setting | Choice |
|---|---|
| Radial × vertical cells | 100×100, 200×200, 400×400 |
| First radial wall / smallest radial scale | 0.01, 0.005, 0.0025 respectively |
| Spatial scheme | nested spacing; cell-averaged dust density and integrated stellar cell luminosity |
| Dust optical depths | (disk, spheroid) = (0,0), (3,0), (0,3), (3,3) |
| Spheroid/disk scale ratio q | 0.1, 1, 10 |
| Emitters | disk and spheroid, separately |
| Wavelengths [μm] | 0.10, 0.2175, 0.55, 1.0, 2.0 |
| Inclinations | ten, 0–90 degrees |
| Azimuths | one, 90 degrees |
| Photon budgets | 40,000 imaging sources per wavelength; 40,000 raytracing sources |
| Independent seed families | three, matched across resolutions |

There are 24 models per configuration (18 dusty, six normalization controls),
nine configurations, and **216 solver runs**. These are diagnostic multi-wavelength
runs, not the future monochromatic training design. The higher photon budget
helps resolve grid shifts; it is not a recommendation for all training labels.

The smallest radial cell does not shrink merely by increasing `radialCells`.
We therefore refine `innerRadiusFraction` too, to include sensitivity to the
Hernquist cusp. Existing configurations retain the old 0.01 default. This is a
combined radial/vertical/core refinement test; it does not separate their errors.
All physical scale lengths, optical-depth normalizations and truncations stay
fixed. The analytic density profile is still represented by constant-density
cells, even though their masses are integrated accurately.

## Run on OBS-HPC

Source and small YAML configurations stay in home; large inputs/outputs go to
scratch. No existing campaigns are overwritten.

```bash
cd /home/arobertson/Galacticus/dust-compendium
git pull --ff-only
conda activate /home/arobertson/Galacticus/dust-compendium/.conda/hyperion
export DUST_RT_WORK=/carnegie/nobackup/users/arobertson/dust-rt-work

python studies/tied-two-component-resolution/study.py configs
```

Build and run, stopping if any command fails. Re-running skips solved models.
The concurrency cap applies to each 24-model configuration, so this conservative
loop runs at most 24 jobs simultaneously (not 96); configurations run sequentially.
Do not launch the whole loop again while it is already running.

```bash
(
  set -e
  for config in studies/tied-two-component-resolution/generated-configs/grid*_seed*.yaml; do
    name=$(basename "$config" .yaml)
    root="$DUST_RT_WORK/spatial-resolution/$name"
    dust-compendium build "$config" --output "$root/models"
    dust-compendium run "$config" \
      --models "$root/models" \
      --output "$root/output" \
      --scheduler slurm \
      --concurrency 96 \
      --tasks 1 \
      --nodes 1 \
      --partition obs \
      --walltime 01:00:00 \
      --memory-per-cpu 2048
  done
)
```

The one-hour job limit and 2 GiB request are starting allocations, not measured
400×400 timings/memory guarantees. If a job times out or runs out of memory,
retain its logs, increase that allocation and resume; do not interpret incomplete
runs as physical failures. This pilot does not require MPI.

After **all nine configurations** finish, export compact arrays:

```bash
python studies/tied-two-component-resolution/study.py export \
  --runs "$DUST_RT_WORK/spatial-resolution" \
  --output "$DUST_RT_WORK/spatial-resolution/compact"
```

Copy the three `grid100.npz`, `grid200.npz`, `grid400.npz` files to the Mac for
analysis (no large `.rtout` files required). Or, if matplotlib is installed, run
the report on HPC:

```bash
python studies/tied-two-component-resolution/study.py report \
  "$DUST_RT_WORK/spatial-resolution/compact/grid100.npz" \
  "$DUST_RT_WORK/spatial-resolution/compact/grid200.npz" \
  "$DUST_RT_WORK/spatial-resolution/compact/grid400.npz" \
  --output "$DUST_RT_WORK/spatial-resolution/report"
```

## What the analysis measures

For seed s, divide each dusty flux by its dust-free control with the same source
and geometry to form T(s). For coarse and fine grids, compare

    ΔT = mean_s(T_coarse − T_fine)
    ΔA = −2.5 log10[mean_s(T_coarse) / mean_s(T_fine)]

Positive ΔA means the coarse grid attenuates more. These are **ordinary magnitude
differences, not E_A**. The standard error of ΔT is the sample standard deviation
of the paired differences divided by sqrt(N_seeds). The delta-method SE of ΔA is
the sample standard deviation of

    −(2.5/ln 10) [T_coarse(s)/mean(T_coarse) − T_fine(s)/mean(T_fine)]

divided by sqrt(N_seeds). This retains covariance across grids. Matching random
seeds does not guarantee matched photon trajectories after the grid changes.
We use empirical seed scatter, not Hyperion's reported error estimate. With
three seeds, SEs themselves are uncertain: even under a Gaussian approximation,
a two-sided 95% Student-t interval is approximately ±4.3 SE, not ±2 SE.
Cells sharing photon histories must not be counted as independent replicates.

The report produces:

- `resolution_summary.png`: signed grid shifts and SEs for 100→200, 200→400
  and 100→400. The magnitude summary uses dusty cells with mean T>0.01 at all
  three resolutions; ±0.01-mag guides are context, not a universal pass/fail cut.
- `resolution_curves.png`: actual attenuation spectra at the six worst
  model/inclination combinations, ranked by the above 100→400 magnitude shift,
  with SEs of each mean. Only five wavelengths were calculated; connecting
  lines do not resolve the detailed UV bump.
- `summary.json`: magnitude shifts, SEs, absolute transmission shifts over
  **all dusty cells**, including T≤0.01, and recorded solver CPU totals.
- `difference_*.npz`: full per-cell shifts, standard errors and masks, allowing
  follow-up plots of dim cases or specific geometries without rerunning Hyperion.

400×400 is **not known truth**. Look for smaller 200→400 shifts, particularly
where transmitted light is appreciable. If shifts are within a few millimagnitudes
or unresolvable against seed scatter, there is little reason to delay the emulator
pilot over them. If coherent shifts are comparable to the emulator error budget,
adopt the finer grid or investigate only the failing cases (more photons/seeds or
an extra resolution); do not automatically launch another broad benchmark.

After this pilot, compare many noisier training labels with fewer more precise
labels at matched measured CPU budgets, against the same small low-noise validation
set. Grid bias and finite-photon noise must remain separate in that comparison.
