# Tied two-component azimuth convergence

This study measures whether asking Hyperion for more peel-off azimuths is a
useful way to reduce Monte Carlo noise. Axisymmetry makes their expectations
equal, but does not make the estimators independent: all directions use the same
photon histories, and every additional direction has a computational cost.

For the completed-data efficiency diagnostics and the next independently varied
imaging/direct-light photon budgets, see [EFFICIENCY.md](EFFICIENCY.md).

The experiment holds the physical 54-model smoke grid and photon count fixed,
uses one, two and four explicit azimuths, and repeats each choice with five
independent seed families. It therefore comprises 15 campaigns or 810 model
runs. Start with one seed of each choice (162 runs) if a timing pilot is wanted
before committing to the full set.

Generate the configurations on OBS-HPC from the repository root:

```bash
python studies/tied-two-component-azimuths/make_configs.py
```

Use this directory layout, where each generated configuration has its own model
and output directory:

```text
$DUST_RT_WORK/azimuth-convergence/
  az1_seed0/{models,output}/
  az2_seed0/{models,output}/
  ...
```

For a three-campaign timing pilot:

```bash
for name in az1_seed0 az2_seed0 az4_seed0; do
  config="studies/tied-two-component-azimuths/generated-configs/${name}.yaml"
  root="$DUST_RT_WORK/azimuth-convergence/$name"
  dust-compendium build "$config" --output "$root/models"
  dust-compendium run "$config" \
    --models "$root/models" \
    --output "$root/output" \
    --scheduler slurm \
    --concurrency 24 \
    --tasks 1 \
    --nodes 1 \
    --partition obs \
    --walltime 00:20:00 \
    --memory-per-cpu 2048
done
```

The jobs are independent, so one task per model is intentional. The suggested
wall time is a safe pilot request, not a measured production requirement.
Record elapsed CPU time from Slurm for the three campaigns: equal photon counts
do not imply equal cost when the number of peel-off directions changes.

After all five seeds of a variant have completed, summarize its actual
between-seed scatter:

```bash
python studies/tied-two-component-azimuths/analyze.py \
  --runs "$DUST_RT_WORK/azimuth-convergence"
```

The resulting CSV reports scatter in transmission and its small-error
magnitude equivalent, plus the ratio of empirical between-seed scatter to the
conservative uncertainty propagated from Hyperion. This is deliberately a
numerical convergence diagnostic; it does not yet choose a production photon
count.
