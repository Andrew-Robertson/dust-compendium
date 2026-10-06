#!/usr/bin/env python3
"""Export small, per-seed arrays on HPC; compare empirical precision per solver CPU-second locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from dustcompendium.campaign import Campaign
from dustcompendium.config import load_campaign
from dustcompendium.postprocess import read_sed_views
from dustcompendium.runner import is_solved

FACTOR = 2.5 / np.log(10.0)
CPU = re.compile(r"Total CPU time elapsed:\s*([\d.Ee+\-]+)")


def cpu_seconds(path):
    """Serial Hyperion solver CPU time, not Slurm queue time or allocation time."""
    if not path.exists():
        return np.nan
    matches = CPU.findall(path.read_text(errors="replace"))
    return float(matches[-1]) if matches else np.nan


def model_records(campaign):
    runs = list(campaign.runs())
    lookup = {(run.emitter, run.indices): n for n, run in enumerate(runs)}
    records, references = [], []
    for run in runs:
        indices = list(run.indices)
        params = {}
        for p, (axis, index) in enumerate(zip(campaign.axes_for(run.emitter), run.indices, strict=True)):
            params[axis.name] = float(axis.values[index])
            if axis.kind == "opticalDepth":
                zeros = np.flatnonzero(axis.values == 0.0)
                if zeros.size != 1:
                    raise ValueError(f"{axis.name}: need exactly one dust-free reference")
                indices[p] = int(zeros[0])
        references.append(lookup[run.emitter, tuple(indices)])
        records.append({"emitter": run.emitter, "parameters": params})
    return runs, records, references


def export(configs, runs_root, output):
    groups = defaultdict(list)
    for path in sorted(configs.glob("*_seed*.yaml")):
        match = re.fullmatch(r"(.+)_seed(\d+)\.yaml", path.name)
        if match:
            groups[match[1]].append((int(match[2]), path))
    if not groups:
        raise ValueError(f"no seed configurations in {configs}")
    output.mkdir(parents=True, exist_ok=True)
    for variant, entries in groups.items():
        entries.sort()
        arrays, errors, times, fingerprints = [], [], [], []
        metadata = None
        for seed, config in entries:
            campaign = Campaign(load_campaign(str(config)))
            runs, records, references = model_records(campaign)
            common = {
                "geometry": campaign.config.geometry.model_dump(mode="json"),
                "dust": campaign.config.dust.model_dump(mode="json"),
                "models": records,
                "references": references,
                "inclinations": campaign.inclinations.tolist(),
                "wavelengths": campaign.wavelengths.tolist(),
                "azimuths": campaign.azimuths.tolist(),
                "imaging_photons": campaign.config.tabulation.imaging_photons
                or campaign.config.tabulation.photons,
                "raytracing_photons": campaign.config.tabulation.raytracing_photons
                or campaign.config.tabulation.photons,
                "seed_policy": campaign.config.tabulation.seed_policy,
            }
            if metadata is not None and metadata != common:
                raise ValueError(f"different physical/numerical settings within {variant}")
            metadata = common
            values, uncertainty, elapsed = [], [], []
            for run in runs:
                path = runs_root / config.stem / "output" / f"{run.file_stem}.rtout"
                if not is_solved(path):
                    raise ValueError(f"missing/unsolved output: {path}; no partial-seed comparison")
                wav, flux, err = read_sed_views(path, azimuth_count=campaign.azimuths.size)
                if not np.allclose(wav, campaign.wavelengths, rtol=1e-6, atol=0):
                    raise ValueError(f"unexpected wavelength order: {path}")
                if flux.shape != (len(campaign.azimuths), len(campaign.inclinations), len(wav)):
                    raise ValueError(f"unexpected viewing grid: {path}")
                if not np.isfinite(flux).all() or np.any(flux < 0):
                    raise ValueError(f"invalid flux: {path}")
                values.append(flux)
                uncertainty.append(err)
                elapsed.append(cpu_seconds(path.parent / "logs" / f"{run.file_stem}.log"))
            arrays.append(values)
            errors.append(uncertainty)
            times.append(elapsed)
            fingerprints.append(
                {
                    "seed_id": seed,
                    "config": str(config.resolve()),
                    "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                    "model_seeds": [run.spec.seed for run in runs],
                }
            )
            print(f"{config.stem}: exported {len(runs)} solved models")
        metadata.update(
            format_version=1,
            variant=variant,
            provenance=fingerprints,
            axes="seed, model, azimuth, inclination, wavelength",
            timing="serial Hyperion solver CPU seconds; excludes I/O wrapper and queue time",
        )
        destination = output / f"{variant}.npz"
        np.savez_compressed(
            destination,
            metadata=json.dumps(metadata),
            flux=np.asarray(arrays),
            reported=np.asarray(errors),
            cpu_seconds=np.asarray(times),
            seeds=[x[0] for x in entries],
        )
        print(f"wrote {destination} ({destination.stat().st_size / 1024:.1f} KiB)")


def transmission(flux, reference_indices):
    """Ratio of azimuth means, not mean of view-by-view ratios."""
    averaged = np.mean(flux, axis=2)
    normal = averaged[:, reference_indices]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(normal > 0, averaged / normal, np.nan)


def statistics(flux, reported, references):
    t = transmission(flux, references)
    mean = t.mean(axis=0)
    sigma = t.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_a = np.where(mean > 0, FACTOR * sigma / mean, np.nan)
        values = flux.mean(axis=2)
        err = np.abs(reported).mean(axis=2)
        # A diagnostic only: ignores numerator/denominator covariance and uses
        # maximally positive azimuth covariance, as the original summary did.
        reported_t = np.abs(t) * np.sqrt(
            (err / values) ** 2 + (err[:, references] / values[:, references]) ** 2
        )
        ratio = sigma / np.median(reported_t, axis=0)
        # Relative variance of the azimuth mean, compared with independent views.
        # Use a common, azimuth-averaged denominator per seed for this diagnostic.
        normal = values[:, references, None, :, :]
        view_t = flux / normal
        view_variance = view_t.var(axis=0, ddof=1)
        gain = t.var(axis=0, ddof=1) / (view_variance.sum(axis=1) / flux.shape[2] ** 2)
    return {
        "t": t,
        "mean": mean,
        "sigma_t": sigma,
        "sigma_a": sigma_a,
        "empirical_reported": ratio,
        "azimuth_covariance_factor": gain,
    }


def finite_summary(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"cells": 0, "median": None, "p95": None, "rms": None}
    return {
        "cells": int(values.size),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "rms": float(np.sqrt(np.mean(values**2))),
    }


def report(paths, output, baseline):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = {}
    physical = None
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            data = {name: archive[name] for name in archive.files}
        meta = json.loads(str(data.pop("metadata")))
        comparison = {
            k: meta[k] for k in ("geometry", "dust", "models", "references", "inclinations", "wavelengths")
        }
        if physical is not None and physical != comparison:
            raise ValueError(f"unmatched geometries, grains or output grids in {path}")
        physical = comparison
        if meta["variant"] in datasets:
            raise ValueError(f"duplicate variant {meta['variant']}")
        datasets[meta["variant"]] = (meta, data)
    if baseline not in datasets:
        raise ValueError(f"baseline {baseline!r} not found")
    # Same replicate identities for every design, never compare different seed subsets.
    common_seeds = sorted(set.intersection(*(set(d["seeds"].tolist()) for _, d in datasets.values())))
    if len(common_seeds) < 3:
        raise ValueError("need at least three common seeds (five recommended)")
    stats = {}
    timings = {}
    for name, (meta, data) in datasets.items():
        index = [data["seeds"].tolist().index(seed) for seed in common_seeds]
        stats[name] = statistics(data["flux"][index], data["reported"][index], meta["references"])
        timings[name] = data["cpu_seconds"][index].mean(axis=0)
    meta = datasets[baseline][0]
    dusty = np.arange(len(meta["models"])) != np.asarray(meta["references"])
    mask = np.broadcast_to(dusty[:, None, None], stats[baseline]["mean"].shape).copy()
    for s in stats.values():
        mask &= np.isfinite(s["sigma_a"]) & (s["mean"] > 0)
    baseline_mean = stats[baseline]["mean"]
    base_efficiency = stats[baseline]["sigma_t"] ** 2 * timings[baseline][:, None, None]
    results = {
        "common_seed_ids": common_seeds,
        "baseline": baseline,
        "input_archives": [
            {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths
        ],
        "definitions": {
            "sigma_A": (
                "(2.5/ln(10))*sample_std_seed(T)/mean_seed(T), ddof=1; "
                "small-error magnitude equivalent, not bias"
            ),
            "efficiency": "sample_var_seed(T)*mean_solver_cpu_seconds per model; lower is better",
            "headline_mask": (
                "non-dust-free cells with finite sigma_A and positive mean T in every design; "
                "equal cell weights"
            ),
            "covariance_factor": (
                "variance(azimuth mean) / [sum(variance(each view))/N_azimuth^2]; 1 if uncorrelated"
            ),
            "uncertainty_warning": (
                "five seeds give noisy per-cell variances; "
                "cells share histories and are not independent replicates"
            ),
            "reference_warning": (
                "dust-free T=1 self-ratios excluded; empirical/reported is for a propagated ratio, "
                "not raw Hyperion flux"
            ),
            "timing_warning": (
                "missing logs give null timing metrics, never zero; do not compare across different hardware"
            ),
        },
        "variants": {},
    }
    names = list(datasets)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    for x, name in enumerate(names):
        s = stats[name]
        cost = timings[name]
        with np.errstate(divide="ignore", invalid="ignore"):
            efficiency_ratio = s["sigma_t"] ** 2 * cost[:, None, None] / base_efficiency
            # Same denominator for every variant: do not let noise in estimated
            # mean T change the cost-normalized aggregate ranking.
            e = FACTOR**2 * s["sigma_t"] ** 2 * cost[:, None, None] / baseline_mean**2
            difference = s["t"] - stats[baseline]["t"]
            mean_shift = difference.mean(axis=0)
            shift_sem = difference.std(axis=0, ddof=1) / np.sqrt(len(common_seeds))
            paired_t = np.where(shift_sem > 0, mean_shift / shift_sem, np.nan)
        row = {
            "sigma_A_mag": finite_summary(s["sigma_a"][mask]),
            "sigma_T": finite_summary(s["sigma_t"][mask]),
            "empirical_over_propagated_reported": finite_summary(s["empirical_reported"][mask]),
            "cost_variance_ratio_to_baseline": finite_summary(efficiency_ratio[mask & (base_efficiency > 0)]),
            "cost_variance_mag2_seconds": finite_summary(e[mask]),
            "solver_seconds_per_dusty_model": finite_summary(cost[dusty]),
            "missing_model_timings": int(np.count_nonzero(~np.isfinite(cost))),
            "azimuth_covariance_factor": finite_summary(s["azimuth_covariance_factor"][mask]),
            "absolute_mean_T_shift_from_baseline": finite_summary(np.abs(mean_shift[mask])),
            "absolute_paired_t_for_mean_shift": finite_summary(np.abs(paired_t[mask])),
            "by_emitter": {},
        }
        for emitter in sorted({m["emitter"] for m in meta["models"]}):
            selection = np.array([m["emitter"] == emitter for m in meta["models"]])[:, None, None] & mask
            row["by_emitter"][emitter] = finite_summary(s["sigma_a"][selection])
        results["variants"][name] = row
        for ax, array in zip(
            axes,
            (
                s["sigma_a"][mask],
                efficiency_ratio[mask & (base_efficiency > 0)],
                s["azimuth_covariance_factor"][mask],
            ),
            strict=True,
        ):
            finite = array[np.isfinite(array) & (array > 0)]
            if finite.size:
                low, median, high = np.quantile(finite, [0.05, 0.5, 0.95])
                ax.errorbar(x, median, yerr=[[median - low], [high - median]], fmt="o", capsize=4)
    for ax, title, ylabel in zip(
        axes,
        ("Between-seed precision", "Cost to reach equal variance", "Shared-history azimuth correlation"),
        (
            r"$\sigma_A=(2.5/\ln10)\,s_T/\bar T$ [mag]",
            r"$t\,s_T^2/(t\,s_T^2)_{\rm baseline}$",
            "Variance / independent-view prediction",
        ),
        strict=True,
    ):
        ax.set(title=title, ylabel=ylabel, yscale="log", xticks=range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_xlim(-0.5, len(names) - 0.5)
        ax.grid(alpha=0.2)
    axes[0].axhline(0.01, color="gray", linestyle="--", label="0.01 mag target")
    axes[0].legend(fontsize=8)
    for ax in axes[1:]:
        ax.axhline(1, color="gray", linestyle="--")
    fig.suptitle(
        f"{len(common_seeds)} matched seeds; dust-free controls excluded; "
        "dots=cell medians, bars=5-95% across cells\n"
        "Bars are NOT confidence intervals. Solver CPU cost only; fixed spatial grid.",
        fontsize=11,
    )
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "efficiency_summary.png", dpi=180)
    plt.close(fig)

    # Maps retain wavelength/inclination context and separate emitters/geometries.
    fig, axes = plt.subplots(
        len(names), 2, figsize=(12, 4.5 * len(names)), squeeze=False, layout="constrained"
    )
    im = None
    for row, name in enumerate(names):
        s = stats[name]
        for col, emitter in enumerate(("disk", "spheroid")):
            models = [j for j, m in enumerate(meta["models"]) if dusty[j] and m["emitter"] == emitter]
            if not models:
                continue
            worst_incl = np.max(s["sigma_a"][models], axis=1)
            im = axes[row, col].imshow(worst_incl, aspect="auto", vmin=0, vmax=0.03, origin="lower")
            axes[row, col].set(
                title=f"{name}: {emitter}",
                xlabel="Wavelength [micron]",
                ylabel=r"$(\tau_{V,d},\tau_{V,s},q)$",
                xticks=range(len(meta["wavelengths"])),
            )
            axes[row, col].set_xticklabels([f"{w:g}" for w in meta["wavelengths"]])
            labels = []
            for j in models:
                params = meta["models"][j]["parameters"]
                keys = ("opticalDepth:disk", "opticalDepth:spheroid", "scaleRadial:spheroid")
                labels.append(
                    ", ".join(f"{params[k]:g}" for k in keys)
                    if all(k in params for k in keys)
                    else f"model {j}"
                )
            axes[row, col].set_yticks(range(len(models)), labels, fontsize=7)
            results["variants"][name].setdefault("map_model_indices", {})[emitter] = models
    fig.suptitle(
        "Empirical seed scatter, not deviation from a known truth; fixed spatial grid\n"
        r"$\sigma_A=(2.5/\ln10)\,s_T/\bar T$; $q=a_{\rm spheroid}/R_{\rm disk}$; colors saturate at 0.03 mag",
        fontsize=11,
    )
    if im is not None:
        fig.colorbar(
            im, ax=axes.ravel().tolist(), label=r"$\max_i\,\sigma_A$ [mag]", shrink=0.8, extend="max"
        )
    fig.savefig(output / "precision_by_geometry.png", dpi=160)
    plt.close(fig)
    results["models"] = meta["models"]
    results["definitions"]["paired_t"] = (
        "mean of per-seed T differences / standard error of those paired differences; "
        "Student-t with n_seed-1 degrees of freedom under normal differences, not a standard normal Z; "
        "correlated multiple comparisons, not a known-truth bias estimate"
    )
    (output / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(f"wrote diagnostics to {output}; based on matched seeds {common_seeds}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("export", help="run on HPC; retain views and seed samples, not just aggregates")
    p.add_argument("--configs", type=Path, default=Path(__file__).resolve().parent / "generated-configs")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser(
        "report", help="run wherever the compact exports are available; no Hyperion required"
    )
    p.add_argument("archives", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", default="az1")
    args = parser.parse_args()
    if args.command == "export":
        export(args.configs, args.runs, args.output)
    else:
        report(args.archives, args.output, args.baseline)


if __name__ == "__main__":
    main()
