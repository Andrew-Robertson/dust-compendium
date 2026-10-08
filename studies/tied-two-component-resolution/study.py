#!/usr/bin/env python3
"""Small spatial-discretization pilot: generate, export, and compare repeated runs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from dustcompendium.config import CampaignConfig

HERE = Path(__file__).resolve().parent
FACTOR = 2.5 / np.log(10.0)
LEVELS = (100, 200, 400)


def exporter():
    path = HERE.parent / "tied-two-component-azimuths/efficiency.py"
    spec = importlib.util.spec_from_file_location("resolution_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configurations(base, seeds=3):
    if seeds < 3:
        raise ValueError("need at least three seeds")
    for cells in LEVELS:
        for seed in range(seeds):
            config = deepcopy(base)
            name = f"grid{cells}_seed{seed}"
            config["label"] = f"experiment:tiedTwoComponent:resolution:{name}"
            config["description"] = "Spatial-grid convergence pilot, not a production design."
            geometry = config["geometry"]
            geometry.update(
                radialCells=cells,
                verticalCells=cells,
                innerRadiusFraction=0.01 * 100 / cells,
                spacing="nested",
                sampling="average",
            )
            for component in ("disk", "spheroid"):
                geometry["components"][component]["dust"]["opticalDepth"] = [0.0, 3.0]
            config["tabulation"].update(
                azimuths=[90.0],
                photons=40000,
                imagingPhotons=40000,
                raytracingPhotons=40000,
                seed=-653 - seed * 10000,
                seedPolicy="geometry",
            )
            CampaignConfig.model_validate(config)
            yield name, config


def difference(coarse, fine):
    """Shift of seed-mean transmissions and its paired-seed standard error.

    Magnitudes are formed from the means, not averaged after taking logs.
    The magnitude SE uses the paired delta method, retaining cross-grid
    covariance. Seed matching does not imply identical photon trajectories.
    """
    if coarse.shape != fine.shape or coarse.shape[0] < 3:
        raise ValueError("need matching arrays with at least three paired seeds")
    n = coarse.shape[0]
    a, b = coarse.mean(axis=0), fine.mean(axis=0)
    dt = coarse - fine
    with np.errstate(divide="ignore", invalid="ignore"):
        da = np.where((a > 0) & (b > 0), -FACTOR * np.log(a / b), np.nan)
        influence = -FACTOR * (coarse / a - fine / b)
    return {
        "delta_T": dt.mean(axis=0),
        "se_T": dt.std(axis=0, ddof=1) / np.sqrt(n),
        "delta_A": da,
        "se_A": influence.std(axis=0, ddof=1) / np.sqrt(n),
    }


def load_matched(paths):
    datasets, common, seed_ids, model_seeds = {}, None, None, None
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            flux = archive["flux"]
            seeds = archive["seeds"].tolist()
            cpu = archive["cpu_seconds"]
        geometry = deepcopy(meta["geometry"])
        cells = geometry.pop("radial_cells")
        vertical = geometry.pop("vertical_cells")
        inner = geometry.pop("inner_radius_fraction", 0.01)
        if cells not in LEVELS or vertical != cells or not np.isclose(inner, 1.0 / cells):
            raise ValueError(f"unexpected refinement settings: {path}")
        settings = {
            k: meta[k]
            for k in (
                "dust",
                "models",
                "references",
                "inclinations",
                "wavelengths",
                "azimuths",
                "imaging_photons",
                "raytracing_photons",
                "seed_policy",
            )
        }
        settings["geometry"] = geometry
        actual_seeds = [p["model_seeds"] for p in meta["provenance"]]
        if common is not None and (common != settings or seeds != seed_ids or actual_seeds != model_seeds):
            raise ValueError(f"unmatched physical settings, budgets or seeds: {path}")
        common, seed_ids, model_seeds = settings, seeds, actual_seeds
        expected = (
            len(seeds),
            len(meta["models"]),
            len(meta["azimuths"]),
            len(meta["inclinations"]),
            len(meta["wavelengths"]),
        )
        if flux.shape != expected or len(seeds) < 3 or len(set(seeds)) != len(seeds):
            raise ValueError(f"invalid flux shape or seed identities: {path}")
        if not np.isfinite(flux).all() or np.any(flux < 0):
            raise ValueError(f"invalid flux: {path}")
        if cells in datasets:
            raise ValueError(f"duplicate grid: {cells}")
        t = exporter().transmission(flux, meta["references"])
        if not np.isfinite(t).all():
            raise ValueError(f"invalid dust-free normalization: {path}")
        datasets[cells] = {"t": t, "cpu": cpu, "metadata": meta}
    if set(datasets) != set(LEVELS):
        raise ValueError("need all three grid levels: 100, 200, 400")
    return datasets


def report(paths, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_matched(paths)
    meta = data[400]["metadata"]
    output.mkdir(parents=True, exist_ok=True)
    dusty = np.arange(len(meta["models"])) != np.asarray(meta["references"])
    shape = data[400]["t"].shape[1:]
    mask = np.broadcast_to(dusty[:, None, None], shape)
    bright = mask.copy()
    for d in data.values():
        bright &= d["t"].mean(axis=0) > 0.01
    pairs = ((100, 200), (200, 400), (100, 400))
    results = {
        "definition": "delta_A = -2.5 log10(mean_seed(T_coarse)/mean_seed(T_fine)); ordinary mag, not E_A",
        "standard_error": (
            "paired-seed delta method for delta_A; sample std(T_coarse-T_fine)/sqrt(N) for delta_T"
        ),
        "mask": (
            "dusty cells; magnitude headline requires mean T > 0.01 at ALL grid levels; equal cell weights"
        ),
        "caveat": (
            "400x400 is a comparison, not known truth; 3 seeds give imprecise SEs; cells are correlated"
        ),
        "seeds": len(data[400]["t"]),
        "archives": [
            {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths
        ],
        "comparisons": {},
        "timings": {},
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for ax, (lo, hi) in zip(axes, pairs, strict=True):
        d = difference(data[lo]["t"], data[hi]["t"])
        label = f"{lo}-{hi}"
        results["comparisons"][label] = {
            "absolute_delta_A": exporter().finite_summary(np.abs(d["delta_A"][bright])),
            "se_A": exporter().finite_summary(d["se_A"][bright]),
            "absolute_delta_T_all_dusty": exporter().finite_summary(np.abs(d["delta_T"][mask])),
            "se_T_all_dusty": exporter().finite_summary(d["se_T"][mask]),
        }
        np.savez_compressed(output / f"difference_{label}.npz", **d, bright_mask=bright, dusty_mask=mask)
        x = data[hi]["t"].mean(axis=0)
        ax.scatter(x[bright], d["delta_A"][bright], s=8, alpha=0.3, label="Shift of seed means")
        ax.scatter(x[bright], d["se_A"][bright], s=5, alpha=0.2, label="Paired 1-SE (positive)")
        ax.axhline(0, color="k", lw=0.6)
        for bound in (-0.01, 0.01):
            ax.axhline(bound, color="gray", ls="--", lw=0.7)
        ax.set(
            xscale="log",
            xlabel=rf"Mean transmission, ${hi}\times{hi}$",
            ylabel=r"$\Delta A$ or SE [mag]",
            title=rf"${lo}\times{lo}$ minus ${hi}\times{hi}$",
        )
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Spatial refinement: ordinary magnitude shifts, NOT noise-normalized errors\n"
        r"$\Delta A = -2.5\log_{10}(\langle T_c\rangle/\langle T_f\rangle)$; "
        r"dusty cells with $\langle T\rangle>0.01$ at every grid; dashed: $\pm0.01$ mag"
    )
    fig.savefig(output / "resolution_summary.png", dpi=170)
    plt.close(fig)

    # Curves for the worst model/inclination combinations; keep complete spectra.
    d = difference(data[100]["t"], data[400]["t"])
    score = np.max(np.where(bright, np.abs(d["delta_A"]), -np.inf), axis=-1)
    order = [v for v in np.argsort(score.ravel())[::-1] if np.isfinite(score.ravel()[v])][:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    wave = np.array(meta["wavelengths"])
    for ax, index in zip(axes.flat, order, strict=False):
        model, inc = np.unravel_index(index, score.shape)
        for level in LEVELS:
            t = data[level]["t"][:, model, inc]
            mean = t.mean(axis=0)
            valid = mean > 0
            a = -FACTOR * np.log(mean[valid])
            se = FACTOR * t[:, valid].std(axis=0, ddof=1) / (np.sqrt(t.shape[0]) * mean[valid])
            ax.errorbar(wave[valid], a, yerr=se, fmt="o-", capsize=2, label=rf"${level}\times{level}$")
        record = meta["models"][model]
        p = record["parameters"]
        ax.set(
            title=(
                f"{record['emitter']}, i={meta['inclinations'][inc]:g}°\n"
                f"τd={p['opticalDepth:disk']:g}, τs={p['opticalDepth:spheroid']:g}, "
                f"q={p['scaleRadial:spheroid']:g}"
            ),
            xscale="log",
            xlabel="Wavelength [μm]",
            ylabel="Attenuation A [mag]",
        )
    for ax in list(axes.flat)[len(order) :]:
        ax.set_visible(False)
    if order:
        axes.flat[0].legend(fontsize=8)
    fig.suptitle(
        "Six largest 100→400 shifts ranked over wavelengths with mean T > 0.01 at all grids\n"
        r"$A=-2.5\log_{10}\langle T\rangle$; bars: delta-method 1-SE of each seed mean; "
        "lines only join sampled wavelengths"
    )
    fig.savefig(output / "resolution_curves.png", dpi=170)
    plt.close(fig)
    for level, d in data.items():
        cpu = d["cpu"]
        results["timings"][str(level)] = {
            "missing": int(np.count_nonzero(~np.isfinite(cpu))),
            "known_solver_core_hours": float(np.nansum(cpu) / 3600),
        }
    (output / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    for name, result in results["comparisons"].items():
        print(f"{name}: |ΔA| {result['absolute_delta_A']}; SE {result['se_A']}")
    print(f"wrote {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("configs")
    make.add_argument("--base", type=Path, default=HERE.parents[1] / "configs/tied-two-component-smoke.yaml")
    make.add_argument("--output", type=Path, default=HERE / "generated-configs")
    make.add_argument("--seeds", type=int, default=3)
    dump = commands.add_parser("export")
    dump.add_argument("--configs", type=Path, default=HERE / "generated-configs")
    dump.add_argument("--runs", type=Path, required=True)
    dump.add_argument("--output", type=Path, required=True)
    analyze = commands.add_parser("report")
    analyze.add_argument("archives", type=Path, nargs="+")
    analyze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "configs":
        base = yaml.safe_load(args.base.read_text())
        args.output.mkdir(parents=True, exist_ok=True)
        for name, config in configurations(base, args.seeds):
            path = args.output / f"{name}.yaml"
            content = yaml.safe_dump(config, sort_keys=False)
            if path.exists() and path.read_text() != content:
                raise ValueError(f"refusing to replace different configuration: {path}")
            path.write_text(content)
            print(path)
    elif args.command == "export":
        exporter().export(args.configs, args.runs, args.output)
    else:
        report(args.archives, args.output)


if __name__ == "__main__":
    main()
