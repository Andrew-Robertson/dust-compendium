"""Paired-seed physical-profile and grid comparisons from compact exports."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FACTOR = 2.5 / np.log(10.0)
COLORS = [".25", "#0072B2", "#56B4E9", "#D55E00", "#E69F00"]
PROFILES = [("cusp", 0.0), ("cavity", 0.003), ("cavity", 0.01), ("core", 0.003), ("core", 0.01)]


def compare(a, b):
    """a,b: paired independent-seed ensembles, then inclination,wavelength."""
    n = len(a)
    ma = a.mean(0)
    mb = b.mean(0)
    dt = (a - b).mean(0)
    # Only this descriptive magnitude summary excludes practically dark cells.
    # Full signed transmission differences include every cell.
    keep = (ma > 1e-4) & (mb > 1e-4)
    da = np.full_like(ma, np.nan)
    da[keep] = -FACTOR * np.log(ma[keep] / mb[keep])
    se_t = np.full_like(ma, np.nan)
    se_a = se_t.copy()
    if n > 1:
        se_t = (a - b).std(0, ddof=1) / np.sqrt(n)
        residual = -FACTOR * (a[:, keep] / ma[keep] - b[:, keep] / mb[keep])
        se_a[keep] = residual.std(0, ddof=1) / np.sqrt(n)
    return dt, da, se_t, se_a, keep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.archive, allow_pickle=False) as a:
        meta = json.loads(str(a["metadata"]))
        design = json.loads(str(a["design"]))
        audit = json.loads(str(a["grid_audit"]))
        t = a["transmission"]
    groups = {}
    for m, v in zip(meta, t, strict=True):
        task = m["task"]
        key = (task["case"], task["kind"], task["epsilon"], task["level"], task["emitter"])
        groups.setdefault(key, {})[task["seed"]] = v
    cases = list(dict.fromkeys(m["task"]["case"] for m in meta))
    waves = np.array(design["wavelengths"])
    angles = np.array(design["inclinations"])
    j = int(np.argmin(abs(angles - 45)))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    summaries = []
    differences = {}
    for case in cases:
        fig, axs = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
        for col, em in enumerate(["disk", "spheroid", "central"]):
            for (kind, e), color in zip(PROFILES, COLORS, strict=True):
                key = (case, kind, e, "inner16", em)
                if key not in groups:
                    continue
                data = np.array(list(groups[key].values()))
                mean = data.mean(0)
                label = kind if kind == "cusp" else f"{kind}: r/a={e:g}"
                good = mean[j] > 0
                axs[0, col].plot(
                    waves[good], -2.5 * np.log10(mean[j, good]), "o-", c=color, label=label, ms=4
                )
                base = groups.get((case, "cusp", 0.0, "inner16", em), {})
                seeds = sorted(set(base) & set(groups[key]))
                if seeds:
                    a = np.array([groups[key][s] for s in seeds])
                    b = np.array([base[s] for s in seeds])
                    dt, da, set_, sea, keep = compare(a, b)
                    axs[1, col].errorbar(
                        waves, da[j], yerr=sea[j] if len(seeds) > 1 else None, c=color, fmt="o-", ms=4
                    )
            old = groups.get((case, "cusp", 0.0, "legacy", em))
            if old:
                mean = np.mean(list(old.values()), axis=0)
                good = mean[j] > 0
                axs[0, col].plot(waves[good], -2.5 * np.log10(mean[j, good]), "k:", label="cusp: legacy grid")
            axs[0, col].set_title(f"{em} source, i={angles[j]:g}°")
            axs[0, col].set_ylabel("Total attenuation A [mag]")
            axs[1, col].set_ylabel("ΔA vs refined cusp [mag]")
            axs[1, col].axhline(0, c=".6", lw=0.6)
            axs[1, col].axhline(0.01, c=".7", ls=":", lw=0.6)
            axs[1, col].axhline(-0.01, c=".7", ls=":", lw=0.6)
            axs[1, col].set_xlabel("Wavelength [μm]")
            for row in [0, 1]:
                axs[row, col].set_xscale("log")
                axs[row, col].grid(alpha=0.15)
        handles, labels = axs[0, 2].get_legend_handles_labels()
        if not handles:
            handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, fontsize=9)
        spec = next(m["task"] for m in meta if m["task"]["case"] == case)
        fig.suptitle(f"{case}: a/Rd={spec['q']:g}, τd={spec['td']:g}, τs={spec['ts']:g}; matched dust mass")
        fig.text(
            0.5,
            0.015,
            "Solid: inner16 grid; dotted: legacy cusp. "
            "Lower panels: modified minus cusp on the SAME refined grid.\n"
            "Bars: paired-seed SE (3 seeds when complete); ΔA shown only where both mean T > 10⁻⁴.\n"
            "Central cusp is a grid-dependent numerical reference, not a converged physical model. "
            "Lines join five sampled wavelengths.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.10, 1, 0.85))
        fig.savefig(args.output / f"attenuation_{case}.png", dpi=170)
        plt.close(fig)
    for key, group in groups.items():
        case, kind, e, level, em = key
        targets = []
        if level == "inner16" and kind != "cusp":
            targets.append(("physical", (case, "cusp", 0.0, level, em)))
        if level in ["legacy", "inner8"]:
            targets.append(("grid", (case, kind, e, "inner16", em)))
        for mode, other in targets:
            if other not in groups:
                continue
            seeds = sorted(set(group) & set(groups[other]))
            a = np.array([group[s] for s in seeds])
            b = np.array([groups[other][s] for s in seeds])
            dt, da, set_, sea, keep = compare(a, b)
            identifier = "|".join(map(str, (mode, *key)))
            summaries.append(
                dict(
                    id=identifier,
                    reference=list(other),
                    seeds=seeds,
                    max_abs_delta_T=float(np.max(abs(dt))),
                    median_abs_delta_A=float(np.median(abs(da[keep]))) if keep.any() else None,
                    max_abs_delta_A=float(np.max(abs(da[keep]))) if keep.any() else None,
                    magnitude_cells=int(keep.sum()),
                    total_cells=keep.size,
                    max_SE_delta_T=float(np.max(set_)) if len(seeds) > 1 else None,
                )
            )
            for label, value in [("delta_T", dt), ("delta_A", da), ("SE_T", set_), ("SE_A", sea)]:
                differences[identifier + "|" + label] = value
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, level in zip(axs, ["legacy", "inner8", "inner16"], strict=True):
        for q in [0.01, 0.1, 1.0, 10.0]:
            r = next(
                r
                for r in audit
                if r["q"] == q and r["kind"] == "cavity" and r["epsilon"] == 0.003 and r["level"] == level
            )
            error = np.array(r["central_column_per_unit_tau_s"]) / r["analytic_column_per_unit_tau_s"] - 1
            ax.plot(angles, 100 * error, "o-", label=f"a/Rd={q:g}")
        ax.axhline(0, c=".5", lw=0.7)
        ax.set_title(level)
        ax.set_xlabel("Inclination [degrees]")
        ax.grid(alpha=0.15)
    axs[0].set_ylabel("Central column error [%]")
    axs[-1].legend(fontsize=9)
    fig.suptitle("Pure spheroid dust: 0.003a cavity; exact grid-ray integral vs analytic spherical column")
    fig.text(
        0.5,
        0.01,
        "This deterministic check isolates spatial-grid error; no photon noise. "
        "A truly spherical column is independent of inclination.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(args.output / "central_columns.png", dpi=170)
    plt.close(fig)
    out = dict(
        comparisons=summaries,
        completed=len(meta),
        planned=len(design["tasks"]),
        solver_cpu_hours=sum(m["solver_cpu_seconds"] for m in meta) / 3600,
        notes=[
            "Magnitude differences are ordinary ΔA, not noise-adjusted E_A.",
            "Three-seed SE is imprecise; a Gaussian 95% paired t interval uses 4.3 SE.",
            "All-cell ΔT retains opaque cases.",
            "Central cusp reference does not have a finite analytic central column.",
        ],
    )
    (args.output / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    np.savez_compressed(args.output / "differences.npz", **differences)
    print(
        f"Wrote {len(summaries)} comparisons to {args.output}; "
        f"{len(meta)}/{len(design['tasks'])} runs present."
    )


if __name__ == "__main__":
    main()
