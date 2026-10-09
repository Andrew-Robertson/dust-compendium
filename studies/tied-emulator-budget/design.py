"""Freeze a full 4-D, nested RT design; no Hyperion jobs are submitted here."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import qmc

from dustcompendium.dust import load_dust, opacity_to_extinction
from dustcompendium.geometry import HernquistSpheroid

HERE = Path(__file__).resolve().parent
BOUNDS = {"wavelength": [0.048, 2.3], "tau_disk": [0, 30], "tau_spheroid": [0, 30], "q": [0.03, 10]}
PATTERN = ["broad"] * 7 + ["population_widened"] * 6 + ["disk_only", "spheroid_only", "dust_free"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def population(path, kappa):
    """Strict, equal-weight z<3/F158<26 sample, matching the preceding population study."""
    with h5py.File(path, "r") as f:
        g = f["Lightcone/Output1/nodeData"]
        d = {}
        for component, short in [("disk", "d"), ("spheroid", "s")]:
            for suffix, key, unit in [
                ("AbundancesGasMetals", "mz", 1.98892e30),
                ("MassGas", "mg", 1.98892e30),
                ("MassStellar", "ms", 1.98892e30),
                ("Radius", "r", 3.08567758135e16),
            ]:
                ds = g[component + suffix]
                d[key + short] = ds[:].astype(float) * float(ds.attrs["unitsInSI"]) / unit
        selected = (g["lightconeRedshiftCosmological"][:] < 3) & (
            f["Lightcone/Output1/dustAttenuatedNodeData/apparentMagnitudeRomanWFI:F158"][:] < 26
        )
        weights = g["angularWeight"][:] * g["nodeSubsamplingWeight"][:]
        if not np.allclose(weights, weights[0]):
            raise ValueError("catalog weights are not constant")
    valid = selected.copy()
    for values in d.values():
        valid &= np.isfinite(values) & (values >= 0)
    valid &= (d["msd"] + d["mss"] > 0) & (d["rd"] > 0) & (d["rs"] > 0)
    valid &= (d["mzd"] <= d["mgd"]) & (d["mzs"] <= d["mgs"])
    rows = np.flatnonzero(valid)
    factor = kappa * 1.98892e33 / (3.08567758135e18**2) * 0.4 * 0.75 / (2 * np.pi)
    c = HernquistSpheroid(1, truncation=10).optical_depth_integral / (10 / 11) ** 2
    x = np.column_stack(
        [
            factor * d["mzd"][rows] / d["rd"][rows] ** 2,
            factor * c * d["mzs"][rows] / d["rs"][rows] ** 2,
            d["rs"][rows] / d["rd"][rows],
        ]
    )
    inside = (x[:, :2] <= 30).all(axis=1) & (x[:, 2] >= 0.03) & (x[:, 2] <= 10)
    info = {
        "catalog_sha256": digest(path),
        "catalog_name": Path(path).name,
        "selected_rows": int(selected.sum()),
        "strict_positive_sizes_rows": len(rows),
        "in_pilot_domain": int(inside.sum()),
        "dust_to_metals": 0.4,
        "cloud_fraction": 0.25,
        "kappa_V": kappa,
        "spheroid_mass_coefficient": c,
        "selection": (
            "z<3, existing dust-attenuated F158<26; strict nonnegative masses, metals<=gas; both radii>0"
        ),
    }
    if inside.sum() < 100:
        raise ValueError("insufficient admissible catalog rows")
    return x[inside], rows[inside], info


def flatten_cdf(x):
    y = np.column_stack([np.log10(1 + x[:, :2] / 0.01), np.log10(x[:, 2])])
    lower = np.array([0, 0, np.log10(0.03)])
    upper = np.array([np.log10(3001), np.log10(3001), 1])
    bins = np.minimum(7, np.maximum(0, ((y - lower) / (upper - lower) * 8).astype(int)))
    _, inv, count = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    weights = 1 / np.sqrt(count[inv])
    return np.cumsum(weights) / weights.sum()


def samples(n, seed, pool, rows, pattern):
    """Category-specific Sobol sequences avoid conditioning coordinates on index parity."""
    cdf = flatten_cdf(pool)
    streams = {
        tag: qmc.Sobol(8, scramble=True, seed=seed + i * 1009) for i, tag in enumerate(sorted(set(pattern)))
    }
    points = []
    for i in range(n):
        tag = pattern[i % len(pattern)]
        for _ in range(10000):
            u = streams[tag].random(1)[0]
            wave = np.exp(np.log(0.048) + u[0] * np.log(2.3 / 0.048))
            td, ts = 0.01 * np.expm1(u[1:3] * np.log(3001))
            q = np.exp(np.log(0.03) + u[3] * np.log(10 / 0.03))
            row = None
            if tag.startswith("population"):
                index = (
                    min(int(u[1] * len(pool)), len(pool) - 1)
                    if tag == "population"
                    else min(np.searchsorted(cdf, u[1]), len(pool) - 1)
                )
                td, ts, q = pool[index]
                row = int(rows[index])
                if tag == "population_widened":
                    # Shared dust normalization plus independent physical radius changes.
                    norm, rd, rs = (u[4] - 0.5) * 0.6, (u[5] - 0.5) * 0.3, (u[6] - 0.5) * 0.3
                    td, ts, q = td * 10 ** (norm - 2 * rd), ts * 10 ** (norm - 2 * rs), q * 10 ** (rs - rd)
                if td > 30 or ts > 30 or not 0.03 <= q <= 10:
                    continue  # Reject, never pile up clipped points on a domain edge.
            if tag == "disk_only":
                ts = 0.0
            elif tag == "spheroid_only":
                td = 0.0
            elif tag == "dust_free":
                td = ts = 0.0
            points.append(
                {"x": [float(wave), float(td), float(ts), float(q)], "category": tag, "catalog_row": row}
            )
            break
        else:
            raise ValueError("population widening rejection exhausted")
    return points


def make_design(pool, rows, population_info, opacity_ratio, grain_sha, many=4096, precise=512):
    if precise < 32 or precise % 16 or many % 16 or many <= precise:
        raise ValueError("counts must be multiples of 16, 32 <= precise < many")
    train = samples(many, 202610090, pool, rows, PATTERN)
    test_pattern = ["population"] * 8 + ["broad"] * 4 + ["disk_only"] * 2 + ["spheroid_only"] * 2
    test = samples(128, 202610091, pool, rows, test_pattern)
    for wave in [0.55, 1.0]:
        test.append({"x": [wave, 3.0, 3.0, 10.0], "category": "resolution_watch", "catalog_row": None})
    for group, points in [("train", train), ("validation", test)]:
        for i, record in enumerate(points):
            record.update(
                id=f"{group}_{i:05d}",
                index=i,
                split=group,
                opacity_ratio=float(opacity_ratio(record["x"][0])),
            )
    if len({tuple(p["x"]) for p in train + test}) != len(train) + len(test):
        raise ValueError("duplicate 4-D locations or training/validation overlap")
    return {
        "version": 1,
        "bounds": BOUNDS,
        "population": population_info,
        "grain_sha256": grain_sha,
        "precise_count": precise,
        "many_count": many,
        "timing_count": 16,
        "validation_seeds": 3,
        "budgets": {
            "low": {"imaging": 40000, "direct": 10000},
            "high": {"imaging": 320000, "direct": 80000},
            "validation": {"imaging": 640000, "direct": 160000},
        },
        "geometry": {
            "cells": 100,
            "inner_radius_fraction": 0.01,
            "cut_off": 10,
            "h_star_d_over_R_star_d": 0.137,
            "R_dust_d_over_R_star_d": 1,
            "h_dust_d_over_h_star_d": 1,
            "a_dust_s_over_a_star_s": 1,
        },
        "inclinations": np.linspace(0, 90, 19).tolist(),
        "azimuths": [90.0],
        "train": train,
        "validation": test,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--grains", type=Path, required=True)
    p.add_argument("--output", type=Path, default=HERE / "pilot.json")
    args = p.parse_args()
    grains = load_dust(str(args.grains))
    kappa = opacity_to_extinction(grains)
    pool, rows, info = population(args.catalog, kappa)
    result = make_design(
        pool, rows, info, lambda w: opacity_to_extinction(grains, w) / kappa, digest(args.grains)
    )
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output.exists() and args.output.read_text() != content:
        raise ValueError("refusing to change an existing design")
    args.output.write_text(content)
    print(json.dumps(info, indent=2))
    print(
        f"wrote {args.output}; {len(result['train'])} training "
        f"and {len(result['validation'])} validation locations"
    )


if __name__ == "__main__":
    main()
