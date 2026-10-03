#!/usr/bin/env python3
"""Summarize between-seed convergence for the generated azimuth campaigns."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from dustcompendium.campaign import Campaign
from dustcompendium.config import load_campaign
from dustcompendium.postprocess import attenuation_of, average_azimuths, read_sed_views

NAME = re.compile(r"(?P<variant>az\d+)_seed(?P<seed>\d+)\.yaml$")


def parse_args() -> argparse.Namespace:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs", type=Path, default=directory / "generated-configs", help="generated YAML directory"
    )
    parser.add_argument(
        "--runs",
        type=Path,
        required=True,
        help="root containing one output/<model>.rtout directory per config stem",
    )
    parser.add_argument(
        "--output", type=Path, default=directory / "results.csv", help="summary CSV to write"
    )
    return parser.parse_args()


def reference_indices(campaign: Campaign, emitter: str, indices: tuple[int, ...]) -> tuple[int, ...]:
    reference = list(indices)
    for position, axis in enumerate(campaign.axes_for(emitter)):
        if axis.kind == "opticalDepth":
            zeros = np.flatnonzero(axis.values == 0.0)
            if zeros.size != 1:
                raise ValueError(f"{axis.name} needs exactly one zero entry")
            reference[position] = int(zeros[0])
    return tuple(reference)


def transmissions(config: Path, output: Path) -> tuple[np.ndarray, np.ndarray]:
    """Flatten averaged transmission and its conservative reported uncertainty."""
    campaign = Campaign(load_campaign(str(config)))
    runs = list(campaign.runs())
    lookup = {(run.emitter, run.indices): run for run in runs}
    cached: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def views(run) -> tuple[np.ndarray, np.ndarray]:
        if run.file_stem not in cached:
            path = output / f"{run.file_stem}.rtout"
            _, values, uncertainty = read_sed_views(path, azimuth_count=campaign.azimuths.size)
            cached[run.file_stem] = values, uncertainty
        return cached[run.file_stem]

    estimates = []
    reported = []
    for run in runs:
        reference = lookup[(run.emitter, reference_indices(campaign, run.emitter, run.indices))]
        values, uncertainty = average_azimuths(*views(run))
        normal, normal_uncertainty = average_azimuths(*views(reference))
        transmission, transmission_uncertainty = attenuation_of(
            values, uncertainty, normal, normal_uncertainty
        )
        estimates.append(transmission.ravel())
        reported.append(transmission_uncertainty.ravel())
    return np.concatenate(estimates), np.concatenate(reported)


def quantile(values: np.ndarray, probability: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, probability)) if finite.size else float("nan")


def main() -> None:
    args = parse_args()
    grouped: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    for config in sorted(args.configs.glob("az*_seed*.yaml")):
        match = NAME.match(config.name)
        if match is None:
            continue
        variant = match.group("variant")
        seed = int(match.group("seed"))
        output = args.runs / config.stem / "output"
        estimate, reported = transmissions(config, output)
        grouped[variant].append((seed, estimate, reported))

    rows = []
    for variant, replicates in sorted(grouped.items()):
        replicates.sort(key=lambda item: item[0])
        estimates = np.stack([item[1] for item in replicates])
        reported = np.stack([item[2] for item in replicates])
        truth = np.mean(estimates, axis=0)
        empirical_sigma_t = np.std(estimates, axis=0, ddof=1)
        median_reported_t = np.median(reported, axis=0)
        usable = np.isfinite(truth) & (truth > 1.0e-6)
        factor = 2.5 / np.log(10.0)
        empirical_sigma_a = factor * empirical_sigma_t[usable] / truth[usable]
        ratio = empirical_sigma_t[usable] / median_reported_t[usable]
        row = {
            "variant": variant,
            "azimuths": int(variant[2:]),
            "seeds": len(replicates),
            "cells": int(np.count_nonzero(usable)),
            "median_sigma_T": quantile(empirical_sigma_t[usable], 0.5),
            "p95_sigma_T": quantile(empirical_sigma_t[usable], 0.95),
            "median_sigma_A_mag": quantile(empirical_sigma_a, 0.5),
            "p95_sigma_A_mag": quantile(empirical_sigma_a, 0.95),
            "median_empirical_over_reported": quantile(ratio, 0.5),
            "p95_empirical_over_reported": quantile(ratio, 0.95),
        }
        rows.append(row)

    if not rows:
        raise SystemExit("no complete generated configurations found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['variant']}: median sigma(A)={row['median_sigma_A_mag']:.4g} mag, "
            f"p95={row['p95_sigma_A_mag']:.4g} mag, "
            "median empirical/reported="
            f"{row['median_empirical_over_reported']:.3g}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
