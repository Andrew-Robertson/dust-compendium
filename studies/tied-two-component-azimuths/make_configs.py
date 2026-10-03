#!/usr/bin/env python3
"""Generate reproducible azimuth-convergence campaign configurations."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


AZIMUTH_SETS = {
    "az1": [90.0],
    "az2": [90.0, 270.0],
    "az4": [0.0, 90.0, 180.0, 270.0],
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=root / "configs" / "tied-two-component-smoke.yaml",
        help="base campaign (default: the tied two-component smoke campaign)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "generated-configs",
        help="directory for generated YAML files",
    )
    parser.add_argument("--seeds", type=int, default=5, help="number of independent seed replicates")
    parser.add_argument(
        "--seed-step",
        type=int,
        default=10000,
        help="amount subtracted from the base seed for each replicate",
    )
    parser.add_argument(
        "--photons",
        type=int,
        default=None,
        help="override photons per wavelength and stage",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds <= 1:
        raise SystemExit("--seeds must be at least 2 to measure between-seed scatter")
    with args.base.open(encoding="utf-8") as stream:
        base = yaml.safe_load(stream)
    args.output.mkdir(parents=True, exist_ok=True)
    first_seed = int(base["tabulation"].get("seed", -653))

    for name, azimuths in AZIMUTH_SETS.items():
        for replicate in range(args.seeds):
            config = yaml.safe_load(yaml.safe_dump(base))
            config["label"] = (
                "experiment:tiedTwoComponent:azimuthConvergence:"
                f"{name}:seed{replicate}:dustD03Rv4.0"
            )
            config["description"] = (
                "Azimuth convergence study for the tied disk+spheroid dust model; "
                f"{len(azimuths)} explicit peel-off azimuth(s), seed replicate {replicate}."
            )
            config["tabulation"]["azimuths"] = azimuths
            config["tabulation"]["seed"] = first_seed - replicate * args.seed_step
            if args.photons is not None:
                config["tabulation"]["photons"] = args.photons
            path = args.output / f"{name}_seed{replicate}.yaml"
            with path.open("w", encoding="utf-8") as stream:
                yaml.safe_dump(config, stream, sort_keys=False)
            print(path)


if __name__ == "__main__":
    main()
