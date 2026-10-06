#!/usr/bin/env python3
"""Three one-azimuth photon-allocation tests, leaving the completed campaigns intact."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from dustcompendium.config import CampaignConfig


def configurations(base, seeds=5, multiplier=4):
    """Keep geometry and seed families fixed; vary the two sampling budgets."""
    if seeds < 2 or multiplier <= 1:
        raise ValueError("need at least two seeds and a multiplier greater than one")
    n = int(base["tabulation"]["photons"])
    for imaging, direct in ((multiplier * n, n), (n, multiplier * n), (multiplier * n, multiplier * n)):
        for seed in range(seeds):
            config = deepcopy(base)
            variant = f"az1_i{imaging}_r{direct}"
            name = f"{variant}_seed{seed}"
            config["label"] = f"experiment:tiedTwoComponent:efficiency:{name}"
            config["description"] = "Photon-allocation convergence test; not a production design."
            config["tabulation"].update(
                azimuths=[90.0],
                imagingPhotons=imaging,
                raytracingPhotons=direct,
                seed=int(base["tabulation"].get("seed", -653)) - seed * 10000,
            )
            CampaignConfig.model_validate(config)
            yield name, config


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=directory.parents[1] / "configs/tied-two-component-smoke.yaml"
    )
    parser.add_argument("--output", type=Path, default=directory / "generated-configs/efficiency")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    base = yaml.safe_load(args.base.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for name, config in configurations(base, seeds=args.seeds):
        path = args.output / f"{name}.yaml"
        content = yaml.safe_dump(config, sort_keys=False)
        if path.exists() and path.read_text() != content:
            raise SystemExit(f"refusing to replace a different configuration: {path}")
        path.write_text(content)
        print(path)


if __name__ == "__main__":
    main()
