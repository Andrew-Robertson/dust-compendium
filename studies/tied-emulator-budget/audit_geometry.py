"""Check the three v1 compendium geometry defects through the pilot's build path."""

import argparse
import json
from pathlib import Path

import numpy as np

import pilot
from dustcompendium.model import dust_density, stellar_emission


def audit(design):
    results = []
    for q in (0.03, 0.137, 1.0, 3.0, 10.0, 31.6):
        point = dict(x=[0.55, 1.0, 1.0, q], index=0, id="audit", category="audit")
        task = ("low", point, "spheroid", 0)
        campaign = pilot.Campaign(pilot.campaign_config(design, task, Path("unused-grain.hdf5")))
        run = next(r for r in campaign.runs() if r.emitter == "spheroid")
        spec = run.spec
        grid = spec.grid(**campaign.grid_options())
        sph = spec.galaxy["spheroid"].stellar
        mass = stellar_emission(spec.galaxy, grid, "spheroid")
        expected = 2 * np.pi * q**3 * (10 / 11) ** 2
        symmetry = float(np.max(abs(mass - mass[:, ::-1, :])) / mass.sum())
        mass_error = float(mass.sum() / expected - 1)
        axis = sph._antiderivative(np.array([0.0, 0.0]), np.array([-1.0, 1.0]))
        np.testing.assert_allclose(axis, [0.25, -0.25], rtol=0, atol=1e-15)
        np.testing.assert_allclose(mass.sum(), expected, rtol=1e-8)
        assert symmetry < 1e-9
        assert sph.density(11 * q, 0) == 0
        assert float(sph.cell_mass(11 * q, 12 * q, 0, q)) == 0
        density = dust_density(spec.galaxy, grid, {"disk": 1.0, "spheroid": 0.0}, 1.0, sampling="average")
        disk = spec.galaxy["disk"].dust
        box_mass = float(
            disk.cell_mass(0, grid.radial_walls[-1], grid.vertical_walls[0], grid.vertical_walls[-1])
        )
        expected_disk = box_mass * disk.density_normalization(1.0, 1.0)
        disk_mass = float(np.sum(density * grid.cell_volumes))
        np.testing.assert_allclose(disk_mass, expected_disk, rtol=1e-12)
        # Sum mean density along the innermost radial column. This is a finite
        # central cylinder, not the exact R=0 ray; near-unity is a resolution check.
        central_depth = float(density[0, :, 0] @ np.diff(grid.vertical_walls))
        assert central_depth > 0.99
        results.append(
            dict(
                q=q,
                relative_spheroid_mass_error=mass_error,
                reflected_cell_error_fraction_of_total=symmetry,
                maximum_spheroid_cell_fraction=float(mass.max() / mass.sum()),
                relative_disk_mass_error=disk_mass / expected_disk - 1,
                central_cylinder_tau_disk=central_depth,
                truncation_radius=sph.truncation_radius,
            )
        )
    return dict(
        reference_pr="https://github.com/galacticusorg/galacticus/pull/1683",
        runtime_source_sha256=pilot.source_fingerprints(),
        checks=results,
        interpretation="Build-path geometry checks, not an independent reanalysis of archived raw Hyperion inputs. q=31.6 is an out-of-domain regression control; the pilot stops at q=10.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=pilot.HERE / "pilot.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(json.loads(args.manifest.read_text()))
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(json.dumps(result["checks"], indent=2))
