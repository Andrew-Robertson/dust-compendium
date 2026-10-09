"""Run/export the frozen matched-budget pilot using existing serial Slurm support."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from dustcompendium.campaign import Campaign
from dustcompendium.config import CampaignConfig
from dustcompendium.runner import Job, Resources, is_solved, scheduler, solver_command

HERE = Path(__file__).resolve().parent
CPU = re.compile(r"Total CPU time elapsed:\s*([\d.Ee+\-]+)")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verified_grain_hash(path, expected):
    """Accept only the frozen file or a specifically audited equivalent build."""
    actual = sha(path)
    if actual == expected:
        return actual
    audit = json.loads((HERE / "grain_compatibility.json").read_text())
    if expected == audit["reference_sha256"] and actual in audit["verified_equivalents"]:
        return actual
    raise ValueError(
        f"unverified grain checksum mismatch for {path}: expected {expected}, got {actual}. "
        "Compare numerical optical properties before accepting another build; do not edit the frozen design."
    )


def tasks(design, stage):
    selections = []
    if stage in ("timing", "training"):
        for allocation, count in [("low", design["many_count"]), ("high", design["precise_count"])]:
            if stage == "timing":
                count = design["timing_count"]
            selections += [(allocation, point, 0) for point in design["train"][:count]]
    elif stage == "validation":
        selections = [
            ("validation", point, seed)
            for point in design["validation"]
            for seed in range(design["validation_seeds"])
        ]
    else:
        raise ValueError(f"unknown stage {stage}")
    return [
        (allocation, point, emitter, seed)
        for allocation, point, seed in selections
        for emitter in ("disk", "spheroid")
    ]


def task_id(task):
    allocation, point, emitter, seed = task
    return f"{allocation}_{point['id']}_{emitter}_r{seed}"


def campaign_config(design, task, grains):
    allocation, point, _, replicate = task
    wave, td, ts, q = point["x"]
    budget = design["budgets"][allocation]
    g = design["geometry"]
    # Independent seed families across locations, allocations and validation.
    seed = -(
        1000
        + point["index"] * 20
        + replicate * 3
        + {"low": 0, "high": 10000000, "validation": 20000000}[allocation]
    )
    return CampaignConfig.model_validate(
        {
            "label": task_id(task),
            "dust": {"file": str(grains)},
            "geometry": {
                "spacing": "nested",
                "sampling": "average",
                "cutOff": g["cut_off"],
                "radialCells": g["cells"],
                "verticalCells": g["cells"],
                "innerRadiusFraction": g["inner_radius_fraction"],
                "components": {
                    "disk": {
                        role: {
                            "profile": "exponentialDisk",
                            "scaleRadial": 1.0 if role == "stellar" else g["R_dust_d_over_R_star_d"],
                            "scaleHeight": g["h_star_d_over_R_star_d"]
                            * (1.0 if role == "stellar" else g["h_dust_d_over_h_star_d"]),
                            "verticalStructure": "sechSquared",
                            **({"opticalDepth": td} if role == "dust" else {}),
                        }
                        for role in ("stellar", "dust")
                    },
                    "spheroid": {
                        role: {
                            "profile": "hernquist",
                            "scaleRadial": q * (1.0 if role == "stellar" else g["a_dust_s_over_a_star_s"]),
                            "truncation": 10.0,
                            **({"opticalDepth": ts} if role == "dust" else {}),
                        }
                        for role in ("stellar", "dust")
                    },
                },
            },
            "tabulation": {
                "wavelengths": [wave],
                "inclinations": design["inclinations"],
                "azimuths": design["azimuths"],
                "photons": budget["direct"],
                "imagingPhotons": budget["imaging"],
                "raytracingPhotons": budget["direct"],
                "seed": seed,
                "seedPolicy": "geometry",
            },
        }
    )


def read_result(path, manifest_sha, task):
    with np.load(path, allow_pickle=False) as a:
        meta = json.loads(str(a["metadata"]))
        if meta["manifest_sha256"] != manifest_sha or meta["task"] != task_id(task):
            raise ValueError(f"stale or mismatched result {path}")
        result = {key: a[key] for key in a.files if key != "metadata"}
    if not np.isfinite(result["transmission"]).all() or np.any(result["transmission"] < 0):
        raise ValueError(f"invalid transmission in {path}")
    return meta, result


def worker(design, manifest, task, root, grains, process_start=None):
    from dustcompendium.dust import load_dust
    from dustcompendium.hyperion_model import write_model
    from dustcompendium.postprocess import read_sed_views

    actual_grain_sha = verified_grain_hash(grains, design["grain_sha256"])
    folder = root / "tasks" / task_id(task)
    folder.mkdir(parents=True, exist_ok=True)
    # Prevent two submissions from mutating the same output directory.
    with (folder / "worker.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_sha = sha(manifest)
        result = folder / "result.npz"
        if result.exists():
            read_result(result, manifest_sha, task)
            return
        config = campaign_config(design, task, grains)
        content = {
            "manifest_sha256": manifest_sha,
            "config": config.model_dump(mode="json"),
            "grain_sha256": actual_grain_sha,
        }
        request = folder / "request.json"
        if request.exists():
            stored = json.loads(request.read_text())
            # Before the compatibility audit, only the original hash was accepted.
            stored.setdefault("grain_sha256", design["grain_sha256"])
            if stored != content:
                raise ValueError("existing raw outputs belong to a different request")
        if not request.exists():
            request.write_text(json.dumps(content, indent=2) + "\n")
        wall = time.perf_counter()
        process = time.process_time() if process_start is None else process_start
        flux, reported, cpu = [], [], []
        analytic = task[1]["x"][1:3] == [0.0, 0.0]
        if analytic:
            transmission = np.ones(len(design["inclinations"]))
        else:
            dust = load_dust(str(grains))
            for name in ("dusty", "clear"):
                raw = config.model_dump(mode="json")
                if name == "clear":
                    for component in raw["geometry"]["components"].values():
                        component["dust"]["optical_depth"] = 0.0
                campaign = Campaign(CampaignConfig.model_validate(raw))
                run = next(r for r in campaign.runs() if r.emitter == task[2])
                source, output = folder / f"{name}.hdf5", folder / f"{name}.rtout"
                log = folder / f"{name}.log"
                if not is_solved(output):
                    if not source.exists():
                        write_model(
                            run.spec, dust, str(source), sampling="average", **campaign.grid_options()
                        )
                    with log.open("w") as stream:
                        subprocess.run(
                            solver_command(source, output),
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                if not is_solved(output):
                    raise ValueError(f"solver wrote no SED: {output}")
                matches = CPU.findall(log.read_text())
                if not matches:
                    raise ValueError(f"missing solver CPU time: {log}")
                cpu.append(float(matches[-1]))
                wavelength, values, error = read_sed_views(output, azimuth_count=1)
                if values.shape != (1, len(design["inclinations"]), 1) or not np.allclose(
                    wavelength, [task[1]["x"][0]], rtol=1e-6, atol=0
                ):
                    raise ValueError("unexpected output wavelength or viewing grid")
                flux.append(values[0, :, 0])
                reported.append(error[0, :, 0])
            flux = np.array(flux)
            if not np.isfinite(flux).all() or np.any(flux < 0) or np.any(flux[1] <= 0):
                raise ValueError("invalid flux or normalization")
            transmission = flux[0] / flux[1]
        # Child CPU is recorded from solver logs; process_time counts Python only.
        python_cpu = time.process_time() - process
        meta = {
            "manifest_sha256": manifest_sha,
            "task": task_id(task),
            "point": task[1],
            "allocation": task[0],
            "emitter": task[2],
            "replicate": task[3],
            "analytic": analytic,
            "worker_sha256": sha(__file__),
            "grain_sha256": actual_grain_sha,
            "grain_reference_sha256": design["grain_sha256"],
            "config": config.model_dump(mode="json"),
            "solver_cpu_seconds": sum(cpu),
            "python_cpu_seconds": python_cpu,
            "compute_seconds": sum(cpu) + python_cpu,
            "worker_wall_seconds": time.perf_counter() - wall,
            "timing_note": (
                "resumed raw solves retain solver CPU, but previous Python/I/O overhead is not recovered"
            ),
        }
        temporary = folder / "result.partial.npz"
        np.savez_compressed(
            temporary,
            metadata=json.dumps(meta),
            transmission=transmission,
            flux=np.array(flux),
            reported=np.array(reported),
        )
        os.replace(temporary, result)
        read_result(result, manifest_sha, task)


def submit(design, manifest, args):
    jobs, done = [], 0
    for task in tasks(design, args.stage):
        result = args.runs / "tasks" / task_id(task) / "result.npz"
        if result.exists():
            read_result(result, sha(manifest), task)
            done += 1
            continue
        jobs.append(
            Job(
                label=task_id(task),
                command=[
                    "/usr/bin/env",
                    "OMP_NUM_THREADS=1",
                    "OPENBLAS_NUM_THREADS=1",
                    "MKL_NUM_THREADS=1",
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--manifest",
                    str(manifest),
                    "--runs",
                    str(args.runs),
                    "--grains",
                    str(args.grains),
                    "--stage",
                    args.stage,
                    "--task",
                    task_id(task),
                ],
                log_file=args.runs / "logs" / f"{task_id(task)}.log",
                resources=Resources(partition=args.partition, walltime=args.walltime, memory_per_cpu=2048),
            )
        )
    print(
        f"{args.stage}: {len(jobs)} jobs; {done} complete. Each dusty job runs one dusty + one clear model."
    )
    if not args.execute:
        print("Dry run only. Add --execute to submit.")
        return
    results = scheduler(args.scheduler, concurrency=args.concurrency).run(
        jobs, on_complete=lambda r: print(f"{'ok' if r.succeeded else 'FAILED'} {r.job.label}", flush=True)
    )
    failures = [r for r in results if not r.succeeded]
    if failures:
        raise RuntimeError("\n".join(r.failure_message() for r in failures))


def export(design, manifest, root, stage, output):
    records, transmissions, elapsed, wall = [], [], [], []
    for task in tasks(design, stage):
        path = root / "tasks" / task_id(task) / "result.npz"
        if not path.exists():
            raise ValueError(f"missing result: {path}; no partial export")
        meta, data = read_result(path, sha(manifest), task)
        if data["transmission"].shape != (len(design["inclinations"]),):
            raise ValueError("incorrect inclination shape")
        records.append(meta)
        transmissions.append(data["transmission"])
        elapsed.append(meta["compute_seconds"])
        wall.append(meta["worker_wall_seconds"])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        metadata=json.dumps(
            {"design": design, "manifest_sha256": sha(manifest), "stage": stage, "records": records}
        ),
        transmission=transmissions,
        compute_seconds=elapsed,
        worker_wall_seconds=wall,
    )
    costs = {}
    for allocation in sorted({r["allocation"] for r in records}):
        ids = [i for i, r in enumerate(records) if r["allocation"] == allocation]
        costs[allocation] = {
            "jobs": len(ids),
            "compute_hours": float(np.array(elapsed)[ids].sum() / 3600),
            "worker_wall_hours": float(np.array(wall)[ids].sum() / 3600),
        }
    print(json.dumps(costs, indent=2))
    print(f"wrote {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "worker", "export"])
    parser.add_argument("--manifest", type=Path, default=HERE / "pilot.json")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--stage", choices=["timing", "training", "validation"], required=True)
    parser.add_argument(
        "--grains", type=Path, default=Path("hyperion-dust-0.1.0/dust_files/d03_4.0_4.0_A.hdf5")
    )
    parser.add_argument("--task")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scheduler", choices=["local", "slurm"], default="slurm")
    parser.add_argument("--partition", default="obs")
    parser.add_argument("--walltime", default="02:00:00")
    parser.add_argument("--concurrency", type=int, default=96)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    args.runs = args.runs.resolve()
    args.grains = args.grains.resolve()
    design = json.loads(manifest.read_text())
    if args.command == "worker":
        task = next(t for t in tasks(design, args.stage) if task_id(t) == args.task)
        # A command-line worker owns this process: include its import/setup CPU.
        worker(design, manifest, task, args.runs, args.grains, process_start=0.0)
    elif args.command == "run":
        actual = verified_grain_hash(args.grains, design["grain_sha256"])
        if actual != design["grain_sha256"]:
            print(f"Using verified numerically equivalent grain build: {actual}")
        submit(design, manifest, args)
    else:
        if args.output is None:
            parser.error("export requires --output")
        export(design, manifest, args.runs, args.stage, args.output)


if __name__ == "__main__":
    main()
