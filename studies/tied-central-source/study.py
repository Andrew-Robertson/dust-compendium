"""Frozen, terminal-independent Slurm study of central cores/cavities and sources."""

import argparse
import fcntl
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from physics import central_grid_column, galaxy, grid_for

HERE = Path(__file__).resolve().parent
WAVELENGTHS = [0.1, 0.2175, 0.55, 1.0, 2.0]
INCLINATIONS = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
CASES = [
    ("compact", 0.01, 1.0, 1.0),
    ("small", 0.1, 1.0, 0.1),
    ("matched", 1.0, 1.0, 1.0),
    ("extended", 10.0, 1.0, 1.0),
    ("sphere", 1.0, 0.0, 0.1),
    ("disk_control", 1.0, 1.0, 0.0),
]
PROFILES = [("cusp", 0.0), ("cavity", 0.003), ("cavity", 0.01), ("core", 0.003), ("core", 0.01)]
LEVELS = ["legacy", "inner8", "inner16"]
CPU = re.compile(r"Total CPU time elapsed:\s*([\d.Ee+\-]+)")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(".partial.json")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def task_list():
    tasks = []
    for case, q, td, ts in CASES:
        for kind, e in PROFILES if ts > 0 else PROFILES[:1]:
            for level in LEVELS:
                for seed in range(3):
                    for emitter in ["disk", "spheroid", "central"]:
                        name = f"{case}_{kind}{e:g}_{level}_{emitter}_s{seed}"
                        tasks.append(
                            dict(
                                id=name,
                                case=case,
                                q=q,
                                td=td,
                                ts=ts,
                                kind=kind,
                                epsilon=e,
                                level=level,
                                seed=seed,
                                emitter=emitter,
                            )
                        )
    return tasks


def read_design(root):
    return json.loads((root / "design.json").read_text()), sha(root / "design.json")


def prepare(args):
    import dustcompendium

    root = args.runs.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "design.json").exists():
        raise ValueError("design exists; use status/submit, not prepare")
    if any(root.iterdir()):
        raise ValueError("prepare needs an empty directory")
    reference = json.loads((HERE.parent / "tied-emulator-budget/grain_compatibility.json").read_text())
    if sha(args.grains) not in [reference["reference_sha256"], *reference["verified_equivalents"]]:
        raise ValueError("grain file has not passed the existing D03 compatibility audit")
    snapshot = root / "snapshot"
    snapshot.mkdir()
    for p in [HERE / "study.py", HERE / "physics.py"]:
        shutil.copy2(p, snapshot / p.name)
    shutil.copytree(
        Path(dustcompendium.__file__).parent,
        snapshot / "dustcompendium",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(args.grains, root / "grains.hdf5")
    payload = {str(p.relative_to(root)): sha(p) for p in snapshot.rglob("*.py")}
    payload["grains.hdf5"] = sha(root / "grains.hdf5")
    tasks = task_list()
    design = dict(
        version=1,
        photons=args.photons,
        wavelengths=WAVELENGTHS,
        inclinations=INCLINATIONS,
        azimuths=[90.0],
        tasks=tasks,
        files=payload,
        normalization="same total spheroid dust mass within 10a; tau_s labels original Hernquist mass",
        modification="spheroid dust only; stellar profiles and disk dust unchanged; not a sublimation model",
        python=sys.executable,
    )
    write_json(root / "design.json", design)
    print(
        f"Prepared {len(tasks)} tasks (each dusty + dust-free normalization), 3 seeds; "
        f"{args.photons} photons/stage."
    )
    print(f"Frozen design: {root / 'design.json'}")
    grid_audit(root, design)


def verify(root, design):
    for p, digest in design["files"].items():
        if sha(root / p) != digest:
            raise ValueError(f"frozen input changed: {p}")


def build(task, design, dust, clear=False):
    from hyperion.model import Model
    from hyperion.util.constants import lsun

    from dustcompendium.dust import opacity_to_extinction
    from dustcompendium.hyperion_model import spectrum_for
    from dustcompendium.model import dust_density, stellar_emission, viewing_angles

    g = galaxy(task["q"], task["kind"], task["epsilon"])
    grid = grid_for(g, task["level"])
    model = Model()
    model.set_seed(-(810000 + 1000 * [v[0] for v in CASES].index(task["case"]) + task["seed"]))
    model.set_cylindrical_polar_grid(grid.radial_walls, grid.vertical_walls, grid.azimuthal_walls)
    depths = {"disk": 0.0 if clear else task["td"], "spheroid": 0.0 if clear else task["ts"]}
    model.add_density_grid(
        dust_density(g, grid, depths, opacity_to_extinction(dust), sampling="average"), dust
    )
    if task["emitter"] == "central":
        source = model.add_point_source()
        source.position = (0.0, 0.0, 0.0)
    else:
        source = model.add_map_source()
        source.map = stellar_emission(g, grid, task["emitter"])
    source.luminosity = lsun
    source.spectrum = spectrum_for(dust)
    model.set_monochromatic(True, wavelengths=design["wavelengths"])
    model.set_n_initial_iterations(0)
    model.set_kill_on_absorb(True)
    model.set_raytracing(True)
    image = model.add_peeled_images(image=False)
    image.set_wavelength_index_range(0, len(design["wavelengths"]) - 1)
    image.set_viewing_angles(*viewing_angles(np.array(design["inclinations"]), np.array([90.0])))
    image.set_uncertainties(True)
    image.set_track_origin("basic")
    model.set_n_photons(
        imaging_sources=design["photons"],
        imaging_dust=0,
        raytracing_sources=design["photons"],
        raytracing_dust=0,
    )
    return model


def result(root, design, fingerprint, task):
    path = root / "tasks" / task["id"] / "result.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as a:
        meta = json.loads(str(a["metadata"]))
        data = {k: a[k].copy() for k in a.files if k != "metadata"}
    if meta["design_sha256"] != fingerprint or meta["task"] != task:
        raise ValueError(f"stale result: {path}")
    expected = (len(design["inclinations"]), len(design["wavelengths"]))
    if (
        data["transmission"].shape != expected
        or not np.isfinite(data["transmission"]).all()
        or np.any(data["transmission"] < 0)
    ):
        raise ValueError(f"invalid transmission: {path}")
    if (
        data["flux"].shape != (2, *expected)
        or not np.isfinite(data["flux"]).all()
        or np.any(data["flux"][1] <= 0)
    ):
        raise ValueError(f"invalid normalization: {path}")
    direct = data["direct_transmission"]
    if direct.shape != expected or not np.isfinite(direct).all() or np.any(direct < 0):
        raise ValueError(f"invalid direct transmission: {path}")
    if not np.allclose(data["transmission"], data["flux"][0] / data["flux"][1], rtol=1e-12, atol=0):
        raise ValueError(f"inconsistent flux ratio: {path}")
    if task["emitter"] == "central":
        predicted = data["predicted_grid_direct"]
        if (
            predicted.shape != expected
            or not np.isfinite(predicted).all()
            or not np.allclose(direct, predicted, rtol=3e-4, atol=1e-10)
        ):
            raise ValueError(f"invalid direct point-source ray check: {path}")
    return meta, data


def worker(args):
    from dustcompendium.dust import load_dust
    from dustcompendium.postprocess import read_sed_views
    from dustcompendium.runner import is_solved, solver_command

    root = args.runs.resolve()
    design, fingerprint = read_design(root)
    verify(root, design)
    index = args.index
    if args.mapping:
        index = json.loads(args.mapping.read_text())[index]
    task = design["tasks"][index]
    folder = root / "tasks" / task["id"]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if result(root, design, fingerprint, task) is not None:
            return
        request = dict(task=task, design_sha256=fingerprint)
        if (folder / "request.json").exists() and json.loads(
            (folder / "request.json").read_text()
        ) != request:
            raise ValueError("raw request mismatch")
        write_json(folder / "request.json", request)
        dust = load_dust(str(root / "grains.hdf5"))
        flux = []
        direct = []
        unc = []
        cpu = 0.0
        start = time.perf_counter()
        for name in ["dusty", "clear"]:
            source = folder / f"{name}.hdf5"
            output = folder / f"{name}.rtout"
            log = folder / f"{name}.log"
            if not is_solved(output):
                build(task, design, dust, clear=name == "clear").write(str(source), overwrite=True)
                with log.open("w") as stream:
                    subprocess.run(
                        solver_command(source, output), stdout=stream, stderr=subprocess.STDOUT, check=True
                    )
            if not is_solved(output):
                raise ValueError(f"no solved SED: {output}")
            wave, v, u = read_sed_views(output, azimuth_count=1)
            if v.shape != (1, len(design["inclinations"]), len(design["wavelengths"])) or not np.allclose(
                wave, design["wavelengths"], rtol=1e-6, atol=0
            ):
                raise ValueError("unexpected output axes")
            from hyperion.model import ModelOutput

            direct.append(np.asarray(ModelOutput(str(output)).get_sed(component="source_emit").val)[:, 0, :])
            flux.append(v[0])
            unc.append(u[0])
            times = CPU.findall(log.read_text())
            if not times:
                raise ValueError("missing CPU timing")
            cpu += float(times[-1])
        flux = np.array(flux)
        if not np.isfinite(flux).all() or np.any(flux < 0) or np.any(flux[1] <= 0):
            raise ValueError("invalid flux")
        direct = np.array(direct)
        predicted = np.full(flux[0].shape, np.nan)
        if task["emitter"] == "central":
            from dustcompendium.dust import opacity_to_extinction
            from dustcompendium.model import dust_density

            g = galaxy(task["q"], task["kind"], task["epsilon"])
            grid = grid_for(g, task["level"])
            rho = dust_density(
                g,
                grid,
                {"disk": task["td"], "spheroid": task["ts"]},
                opacity_to_extinction(dust),
                sampling="average",
            )
            column = np.array([central_grid_column(rho, grid, i) for i in design["inclinations"]])
            predicted = np.exp(
                -column[:, None]
                * dust.optical_properties.interp_chi_wav(np.array(design["wavelengths"]))[None, :]
            )
            if not np.allclose(direct[0] / flux[1], predicted, rtol=3e-4, atol=1e-10):
                raise ValueError("central direct Hyperion flux disagrees with exact gridded ray integral")
        tmp = folder / "result.partial.npz"
        meta = dict(**request, solver_cpu_seconds=cpu, worker_wall_seconds=time.perf_counter() - start)
        np.savez_compressed(
            tmp,
            transmission=flux[0] / flux[1],
            direct_transmission=direct[0] / flux[1],
            predicted_grid_direct=predicted,
            flux=flux,
            reported_flux_uncertainty=unc,
            metadata=json.dumps(meta),
        )
        tmp.replace(folder / "result.npz")
        result(root, design, fingerprint, task)
        # Only this task's reproducible scratch intermediates; keep measured flux,
        # uncertainties, logs, request, and frozen scientific inputs permanently.
        if not args.keep_raw:
            for name in ["dusty.hdf5", "dusty.rtout", "clear.hdf5", "clear.rtout"]:
                (folder / name).unlink(missing_ok=True)
        print(task["id"], f"completed; solver CPU {cpu:.1f}s", flush=True)


def grid_audit(root, design):
    from dustcompendium.model import dust_density

    rows = []
    for q in [0.01, 0.1, 1.0, 10.0]:
        for kind, e in PROFILES:
            g = galaxy(q, kind, e)
            profile = g["spheroid"].dust
            norm = profile.density_normalization(1.0, 1.0)
            analytic = norm * profile.central_column()
            for level in LEVELS:
                grid = grid_for(g, level)
                rho = dust_density(g, grid, {"disk": 0.0, "spheroid": 1.0}, 1.0, sampling="average")
                cols = np.array([central_grid_column(rho, grid, i) for i in design["inclinations"]])
                rows.append(
                    dict(
                        q=q,
                        kind=kind,
                        epsilon=e,
                        level=level,
                        grid_shape=grid.shape,
                        first_R_over_a=float(grid.radial_walls[1] / q),
                        first_z_over_a=float(grid.vertical_walls[grid.vertical_walls > 0][0] / q),
                        central_column_per_unit_tau_s=cols.tolist(),
                        analytic_column_per_unit_tau_s=None if not np.isfinite(analytic) else analytic,
                        mass_factor=profile.mass_factor,
                    )
                )
    write_json(root / "grid-audit.json", rows)


def status(args):
    root = args.runs.resolve()
    design, fingerprint = read_design(root)
    done = sum(result(root, design, fingerprint, t) is not None for t in design["tasks"])
    print(f"{done}/{len(design['tasks'])} validated compact results")


def submit(args):
    root = args.runs.resolve()
    design, fingerprint = read_design(root)
    verify(root, design)
    jobs = root / "submissions"
    jobs.mkdir(exist_ok=True)
    with (jobs / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for p in jobs.glob("*/receipt.json"):
            old = json.loads(p.read_text())
            if old["state"] != "submitted":
                raise ValueError(f"inspect unfinished submission {p} before resubmitting")
            active = subprocess.run(
                ["squeue", "--me", "--noheader", "--format=%F"], capture_output=True, text=True, check=True
            ).stdout.split()
            if any(j in active for j in old["job_ids"]):
                raise ValueError("previous submission is still active")
        selected = []
        for i, t in enumerate(design["tasks"]):
            pilot = t["seed"] == 0 and (
                (
                    t["case"] == "sphere"
                    and (
                        (t["emitter"] == "central" and (t["kind"] == "cusp" or t["epsilon"] == 0.003))
                        or (
                            t["emitter"] != "central"
                            and t["kind"] == "cavity"
                            and t["epsilon"] == 0.003
                            and t["level"] == "inner16"
                        )
                    )
                )
                or (t["case"] == "disk_control" and t["emitter"] == "central" and t["level"] == "legacy")
            )
            if (not args.pilot or pilot) and result(root, design, fingerprint, t) is None:
                selected.append(i)
        if not selected:
            print("No unfinished selected tasks")
            return
        cfg = subprocess.run(
            ["scontrol", "show", "config"], capture_output=True, text=True, check=True
        ).stdout
        m = re.search(r"^\s*MaxArraySize\s*=\s*(\d+)", cfg, re.M)
        if not m or len(selected) > int(m[1]):
            raise ValueError("study exceeds cluster array limit")
        m = re.search(r"\bmax_array_tasks=(\d+)", cfg)
        if m and len(selected) > int(m[1]):
            raise ValueError("study exceeds max_array_tasks")
        import tempfile

        folder = Path(tempfile.mkdtemp(prefix="submit-", dir=jobs))
        (folder / "logs").mkdir()
        write_json(folder / "mapping.json", selected)
        common = [design["python"], root / "snapshot/study.py"]

        def script(name, argv, array=False):
            p = folder / f"{name}.sh"
            p.write_text(
                "#!/bin/bash\nset -euo pipefail\n"
                "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
                + f'export PATH={shlex.quote(str(Path(design["python"]).parent))}:"$PATH"\n'
                + f"export PYTHONPATH={shlex.quote(str(root / 'snapshot'))}\n"
                + "exec "
                + shlex.join([str(x) for x in argv])
                + (' --index "${SLURM_ARRAY_TASK_ID:?}"' if array else "")
                + "\n"
            )
            return p

        taskscript = script(
            "tasks", [*common, "worker", "--runs", root, "--mapping", folder / "mapping.json"], True
        )
        record = dict(state="submitting", job_ids=[], count=len(selected), pilot=args.pilot)
        write_json(folder / "receipt.json", record)

        def sbatch(script_path, extra, wall):
            cmd = [
                "sbatch",
                "--parsable",
                "--partition",
                args.partition,
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=1",
                "--mem=4096M",
                f"--time={wall}",
                f"--output={folder}/logs/%x-%A_%a.log",
                *extra,
                str(script_path),
            ]
            out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().split(";")[0]
            if not out.isdigit():
                raise ValueError(f"unrecognized sbatch output {out}")
            record["job_ids"].append(out)
            write_json(folder / "receipt.json", record)
            return out

        job = sbatch(
            taskscript,
            ["--job-name=dust-core", f"--array=0-{len(selected) - 1}%{args.concurrency}"],
            args.walltime,
        )
        exportscript = script(
            "export", [*common, "export", "--runs", root, *(["--allow-partial"] if args.pilot else [])]
        )
        sbatch(exportscript, ["--job-name=dust-core-export", f"--dependency=afterany:{job}"], "00:30:00")
        record["state"] = "submitted"
        write_json(folder / "receipt.json", record)
        print(
            f"Submitted {len(selected)} one-core tasks, at most {args.concurrency} concurrently; "
            "safe to disconnect."
        )
        print(f"Receipt: {folder / 'receipt.json'}")


def export(args):
    root = args.runs.resolve()
    design, fingerprint = read_design(root)
    data = []
    meta = []
    for t in design["tasks"]:
        a = result(root, design, fingerprint, t)
        if a is None:
            if args.allow_partial:
                continue
            raise ValueError(f"missing {t['id']}; run status/submit to resume")
        m, d = a
        data.append(d)
        meta.append(m)
    if not data:
        raise ValueError("no results")
    out = root / "compact"
    out.mkdir(exist_ok=True)
    name = "pilot.npz" if args.allow_partial else "results.npz"
    tmp = out / "export.partial.npz"
    np.savez_compressed(
        tmp,
        **{k: np.stack([d[k] for d in data]) for k in data[0]},
        metadata=json.dumps(meta),
        design=json.dumps(design),
        grid_audit=(root / "grid-audit.json").read_text(),
    )
    tmp.replace(out / name)
    print(
        f"Exported {len(data)}/{len(design['tasks'])} results to {out / name}; "
        f"solver CPU {sum(m['solver_cpu_seconds'] for m in meta) / 3600:.2f} h"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ["prepare", "worker", "submit", "status", "export"]:
        s = sub.add_parser(name)
        s.add_argument("--runs", type=Path, required=True)
        if name == "prepare":
            s.add_argument("--grains", type=Path, required=True)
            s.add_argument("--photons", type=int, default=40000)
        if name == "worker":
            s.add_argument("--index", type=int, required=True)
            s.add_argument("--mapping", type=Path)
            s.add_argument("--keep-raw", action="store_true")
        if name == "submit":
            s.add_argument("--pilot", action="store_true")
            s.add_argument("--concurrency", type=int, default=96)
            s.add_argument("--partition", default="obs")
            s.add_argument("--walltime", default="02:00:00")
        if name == "export":
            s.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    if getattr(args, "photons", 1) < 1 or getattr(args, "concurrency", 1) < 1:
        p.error("counts must be positive")
    globals()[args.command](args)


if __name__ == "__main__":
    main()
