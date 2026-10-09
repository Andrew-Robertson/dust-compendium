"""Submit the frozen pilot as terminal-independent, dependency-chained Slurm arrays."""

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def command(argv):
    return subprocess.run(argv, capture_output=True, text=True, check=True)


def array_limit(config):
    """For contiguous zero-based arrays, respect both index and task-count limits."""
    match = re.search(r"^\s*MaxArraySize\s*=\s*(\d+)\s*$", config, re.M)
    if match is None:
        raise ValueError("scontrol did not report MaxArraySize; refusing to guess")
    limit = int(match[1])
    parameters = re.search(r"^\s*SchedulerParameters\s*=\s*(.*)$", config, re.M)
    if parameters:
        smaller = re.search(r"(?:^|,)\s*max_array_tasks=(\d+)(?:,|$)", parameters[1])
        if smaller:
            limit = min(limit, int(smaller[1]))
    if limit < 1:
        raise ValueError("Slurm arrays are disabled on this cluster")
    return limit


def unfinished(design, manifest, root, stage):
    import pilot

    pending, done = [], 0
    fingerprint = pilot.sha(manifest)
    for task in pilot.tasks(design, stage):
        path = root / "tasks" / pilot.task_id(task) / "result.npz"
        if path.exists():
            _, data = pilot.read_result(path, fingerprint, task)
            if data["transmission"].shape != (len(design["inclinations"]),):
                raise ValueError(f"incorrect inclination shape: {path}")
            done += 1
        else:
            pending.append(pilot.task_id(task))
    return pending, done


def ensure_idle(root):
    """Do not duplicate a previous array submission; unknown submission state is unsafe."""
    identifiers = set()
    for path in root.glob("arrays/*/receipt.json"):
        receipt = json.loads(path.read_text())
        if receipt["state"] == "submitting":
            raise ValueError(
                f"Unconfirmed submission in {path}. Inspect its attempted job name with squeue/sacct "
                "before retrying; do not blindly resubmit."
            )
        identifiers.update(job["job_id"] for job in receipt["jobs"])
    if identifiers:
        active = set(command(["squeue", "--me", "--noheader", "--format=%F"]).stdout.split())
        overlap = active & identifiers
        if overlap:
            raise ValueError(f"Earlier arrays/export jobs are still active: {', '.join(sorted(overlap))}")


def write_json(path, value):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def shell_script(path, argv, array=False):
    invocation = shlex.join([str(v) for v in argv])
    if array:
        invocation += ' --index "${SLURM_ARRAY_TASK_ID:?not an array task}"'
    path.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
        f"exec {invocation}\n"
    )


def prepare(args, selections, limit, config):
    """Snapshot the worker, task mapping and scientific manifest for this submission."""
    import pilot

    parent = args.runs / "arrays"
    parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="submit-", dir=parent))
    for name in ("pilot.py", "arrays.py", "grain_compatibility.json"):
        shutil.copyfile(HERE / name, folder / name)
    manifest = folder / "pilot.json"
    shutil.copyfile(args.manifest, manifest)
    (folder / "slurm-config.txt").write_text(config)
    (folder / "logs").mkdir()
    common = [
        "--manifest",
        str(manifest),
        "--runs",
        str(args.runs),
        "--grains",
        str(args.grains),
    ]
    plan = []
    for stage, identifiers in selections.items():
        for start in range(0, len(identifiers), limit):
            selected = identifiers[start : start + limit]
            label = f"{stage}-{start // limit:03d}"
            mapping = folder / f"{label}.json"
            write_json(
                mapping,
                {
                    "command": [
                        sys.executable,
                        str(folder / "pilot.py"),
                        "worker",
                        *common,
                        "--stage",
                        stage,
                    ],
                    "tasks": selected,
                },
            )
            script = folder / f"{label}.sh"
            shell_script(
                script, [sys.executable, folder / "arrays.py", "task", "--mapping", mapping], array=True
            )
            plan.append({"label": label, "script": str(script), "count": len(selected), "stage": stage})
        # afterany dependencies allow this audit to run even if some tasks failed.
        # The existing exporter refuses missing/invalid results, never a partial archive.
        label = f"{stage}-export"
        script = folder / f"{label}.sh"
        shell_script(
            script,
            [
                sys.executable,
                folder / "pilot.py",
                "export",
                *common,
                "--stage",
                stage,
                "--output",
                args.runs / "compact" / f"{stage}.npz",
            ],
        )
        plan.append({"label": label, "script": str(script), "count": None, "stage": stage})
    receipt = {
        "state": "prepared",
        "jobs": [],
        "plan": plan,
        "array_size": limit,
        "concurrency": args.concurrency,
        "partition": args.partition,
        "walltime": args.walltime,
        "manifest_sha256": pilot.sha(manifest),
        "python": sys.executable,
    }
    write_json(folder / "receipt.json", receipt)
    return folder, receipt


def submit_plan(folder, receipt, runs):
    """Submit once per chunk, record every ID, then return without polling jobs."""
    path = folder / "receipt.json"
    previous = None
    for item in receipt["plan"]:
        job_name = f"dust-{folder.name}-{item['label']}"
        output = folder / "logs" / f"{item['label']}-%A_%a.log"
        argv = [
            "sbatch",
            "--parsable",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=1",
            "--mem=2048M",
            f"--partition={receipt['partition']}",
            f"--time={receipt['walltime'] if item['count'] else '00:30:00'}",
            f"--job-name={job_name}",
            f"--output={output}",
            f"--error={output}",
            f"--chdir={runs}",
            "--export=ALL",
        ]
        if item["count"]:
            argv.append(f"--array=0-{item['count'] - 1}%{receipt['concurrency']}")
        if previous:
            argv.append(f"--dependency=afterany:{previous}")
        argv.append(item["script"])
        receipt.update(state="submitting", attempted_job_name=job_name, attempted_command=argv)
        write_json(path, receipt)
        # No automatic sbatch retries: an ambiguous response could duplicate jobs.
        try:
            result = command(argv)
        except subprocess.CalledProcessError as exc:
            receipt.update(state="rejected", error=exc.stderr)
            write_json(path, receipt)
            raise RuntimeError(
                f"sbatch rejected {job_name}: {exc.stderr}\nAlready accepted jobs continue. "
                f"Receipt: {path}. Wait for those jobs before submitting only unfinished work again."
            ) from exc
        match = re.fullmatch(r"(\d+)(?:;[\w.-]+)?", result.stdout.strip())
        if not match:
            raise RuntimeError(
                f"Unrecognized sbatch response {result.stdout!r}; inspect {path} before retrying"
            )
        previous = match[1]
        receipt["jobs"].append({**item, "job_id": previous, "job_name": job_name})
        receipt.update(state="partial")
        write_json(path, receipt)
        print(f"{item['label']}: Slurm job {previous}", flush=True)
    receipt.update(state="submitted")
    write_json(path, receipt)
    print(f"All stages submitted; safe to disconnect. Receipt: {path}")
    print("Exports will appear under " + str(runs / "compact"))


def dispatch(mapping, index):
    payload = json.loads(mapping.read_text())
    if not 0 <= index < len(payload["tasks"]):
        raise ValueError(f"array index {index} outside frozen mapping")
    argv = payload["command"] + ["--task", payload["tasks"][index]]
    os.execvpe(argv[0], argv, os.environ.copy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    task_parser = commands.add_parser("task", help="internal array worker dispatch")
    task_parser.add_argument("--mapping", type=Path, required=True)
    task_parser.add_argument("--index", type=int, required=True)
    submit_parser = commands.add_parser("submit", help="plan or submit all arrays and export jobs")
    submit_parser.add_argument("--manifest", type=Path, default=HERE / "pilot.json")
    submit_parser.add_argument("--runs", type=Path, required=True)
    submit_parser.add_argument(
        "--stages",
        nargs="+",
        choices=["timing", "training", "validation"],
        default=["training", "validation"],
    )
    submit_parser.add_argument(
        "--grains", type=Path, default=Path("hyperion-dust-0.1.0/dust_files/d03_4.0_4.0_A.hdf5")
    )
    submit_parser.add_argument("--partition", default="obs")
    submit_parser.add_argument("--walltime", default="08:00:00")
    submit_parser.add_argument("--concurrency", type=int, default=96)
    submit_parser.add_argument(
        "--array-size", type=int, help="optional smaller cap, not a cluster-limit override"
    )
    submit_parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "task":
        dispatch(args.mapping, args.index)
        return

    import pilot

    if args.concurrency < 1 or (args.array_size is not None and args.array_size < 1):
        parser.error("concurrency and array size must be positive")
    if len(set(args.stages)) != len(args.stages) or {"timing", "training"} <= set(args.stages):
        parser.error("stages must be distinct; timing is already part of training")
    args.manifest, args.runs, args.grains = (p.resolve() for p in (args.manifest, args.runs, args.grains))
    design = json.loads(args.manifest.read_text())
    actual = pilot.verified_grain_hash(args.grains, design["grain_sha256"])
    print(f"Verified grain SHA-256: {actual}")
    config = command(["scontrol", "show", "config"]).stdout
    cluster_limit = array_limit(config)
    limit = min(cluster_limit, args.array_size or cluster_limit)
    print(f"Cluster array limit: {cluster_limit}; selected chunk size: {limit}")
    print(f"At most {args.concurrency} one-core tasks at once across this chained submission.")
    args.runs.mkdir(parents=True, exist_ok=True)
    with (args.runs / "array-submission.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_idle(args.runs)
        selections = {}
        for stage in args.stages:
            pending, done = unfinished(design, args.manifest, args.runs, stage)
            selections[stage] = pending
            print(
                f"{stage}: {done} complete; {len(pending)} remaining in "
                f"{(len(pending) + limit - 1) // limit} arrays, then one export job."
            )
        if not args.execute:
            print("Dry run: no jobs submitted. Add --execute to submit. Stop any old polling driver first.")
            return
        folder, receipt = prepare(args, selections, limit, config)
        submit_plan(folder, receipt, args.runs)


if __name__ == "__main__":
    main()
