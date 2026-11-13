"""Append low-photon locations without mutating or rerunning the frozen pilot."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

import design as sampling
import pilot


def extension(parent, regenerated, parent_sha, total):
    count = parent["many_count"]
    if total <= count or total % 16:
        raise ValueError("total must exceed the parent count and be a multiple of 16")
    if len(regenerated["train"]) != total:
        raise ValueError("incorrect regenerated length")
    if parent["population"]["catalog_sha256"] != regenerated["population"]["catalog_sha256"]:
        raise ValueError("population catalog changed")
    # Independently regenerated prefixes certify continuation of the same Sobol
    # streams. Roundoff from the audited equivalent grain build is allowed;
    # original values and records are ALWAYS retained in the combined dataset.
    for group in ("train", "validation"):
        old = parent[group]
        new = regenerated[group][: len(old)]
        if len(old) != len(new):
            raise ValueError("changed reference set")
        for a, b in zip(old, new, strict=True):
            for key in ("id", "index", "split", "category", "catalog_row"):
                if a[key] != b[key]:
                    raise ValueError(f"changed Sobol prefix at {a['id']}: {key}")
            if not np.allclose(a["x"], b["x"], rtol=5e-13, atol=0):
                raise ValueError(f"changed Sobol prefix coordinates at {a['id']}")
            if not np.isclose(a["opacity_ratio"], b["opacity_ratio"], rtol=5e-13, atol=0):
                raise ValueError("grain opacity no longer matches original")
    result = copy.deepcopy(parent)
    result.update(
        train=regenerated["train"][count:],
        validation=[],
        many_count=total - count,
        precise_count=0,
        timing_count=0,
        runtime_source_sha256=pilot.source_fingerprints(),
        retain_raw_first_locations=16,
        extension={
            "parent_manifest_sha256": parent_sha,
            "parent_low_count": count,
            "total_low_count": total,
            "purpose": "Additional low-photon training only; reuse parent validation and high-photon data",
        },
    )
    combined_points = parent["train"] + result["train"] + parent["validation"]
    if len({tuple(p["x"]) for p in combined_points}) != len(combined_points):
        raise ValueError("duplicate training locations or validation overlap")
    return result


def build(args):
    from dustcompendium.dust import load_dust, opacity_to_extinction

    parent = json.loads(args.parent.read_text())
    pilot.verified_grain_hash(args.grains, parent["grain_sha256"])
    dust = load_dust(str(args.grains))
    kappa = opacity_to_extinction(dust)
    if not np.isclose(kappa, parent["population"]["kappa_V"], rtol=5e-13, atol=0):
        raise ValueError("V-band opacity changed")
    pool, rows, info = sampling.population(args.catalog, parent["population"]["kappa_V"])
    regenerated = sampling.make_design(
        pool,
        rows,
        info,
        lambda w: opacity_to_extinction(dust, w) / kappa,
        parent["grain_sha256"],
        many=args.total,
        precise=parent["precise_count"],
    )
    result = extension(parent, regenerated, pilot.sha(args.parent), args.total)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output.exists() and args.output.read_text() != text:
        raise ValueError("refusing to overwrite a different extension manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(f"Preserved {parent['many_count']} old locations; {result['many_count']} new; {args.total} total.")
    print(f"New tasks: {len(pilot.tasks(result, 'training'))}; high-photon and validation tasks: 0.")
    print(f"Manifest SHA-256: {pilot.sha(args.output)}")


def load_archive(path, design, fingerprint, stage):
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["metadata"]))
        arrays = {k: f[k] for k in ("transmission", "compute_seconds", "worker_wall_seconds")}
    if meta["design"] != design or meta["manifest_sha256"] != fingerprint or meta["stage"] != stage:
        raise ValueError(f"incorrect export provenance: {path}")
    expected = pilot.tasks(design, stage)
    if len(meta["records"]) != len(expected):
        raise ValueError("incomplete export")
    for record, task in zip(meta["records"], expected, strict=True):
        if (
            record["task"] != pilot.task_id(task)
            or record["point"] != task[1]
            or record["manifest_sha256"] != fingerprint
            or record["allocation"] != task[0]
            or record["emitter"] != task[2]
            or record["replicate"] != task[3]
        ):
            raise ValueError("wrong, reordered or duplicate result")
        if (
            "runtime_source_sha256" in design
            and record.get("runtime_source_sha256") != design["runtime_source_sha256"]
        ):
            raise ValueError("result source fingerprint mismatch")
    t = arrays["transmission"]
    if t.shape != (len(expected), len(design["inclinations"])) or not np.isfinite(t).all() or (t < 0).any():
        raise ValueError("invalid transmissions")
    for k in ("compute_seconds", "worker_wall_seconds"):
        if arrays[k].shape != (len(expected),) or not np.isfinite(arrays[k]).all() or (arrays[k] < 0).any():
            raise ValueError("invalid timings")
    return meta["records"], arrays


def merge(args):
    parent = json.loads(args.parent.read_text())
    extra = json.loads(args.manifest.read_text())
    if extra["extension"]["parent_manifest_sha256"] != pilot.sha(args.parent):
        raise ValueError("wrong parent manifest")
    full = copy.deepcopy(parent)
    full["train"] += extra["train"]
    full["many_count"] = extra["extension"]["total_low_count"]
    if len(full["train"]) != full["many_count"]:
        raise ValueError("wrong total count")
    full["provenance"] = {
        "parent_manifest_sha256": pilot.sha(args.parent),
        "extension_manifest_sha256": pilot.sha(args.manifest),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "combined.json"
    content = json.dumps(full, indent=2, allow_nan=False) + "\n"
    if manifest.exists() and manifest.read_text() != content:
        raise ValueError("refusing to replace different combined design")
    manifest.write_text(content)
    for stage in ("training", "validation"):
        rec, arr = load_archive(args.original / f"{stage}.npz", parent, pilot.sha(args.parent), stage)
        sources = [args.original / f"{stage}.npz"]
        if stage == "training":
            more, values = load_archive(args.extra, extra, pilot.sha(args.manifest), stage)
            sources.append(args.extra)
            rec += more
            arr = {k: np.concatenate([arr[k], values[k]]) for k in arr}
        # Restore canonical low-then-high ordering. Keep original execution
        # manifests and source fingerprints in each record, without rewriting history.
        lookup = {r["task"]: i for i, r in enumerate(rec)}
        if len(lookup) != len(rec):
            raise ValueError("duplicate task IDs")
        order = [lookup[pilot.task_id(task)] for task in pilot.tasks(full, stage)]
        meta = dict(
            design=full,
            manifest_sha256=pilot.sha(manifest),
            stage=stage,
            records=[rec[i] for i in order],
            source_archives={p.name + f"_{i}": pilot.sha(p) for i, p in enumerate(sources)},
            provenance_note="Top-level manifest describes combined dataset; record manifests are original execution manifests.",
        )
        np.savez_compressed(
            args.output / f"{stage}.npz", metadata=json.dumps(meta), **{k: v[order] for k, v in arr.items()}
        )
        print(f"{stage}: {len(order)} records, {arr['compute_seconds'].sum()/3600:.2f} CPU-hours")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--grains", type=Path, required=True)
    p.add_argument("--total", type=int, default=32768)
    p.add_argument("--parent", type=Path, default=pilot.HERE / "pilot.json")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("merge")
    p.add_argument("--parent", type=Path, default=pilot.HERE / "pilot.json")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--original", type=Path, required=True, help="Parent compact directory")
    p.add_argument("--extra", type=Path, required=True, help="Extension compact/training.npz")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (build if args.command == "build" else merge)(args)


if __name__ == "__main__":
    main()
