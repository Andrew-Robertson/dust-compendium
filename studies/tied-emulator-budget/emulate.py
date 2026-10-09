"""Local PCA--MLP comparison on complete, independent pilot exports (no Hyperion needed)."""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

FACTOR = 2.5 / np.log(10)


def archive(path, stage):
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["metadata"]))
        t, cpu = f["transmission"], f["compute_seconds"]
    if meta["stage"] != stage or len(meta["records"]) != len(t):
        raise ValueError("incorrect or incomplete export")
    if not np.isfinite(t).all() or np.any(t < 0) or not np.isfinite(cpu).all() or np.any(cpu < 0):
        raise ValueError("invalid transmissions or CPU costs")
    return meta, t, cpu


def rows(meta, t, cpu, allocation, emitter):
    records = meta["records"]
    ids = [i for i, r in enumerate(records) if r["allocation"] == allocation and r["emitter"] == emitter]
    ids.sort(key=lambda i: (records[i]["point"]["index"], records[i]["replicate"]))
    return [records[i] for i in ids], t[ids], cpu[ids]


def features(points):
    x = np.array([p["x"] for p in points])
    k = np.array([p["opacity_ratio"] for p in points])
    # Keep wavelength: grain properties are not a dimensional compression.
    return np.column_stack(
        [np.log10(x[:, 0]), np.log10(1 + x[:, 1:3] * k[:, None] / 0.01), np.log10(x[:, 3])]
    )


def angle_weights(angles):
    edges = np.r_[0, (np.array(angles[:-1]) + np.array(angles[1:])) / 2, 90]
    return -np.diff(np.cos(np.deg2rad(edges)))


def matched_count(low_cost, high_cost, minimum, block=16, tolerance=0.05):
    counts = np.arange(minimum, len(low_cost) + 1, block)
    target = np.sum(high_cost)
    if target <= 0 or not len(counts):
        raise ValueError("invalid matching costs")
    totals = np.cumsum(low_cost)[counts - 1]
    best = np.argmin(abs(totals - target))
    ratio = float(totals[best] / target)
    if abs(ratio - 1) > tolerance:
        raise ValueError(
            f"no matched-budget prefix within {tolerance:.0%}: closest cost ratio {ratio:.3f}; extend runs"
        )
    return int(counts[best]), ratio


def fit(x, t, weights, seed, max_iter=800):
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPRegressor

    start = time.perf_counter()
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 0, scale, 1)
    angular = np.sqrt(weights / weights.mean())
    rank = min(12, t.shape[1], len(t) - 1)
    pca = PCA(n_components=rank, svd_solver="full")
    y = pca.fit_transform(t * angular)
    centre = y.mean(axis=0)
    # Same coefficient weighting for every arm, fitted using training labels only.
    coefficient_scale = np.sqrt(np.maximum(y.std(axis=0), 1e-8))
    network = MLPRegressor(
        hidden_layer_sizes=(128, 128, 128),
        activation="relu",
        alpha=1e-6,
        batch_size=min(256, len(t)),
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        tol=1e-6,
        n_iter_no_change=40,
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        network.fit((x - mean) / scale, (y - centre) / coefficient_scale)
    model = {
        "input_mean": mean,
        "input_scale": scale,
        "pca_mean": pca.mean_,
        "basis": pca.components_,
        "coefficient_mean": centre,
        "coefficient_scale": coefficient_scale,
        "angular_scale": angular,
        "layers": np.array(len(network.coefs_)),
    }
    for i, (weight, bias) in enumerate(zip(network.coefs_, network.intercepts_, strict=True)):
        model[f"weight_{i}"] = weight
        model[f"bias_{i}"] = bias
    oracle = pca.inverse_transform(pca.transform(t * angular)) / angular
    info = {
        "fit_seconds": time.perf_counter() - start,
        "iterations": network.n_iter_,
        "warnings": [str(w.message) for w in caught],
        "pca_rank": rank,
        "pca_explained_fraction": float(pca.explained_variance_ratio_.sum()),
        "training_compression_rms_T": float(np.sqrt(np.mean((oracle - t) ** 2))),
    }
    return model, info


def predict(model, x, dust_free=None):
    """Pure-NumPy forward pass; released artifacts need no pickle or sklearn."""
    a = (x - model["input_mean"]) / model["input_scale"]
    for i in range(int(model["layers"])):
        a = a @ model[f"weight_{i}"] + model[f"bias_{i}"]
        if i < int(model["layers"]) - 1:
            a = np.maximum(a, 0)
    t = (
        (a * model["coefficient_scale"] + model["coefficient_mean"]) @ model["basis"] + model["pca_mean"]
    ) / model["angular_scale"]
    if dust_free is not None:
        t[dust_free] = 1
    return t


def weighted_summary(x, weights):
    w = np.broadcast_to(weights, x.shape)
    valid = np.isfinite(x)
    v, w = x[valid], w[valid]
    if not len(v):
        return {"cells": 0}
    w = w / w.sum()
    mean = float(w @ v)
    order = np.argsort(abs(v))
    cdf = np.cumsum(w[order]) - 0.5 * w[order]
    scale = max(float(np.max(abs(v))), np.finfo(float).tiny)
    return {
        "cells": len(v),
        "mean": mean,
        "scatter": float(scale * np.sqrt(w @ ((v / scale - mean / scale) ** 2))),
        "rms": float(scale * np.sqrt(w @ ((v / scale) ** 2))),
        "p95_absolute": float(np.interp(0.95, cdf, abs(v)[order])),
        "maximum_absolute": float(np.max(abs(v))),
    }


def diagnostics(pred, ref, u, weights):
    # Do not hide negative neural-network predictions behind clipping.
    delta = pred - ref
    denom = np.hypot(u, 0.01 / FACTOR * ref)
    ea = np.divide(-0.01 * delta, denom, out=np.full_like(ref, np.nan), where=denom > 0)
    da = np.full_like(ref, np.nan)
    good = (pred > 0) & (ref > 0.01)
    da[good] = -2.5 * np.log10(pred[good] / ref[good])
    return {
        "E_A": weighted_summary(ea, weights),
        "delta_A_Tref_gt_0p01": weighted_summary(da, weights),
        "delta_T": weighted_summary(delta, weights),
        "negative_predictions": int((pred < 0).sum()),
        "invalid_mag_at_bright_reference": int(((ref > 0.01) & (pred <= 0)).sum()),
        "undefined_E_A_cells": int((denom == 0).sum()),
    }


def validate_records(meta, stage):
    from pilot import task_id, tasks

    expected = {task_id(t) for t in tasks(meta["design"], stage)}
    actual = [r["task"] for r in meta["records"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("duplicate or missing training/validation records")


def run(training, validation, output, max_iter=800):
    mt, tt, ct = archive(training, "training")
    mv, tv, cv = archive(validation, "validation")
    validate_records(mt, "training")
    validate_records(mv, "validation")
    if mt["manifest_sha256"] != mv["manifest_sha256"] or mt["design"] != mv["design"]:
        raise ValueError("training/validation designs differ")
    design = mt["design"]
    low_cost = sum(rows(mt, tt, ct, "low", e)[2] for e in ("disk", "spheroid"))
    high_cost = sum(rows(mt, tt, ct, "high", e)[2] for e in ("disk", "spheroid"))
    n, ratio = matched_count(low_cost, high_cost, design["precise_count"])
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "manifest_sha256": mt["manifest_sha256"],
        "matched_low_count": n,
        "cost_ratio_low_high": ratio,
        "precise_locations": design["precise_count"],
        "high_compute_hours": float(high_cost.sum() / 3600),
        "validation_compute_hours": float(cv.sum() / 3600),
        "results": {},
        "definitions": {
            "E_A": "-0.01*(Tpred-Tref)/sqrt(u_ref^2+(ln(10)/2.5*0.01*Tref)^2)",
            "reference": "mean of 3 independent validation seeds; u_ref=sample_std(T)/sqrt(3)",
            "weights": "equal panels within each validation category; uniform cos(inclination) quadrature",
            "target": "PCA of transmission, NOT attenuation; low-noise test never used to fit/select models",
            "cost": "recorded solver + Python CPU; clear controls included; workers are serial",
            "caveat": "small validation sets; repeated inclinations are correlated; 3-seed SEs uncertain",
        },
    }
    weights = angle_weights(design["inclinations"])
    for emitter in ("disk", "spheroid"):
        records, values, _ = rows(mv, tv, cv, "validation", emitter)
        seeds = design["validation_seeds"]
        points = [r["point"] for r in records[::seeds]]
        values = values.reshape(len(points), seeds, -1)
        ref, u = values.mean(axis=1), values.std(axis=1, ddof=1) / np.sqrt(seeds)
        query = features(points)
        categories = np.array([p["category"] for p in points])
        allpred, labels = [], []
        for arm, allocation, count in [
            ("low_same_locations", "low", design["precise_count"]),
            ("low_matched_budget", "low", n),
            ("high_precise", "high", design["precise_count"]),
        ]:
            rec, t, _ = rows(mt, tt, ct, allocation, emitter)
            train_points = [r["point"] for r in rec[:count]]
            x = features(train_points)
            for seed in (13, 29, 47):
                model, info = fit(x, t[:count], weights, seed, max_iter=max_iter)
                started = time.perf_counter()
                pred = predict(model, query)
                info["test_prediction_seconds"] = time.perf_counter() - started
                benchmark_query = np.tile(query, (int(np.ceil(4096 / len(query))), 1))[:4096]
                timings = []
                for _ in range(3):
                    started = time.perf_counter()
                    predict(model, benchmark_query)
                    timings.append(time.perf_counter() - started)
                info["inference_panels_per_second"] = float(4096 / np.median(timings))
                info["inference_scope"] = "4096 panels, median of 3; includes PCA, excludes grain lookup"
                oracle = (
                    (ref * model["angular_scale"] - model["pca_mean"]) @ model["basis"].T @ model["basis"]
                    + model["pca_mean"]
                ) / model["angular_scale"]
                info["validation_compression_oracle"] = diagnostics(oracle, ref, u, weights)
                key = f"{emitter}_{arm}_seed{seed}"
                info.update(
                    training_locations=count,
                    category_metrics={
                        category: diagnostics(
                            pred[categories == category],
                            ref[categories == category],
                            u[categories == category],
                            weights,
                        )
                        for category in np.unique(categories)
                    },
                )
                model["metadata"] = json.dumps(
                    {
                        "manifest_sha256": mt["manifest_sha256"],
                        "emitter": emitter,
                        "bounds": design["bounds"],
                        "inclinations": design["inclinations"],
                        "features": (
                            "log10(lambda), log10(1+tau_d*Klambda/.01), log10(1+tau_s*Klambda/.01), log10(q)"
                        ),
                        "dust_free_policy": "return exactly 1 when both optical depths are zero",
                        "output": (
                            "unclipped transmission; no extrapolation certified; grain opacity required"
                        ),
                    }
                )
                np.savez_compressed(output / f"{key}.npz", **model)
                summary["results"][key] = info
                allpred.append(pred)
                labels.append(key)
                print(f"{key}: {info['fit_seconds']:.1f}s", flush=True)
        np.savez_compressed(
            output / f"{emitter}_predictions.npz",
            predictions=allpred,
            labels=labels,
            reference=ref,
            reference_se=u,
            reference_seeds=values,
            points=json.dumps(points),
        )
        plots(emitter, points, ref, u, np.array(allpred), labels, design["inclinations"], output)
    (output / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


def plots(emitter, points, ref, u, pred, labels, angles, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # First prespecified initialization for each arm; never pick the best using the test.
    chosen = [0, 3, 6]
    denominator = np.hypot(u, 0.01 / FACTOR * ref)
    score = np.divide(abs(pred[3] - ref), denominator, out=np.zeros_like(ref), where=denominator > 0)
    score[(denominator == 0) & (pred[3] != ref)] = np.inf
    worst = np.argmax(np.max(score, axis=1))
    examples = [*np.random.default_rng(190).choice(len(points), 5, replace=False), worst]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, j in zip(axes.flat, examples, strict=True):
        valid = ref[j] > 0
        ax.errorbar(
            np.array(angles)[valid],
            -2.5 * np.log10(ref[j, valid]),
            yerr=FACTOR * u[j, valid] / ref[j, valid],
            fmt="k.-",
            label="RT mean +/- 1-SE",
        )
        for index, style in zip(chosen, [":", "--", "-"], strict=True):
            good = pred[index, j] > 0
            ax.plot(
                np.array(angles)[good],
                -2.5 * np.log10(pred[index, j, good]),
                style,
                label=labels[index].removeprefix(emitter + "_").removesuffix("_seed13"),
            )
        wave, td, ts, q = points[j]["x"]
        ax.set(
            title=f"{points[j]['category']}; {wave:.3g} micron\ntd={td:.2g}, ts={ts:.2g}, q={q:.2g}",
            xlabel="Inclination [degrees]",
            ylabel="Attenuation [mag]",
        )
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(
        f"{emitter}: five random validation locations + worst low-matched-budget case (last)\n"
        "A=-2.5 log10 T; RT reference: 3 seed mean; lines: prespecified MLP seed 13; nonpositive T omitted"
    )
    fig.savefig(output / f"{emitter}_curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        run(args.training, args.validation, args.output)
