#!/usr/bin/env python3
"""Cochran's Q across runs on task data, read against a rest-run null: Part R's test R-2.

For each contrast, each voxel's per-run estimates ``cβ̂_r`` (from ``run_consistency.py``) are pooled
by precision-weighted fixed effects, and

    Q = Σ_r (cβ̂_r − FE)² / cV̂_r c,      df = R_v − 1,

over the runs where that voxel's noise fit is valid (R_v ≥ 2; fewer is excluded and counted).
If the per-run variances are right and the effect does not vary across runs, Q ~ χ²(df). The
cell's numbers, over the whole mask:

- **ρ = median over voxels of Q / median χ²(df)**: a variance-scale statistic. If every V̂ is k×
  too small, ρ ≈ k at any R, so a task cell and a null with different run counts compare on it.
- the fraction of voxels with p < 0.05 against χ²(df), and the median Q / df (the descriptives).

**Reference.** A χ² reference alone cannot separate between-run heterogeneity from a noise
model that is slightly liberal. So the same statistic is computed on the subject's resting runs
under fake designs (``null_design.py --save-maps``), where the true effect is zero in every run.
A null replicate draws R rest runs without replacement (R = the task contrast's run count, capped
at the rest pool) and, for each, one of its design draws. The contrast is matched by shape (see
``NULL_SHAPE``). The task ρ is reported against the null's percentiles.

**Held-out views, reported, not scored:** the runs split into the bake-off halves (odd vs even
positions in (session, run) order). The odd half's fixed-effects z selects voxels — top-N by z at
each of the model's N, and deciles of |z| — and Q is computed in them from the even half alone.

The excess rule belongs to the record that pre-registered it; this script reports the numbers.

Usage (compute node; several GB for the null maps):
    .venv/bin/python scripts/braintwill_glm/score_consistency.py --subject sub-## \\
        --task-dir /path/run_consistency/out --null-dir /path/null_maps --rest-task <rest task> [...] \\
        --out /path/score.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter import config_bids_root  # noqa: E402

from neuroimaging.glm.reference import reference_config  # noqa: E402
from neuroimaging.io import mask_intersection  # noqa: E402

import mmmget  # noqa: E402

#: The bake-off's nine contrasts, by model.
BAKEOFF = {
    "floc": ("faceVsObject", "placeVsObject", "bodyVsObject", "characterVsObject"),
    "motor": ("handVsRest", "footVsRest", "mouthVsRest", "saccadeVsRest"),
    "tbrepetition": ("firstVsLater",),
}
#: Which null design × contrast stands in for each model's contrasts: condition vs condition
#: on blocks, condition vs baseline on blocks, condition vs condition on events.
NULL_SHAPE = {"floc": "block_AvsB", "motor": "block_AvsBaseline", "tbrepetition": "event_AvsB"}
#: The bake-off's top-N sets (``neuroimaging.glm.harness.N_SETS``), repeated here so the scorer
#: does not import the harness's frozen-spec machinery.
N_SETS = {"floc": (250, 500, 1000, 2000), "motor": (500, 1000, 2000, 4000),
          "tbrepetition": (1000, 2500, 5000, 10000)}


def cochran(effect: np.ndarray, variance: np.ndarray):
    """Per voxel: fixed effect, its variance, Q and df over the runs with a valid fit.

    ``effect``/``variance`` are (runs, voxels). A run counts for a voxel where both are finite
    and the variance is positive. Voxels with fewer than two such runs get NaN Q and df 0.
    """
    effect = np.asarray(effect, dtype=float)
    variance = np.asarray(variance, dtype=float)
    valid = np.isfinite(effect) & np.isfinite(variance) & (variance > 0)
    w = np.where(valid, 1.0 / np.where(valid, variance, 1.0), 0.0)
    e = np.where(valid, effect, 0.0)
    sw = w.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fe = (w * e).sum(0) / sw
        fe_var = 1.0 / sw
    q = (w * np.where(valid, (e - fe) ** 2, 0.0)).sum(0)
    df = valid.sum(0) - 1
    q = np.where(df >= 1, q, np.nan)
    return fe, fe_var, q, np.maximum(df, 0)


def summarise_q(q: np.ndarray, df: np.ndarray) -> dict:
    """ρ (median Q / median χ²(df)), fraction p < 0.05, median Q/df, over voxels with df ≥ 1."""
    from scipy import stats

    keep = (df >= 1) & np.isfinite(q)
    if not keep.any():
        return {"voxels": 0, "excluded": int((~keep).sum())}
    qk, dk = q[keep], df[keep]
    levels = np.unique(dk)
    med = np.empty_like(qk)
    for d in levels:
        med[dk == d] = stats.chi2.median(d)
    p = stats.chi2.sf(qk, dk)
    return {"voxels": int(keep.sum()), "excluded": int((~keep).sum()),
            "rho": float(np.median(qk / med)), "frac_p05": float(np.mean(p < 0.05)),
            "median_q_over_df": float(np.median(qk / dk)), "df_min": int(dk.min()), "df_max": int(dk.max())}


def halves(run_labels: list[str], all_runs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (odd, even) over ``run_labels``, by 1-based position in sorted ``all_runs``."""
    pos = {r: i for i, r in enumerate(sorted(all_runs))}
    idx = np.array([pos[r] for r in run_labels])
    return idx % 2 == 0, idx % 2 == 1


def held_out(effect, variance, odd, even, n_set) -> dict:
    """Odd half selects (top-N by z, deciles of |z|); Q comes from the even half only."""
    fe, fe_var, _, _ = cochran(effect[odd], variance[odd])
    with np.errstate(invalid="ignore"):
        z = fe / np.sqrt(fe_var)
    _, _, q, df = cochran(effect[even], variance[even])
    finite = np.isfinite(z)
    out = {"odd_runs": int(odd.sum()), "even_runs": int(even.sum()), "top_n": {}, "abs_z_deciles": []}
    order = np.argsort(np.where(finite, z, -np.inf))[::-1]
    for n in n_set:
        sel = np.zeros(z.shape, bool)
        sel[order[:min(n, int(finite.sum()))]] = True
        out["top_n"][str(n)] = summarise_q(q[sel], df[sel])
    az = np.abs(z[finite])
    edges = np.quantile(az, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, np.abs(z), side="right") - 1, 0, 9)
    for b in range(10):
        sel = finite & (bins == b)
        entry = summarise_q(q[sel], df[sel])
        entry.update({"abs_z_lo": float(edges[b]), "abs_z_hi": float(edges[b + 1])})
        out["abs_z_deciles"].append(entry)
    return out


def load_null(paths: list[Path], shape: str) -> tuple[np.ndarray, np.ndarray]:
    """(runs, draws, voxels) effect and variance for one null design × contrast."""
    effects, variances = [], []
    for path in paths:
        with np.load(path) as z:
            draws = sorted(k[: -len("_effect")] for k in z.files if k.startswith(shape + "_d")
                           and k.endswith("_effect"))
            if not draws:
                raise SystemExit(f"ERROR: {path} has no {shape} maps")
            effects.append(np.stack([z[f"{d}_effect"] for d in draws]))
            variances.append(np.stack([z[f"{d}_variance"] for d in draws]))
    if len({e.shape for e in effects}) != 1:
        raise SystemExit(f"ERROR: null maps for {shape} differ in draws or voxels across runs")
    return np.stack(effects), np.stack(variances)


def null_replicates(effect, variance, n_runs: int, n_rep: int, rng: np.random.Generator) -> dict:
    """ρ and frac p < 0.05 over ``n_rep`` draws of ``n_runs`` rest runs × one design draw each."""
    pool, n_draws, _ = effect.shape
    r = min(n_runs, pool)
    rhos, fracs = np.empty(n_rep), np.empty(n_rep)
    for i in range(n_rep):
        runs = rng.choice(pool, r, replace=False)
        draws = rng.integers(n_draws, size=r)
        _, _, q, df = cochran(effect[runs, draws], variance[runs, draws])
        s = summarise_q(q, df)
        rhos[i], fracs[i] = s["rho"], s["frac_p05"]
    pct = (5, 50, 95, 99)
    return {"runs": int(r), "requested_runs": int(n_runs), "replicates": int(n_rep),
            "rho": {f"p{p}": float(np.percentile(rhos, p)) for p in pct},
            "frac_p05": {f"p{p}": float(np.percentile(fracs, p)) for p in pct},
            "rho_all": rhos.round(5).tolist()}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", required=True)
    p.add_argument("--task-dir", type=Path, required=True, help="run_consistency.py --out-dir")
    p.add_argument("--null-dir", type=Path, required=True, help="null_design.py --save-maps")
    p.add_argument("--rest-task", nargs="+", required=True, help="resting task label(s), as given to null_design.py")
    p.add_argument("--noise", default="ar3-bc")
    p.add_argument("--space", default="T1w")
    p.add_argument("--regime", default="reference")
    p.add_argument("--replicates", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261012)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--maps", type=Path, help="directory for per-contrast Q/median-χ² NIfTIs")
    return p.parse_args(argv)


def main(argv=None) -> int:
    import nibabel as nib

    args = parse_args(argv)
    subject = args.subject.removeprefix("sub-")
    bids_root = config_bids_root()
    cfg = dataclasses.replace(reference_config(args.regime), space=args.space)

    rest = []
    for task in args.rest_task:
        # the same rest pool null_design.py fitted: mmmget.runs, QC exclusions applied
        rest += sorted(mmmget.runs(subject=subject, task=task, variant=cfg.variant, space=args.space,
                                   bids_root=bids_root), key=lambda r: (r.session, r.run or ""))
    null_paths = [args.null_dir / f"sub-{subject}" / f"sub-{subject}_{r.entity_prefix.split('_', 1)[1]}"
                  f"_noise-{args.noise}_maps.npz" for r in rest]
    missing = [str(p) for p in null_paths if not p.exists()]
    if not rest or missing:
        raise SystemExit(f"ERROR: {len(missing)} of {len(rest)} rest runs have no null maps, e.g. {missing[:3]}")

    record = {"subject": subject, "noise": args.noise, "space": args.space, "rest_runs": len(rest),
              "replicates": args.replicates, "seed": [args.seed, int(subject)],
              "statistic": "rho = median_v Q_v / median chi2(R_v - 1)", "null_shape": NULL_SHAPE, "models": {}}
    rng = np.random.default_rng([args.seed, int(subject)])
    null_cache: dict[tuple[str, int], dict] = {}
    null_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def null_for(shape: str, n_runs: int) -> dict:
        if shape not in null_arrays:
            null_arrays[shape] = load_null(null_paths, shape)
        key = (shape, min(n_runs, null_arrays[shape][0].shape[0]))
        if key not in null_cache:
            null_cache[key] = null_replicates(*null_arrays[shape], n_runs, args.replicates, rng)
        return {**null_cache[key], "requested_runs": int(n_runs)}

    for model, contrasts in BAKEOFF.items():
        metas = sorted(args.task_dir.glob(f"sub-{subject}_*model-{model}_space-{args.space}_noise-{args.noise}.json"))
        if len(metas) != 1:
            raise SystemExit(f"ERROR: expected one {model} fit for sub-{subject}, found {[m.name for m in metas]}")
        meta = json.loads(metas[0].read_text())
        all_runs = [r["run"] for r in meta["runs"]]
        shape = NULL_SHAPE[model]
        entry = {"fit": metas[0].name, "runs": len(all_runs), "mask_voxels": meta["mask_voxels"], "contrasts": {}}
        mask_img = mask = None
        if args.maps:
            # include_excluded: the fit's recorded run list decides, so a run QC has
            # excluded since must still be found, not reported missing
            found = {(r.session, r.run): r for r in mmmget.runs(
                subject=subject, task=_task_of(all_runs[0]),
                variant=meta["config"]["variant"], space=args.space, bids_root=bids_root,
                include_excluded=True)}
            mask_img, mask = mask_intersection([found[tuple(w)] for w in meta["stand_in_metadata_runs"]])
            args.maps.mkdir(parents=True, exist_ok=True)
        with np.load(metas[0].with_name(metas[0].stem + "_runs.npz")) as z:
            for name in contrasts:
                effect, variance = z[f"{name}_effect"], z[f"{name}_variance"]
                labels = [str(x) for x in z[f"{name}_runs"]]
                _, _, q, df = cochran(effect, variance)
                task = summarise_q(q, df)
                null = null_for(shape, len(labels))
                c = {"runs": len(labels), "task": task,
                     "null": {k: v for k, v in null.items() if k != "rho_all"},
                     "rho_above_null_p95": bool(task["rho"] > null["rho"]["p95"]),
                     "rho_null_percentile": float(np.mean(np.array(null["rho_all"]) < task["rho"]) * 100)}
                odd, even = halves(labels, all_runs)
                if odd.sum() >= 1 and even.sum() >= 2:
                    ho = held_out(effect, variance, odd, even, N_SETS[model])
                    ho["null_even_half"] = {k: v for k, v in null_for(shape, int(even.sum())).items()
                                            if k != "rho_all"}
                    c["held_out"] = ho
                else:
                    c["held_out"] = {"skipped": f"{int(odd.sum())} odd / {int(even.sum())} even runs; "
                                                "Q needs two runs in the scoring half"}
                entry["contrasts"][name] = c
                print(f"sub-{subject} {name}: R {len(labels)}, rho {task['rho']:.3f} "
                      f"(null p50 {null['rho']['p50']:.3f}, p95 {null['rho']['p95']:.3f}), "
                      f"p<.05 {task['frac_p05']:.3f}")
                if args.maps:
                    s = np.full(q.shape, np.nan, np.float32)
                    from scipy import stats
                    ok = df >= 1
                    s[ok] = q[ok] / stats.chi2.median(df[ok])
                    vol = np.full(mask.shape, np.nan, np.float32)
                    vol[mask] = s
                    nib.Nifti1Image(vol, mask_img.affine).to_filename(
                        args.maps / f"sub-{subject}_model-{model}_contrast-{name}_noise-{args.noise}_qratio.nii.gz")
        record["models"][model] = entry
    for v in null_cache.values():
        v.pop("rho_all", None)
    record["null"] = {f"{s}_R{r}": v for (s, r), v in null_cache.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out}")
    return 0


def _task_of(run_label: str) -> str:
    return next(p.split("-", 1)[1] for p in run_label.split("_") if p.startswith("task-"))


if __name__ == "__main__":
    sys.exit(main())
