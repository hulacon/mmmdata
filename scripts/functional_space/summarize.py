#!/usr/bin/env python3
"""Descriptive summaries of the functional-space scores, with hierarchical-bootstrap intervals.

Pre-registration §9: intervals for figures and every secondary analysis come
from a hierarchical bootstrap (draws, then films within each draw), reported
without Holm. This module only describes: the H1/H2 verdict is
``score_route.py decide``'s.

Per level (scenario x overlap %), target (and the mean over targets), film set,
family, ROI and model, it reports the draw-averaged score and its interval. It
also reports the gain over each reference model in REFERENCES, with an interval.
Per-film metrics (M2b, M2a, M4) resample draws and then films. Per-job metrics
(M3, M1) resample draws only. The interval for the mean over targets averages
the targets' replicates, so the targets are treated as fixed.

Inputs: every job's ``scores/<scenario>/pct-<pct>/draw-<draw>/target-<sub>/``,
and its ``families/`` subdirectory where ``--families`` is given. A job the
partition table lists but which has no scores is an error.

Outputs: ``<derivatives>/functional_space/scores/summary/`` (``--out`` overrides):
``m2b.tsv``, ``m2a.tsv``, ``m3.tsv``, ``m1.tsv``, ``m4.tsv`` and, with
``--families``, ``smoothing.tsv``.

Usage:
    python summarize.py [--families] [--n-boot 1000] [--seed 0]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT / "src" / "python", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

#: Reference models a gain is reported against, where the level has them. The psytwill arm adds EBind's
#: routes, scored in the same jobs on the same columns, so a gain over them is the paired psytwill - EBind
#: difference (P1-P4, DECIDED 2026-10-07).
REFERENCES_BY_ARM = {"ebind": ("anatomical", "cha"),
                     "psytwill": ("anatomical", "cha", "stimulus-ebind", "combined-ebind")}
REFERENCES = REFERENCES_BY_ARM["ebind"]
CI = (2.5, 97.5)


def job_dirs(derivatives: Path, parts: pd.DataFrame, families: bool, arm: str = "ebind",
             scenario: str | None = None, pct: int | None = None) -> list[tuple[dict, Path]]:
    import partitions as pt
    import score_route as scr

    out, missing = [], []
    jobs = pt.job_list(parts)
    if scenario is not None:
        jobs = jobs[jobs["scenario"] == scenario]
    if pct is not None:
        jobs = jobs[jobs["pct"] == pct]
    for j in jobs.itertuples(index=False):
        d = scr.out_dir(derivatives, j.scenario, int(j.pct), int(j.draw), j.target, arm)
        if families:
            d = d / "families"
        if not d.is_dir():
            missing.append(str(d))
            continue
        out.append(({"scenario": j.scenario, "pct": int(j.pct), "draw": int(j.draw), "target": j.target}, d))
    if missing:
        raise FileNotFoundError(f"{len(missing)} job score dirs missing, e.g. {missing[:3]}")
    return out


def read_all(jobs: list[tuple[dict, Path]], name: str) -> pd.DataFrame:
    frames = []
    for key, d in jobs:
        path = d / name
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing")
        df = pd.read_parquet(path) if name.endswith(".parquet") else pd.read_csv(path, sep="\t")
        frames.append(df.assign(**key))
    return pd.concat(frames, ignore_index=True)


def boot_indices(n_draws: int, n_films: int | None, n_boot: int, rng) -> tuple[np.ndarray, np.ndarray | None]:
    d = rng.integers(0, n_draws, size=(n_boot, n_draws))
    f = None if n_films is None else rng.integers(0, n_films, size=(n_boot, n_draws, n_films))
    return d, f


def replicate_means(cube: np.ndarray, d: np.ndarray, f: np.ndarray | None) -> np.ndarray:
    """``cube`` (cells, draws[, films]) -> (cells, n_boot) bootstrap means: draws, then films within draw."""
    if f is None:
        return np.nanmean(cube[:, d], axis=2)
    b = cube[:, d[:, :, None], f]  # cells, boot, draws, films
    return np.nanmean(b, axis=(2, 3))


def summarize_metric(df: pd.DataFrame, value: str, group: list[str], unit: list[str], film: str | None,
                     n_boot: int, seed: int, references: tuple[str, ...] = REFERENCES) -> pd.DataFrame:
    """Draw-averaged ``value`` and gains over REFERENCES, with bootstrap intervals, per ``group`` x ROI x model.

    ``unit`` names the ROI column(s). ``film`` is the per-film column; None
    makes it a per-job metric.
    """
    rows = []
    keys = [k for k in group if k != "target"]
    for gkey, g in df.groupby(keys, dropna=False):
        gkey = dict(zip(keys, gkey if isinstance(gkey, tuple) else (gkey,)))
        reps_by_target: dict[str, dict] = {}
        for t, gt in g.groupby("target"):
            draws = np.sort(gt["draw"].unique())
            films = None if film is None else np.sort(gt[film].unique())
            idx = [*unit, "model"]
            cols = ["draw"] if film is None else ["draw", film]
            piv = gt.pivot_table(index=idx, columns=cols, values=value, aggfunc="mean", dropna=False)
            piv = piv.reindex(columns=draws if film is None else pd.MultiIndex.from_product([draws, films]))
            cube = piv.to_numpy().reshape(len(piv), len(draws), *(() if film is None else (len(films),)))
            rng = np.random.default_rng([seed, int(t)])
            d, f = boot_indices(len(draws), None if film is None else len(films), n_boot, rng)
            reps = replicate_means(cube, d, f)
            point = np.nanmean(cube.reshape(len(piv), -1), axis=1)
            reps_by_target[t] = {"index": piv.index, "reps": reps, "point": point}
        targets = sorted(reps_by_target)
        index = reps_by_target[targets[0]]["index"]
        if any(not reps_by_target[t]["index"].equals(index) for t in targets):
            raise ValueError(f"{gkey}: targets differ in ROI x model cells")
        pooled = {"index": index, "reps": np.mean([reps_by_target[t]["reps"] for t in targets], axis=0),
                  "point": np.mean([reps_by_target[t]["point"] for t in targets], axis=0)}
        for t, r in [*reps_by_target.items(), ("all", pooled)]:
            frame = pd.DataFrame(list(r["index"]), columns=[*unit, "model"])
            lo, hi = np.nanpercentile(r["reps"], CI, axis=1)
            out = frame.assign(**gkey, target=t, mean=r["point"], ci_lo=lo, ci_hi=hi)
            for ref in references:
                gm = np.full(len(frame), np.nan)
                glo, ghi = gm.copy(), gm.copy()
                pos = {tuple(v): i for i, v in enumerate(frame[[*unit, "model"]].itertuples(index=False, name=None))}
                for i, v in enumerate(frame[[*unit, "model"]].itertuples(index=False, name=None)):
                    j = pos.get((*v[:-1], ref))
                    if j is None:
                        continue
                    diff = r["reps"][i] - r["reps"][j]
                    gm[i] = r["point"][i] - r["point"][j]
                    glo[i], ghi[i] = np.nanpercentile(diff, CI)
                out[f"gain_{ref}"] = gm
                out[f"gain_{ref}_lo"] = glo
                out[f"gain_{ref}_hi"] = ghi
            rows.append(out)
    res = pd.concat(rows, ignore_index=True)
    res = res[res["mean"].notna()]  # pivot_table(dropna=False) spans every index combination; drop the empty ones
    order = [*keys, "target", *unit, "model"]
    return res[order + [c for c in res.columns if c not in order]].sort_values(order).reset_index(drop=True)


def main() -> None:
    import encoding as enc
    import partitions as pt
    import score_route as scr
    import scoring as sc

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", action="store_true", help="read each job's families/ scores (all §10 families)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--arm", default="ebind", choices=tuple(REFERENCES_BY_ARM))
    ap.add_argument("--scenario", default=None, help="only this scenario's jobs (default: every job)")
    ap.add_argument("--pct", type=int, default=None, help="only this overlap level's jobs (default: every level)")
    args = ap.parse_args()
    refs = REFERENCES_BY_ARM[args.arm]

    paths = enc.Paths()
    parts = pt.load_partitions(paths.derivatives)
    jobs = job_dirs(paths.derivatives, parts, args.families, args.arm, args.scenario, args.pct)
    dest = args.out or (scr.scores_root(paths.derivatives, args.arm) / "summary"
                        / ("families" if args.families else "primary"))
    dest.mkdir(parents=True, exist_ok=True)
    level = ["scenario", "pct", "target"]
    fam = ["family"] if args.families else []
    written = {}

    m2b = read_all(jobs, "m2b.parquet")
    m2b = m2b.groupby([*level, "draw", "set", *fam, "network", "model", "film"], as_index=False)[["rank_acc", "top1"]].mean()
    for value in ("rank_acc", "top1"):
        t = summarize_metric(m2b, value, [*level, "set", *fam], ["network"], "film",
                             args.n_boot, args.seed, references=refs)
        t.to_csv(dest / f"m2b_{value}.tsv", sep="\t", index=False, float_format="%.5g")
        written[f"m2b_{value}"] = len(t)
    m2a = read_all(jobs, "m2a.tsv")
    t = summarize_metric(m2a, "r", [*level, "set", *fam], ["network"], "film",
                         args.n_boot, args.seed, references=refs)
    t.to_csv(dest / "m2a.tsv", sep="\t", index=False, float_format="%.5g")
    written["m2a"] = len(t)
    m3 = read_all(jobs, "m3.tsv")
    m3_group = [*level, "foils", *fam, *(["items"] if args.families else [])]
    t = summarize_metric(m3, "rank_acc", m3_group, ["network"], None, args.n_boot, args.seed, references=refs)
    t.to_csv(dest / "m3.tsv", sep="\t", index=False, float_format="%.5g")
    written["m3"] = len(t)
    m1 = read_all(jobs, "m1.tsv")
    m1 = m1[m1["contrast"].isin(["median", "angle"])]
    t = summarize_metric(m1, "r", [*level, *fam], ["network", "component", "read"], None,
                         args.n_boot, args.seed, references=refs)
    t.to_csv(dest / "m1.tsv", sep="\t", index=False, float_format="%.5g")
    written["m1"] = len(t)
    if args.families:
        sm = read_all(jobs, "smoothing.tsv")
        t = (sm.groupby(["scenario", "pct", "target", "model", "steps"], dropna=False)
             .agg(neighbour_r=("neighbour_r", "mean"), n_jobs=("draw", "nunique")).reset_index())
        t.to_csv(dest / "smoothing.tsv", sep="\t", index=False, float_format="%.5g")
        written["smoothing"] = len(t)
    else:
        m4_jobs = [(k, d) for k, d in jobs if (d / "m4.tsv").exists()]
        m4 = read_all(m4_jobs, "m4.tsv")
        t = summarize_metric(m4, "r", level, ["network", "space", "kind"], "film",
                             args.n_boot, args.seed, references=refs)
        t.to_csv(dest / "m4.tsv", sep="\t", index=False, float_format="%.5g")
        written["m4"] = len(t)
    (dest / "summary.json").write_text(json.dumps({
        "description": "Descriptive functional-space summaries with hierarchical-bootstrap intervals (draws, then "
                       "films within draw; draws only for per-job metrics); gains are model minus reference, "
                       "computed within each bootstrap replicate; target 'all' = mean over targets",
        "n_boot": args.n_boot, "seed": args.seed, "ci": list(CI), "references": list(refs), "arm": args.arm,
        "jobs": {"scenario": args.scenario or "all", "pct": "all" if args.pct is None else args.pct},
        "families": args.families, "rows": written, "code_version": sc._code_version(),
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")
    print(f"wrote {dest}: {written}")


if __name__ == "__main__":
    main()
