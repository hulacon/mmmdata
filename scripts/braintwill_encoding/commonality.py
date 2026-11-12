#!/usr/bin/env python3
"""Commonality partition of brain-explained variance over a harness arm's groups.

An arm (``harness.COMMON_ARMS``) names k groups of feature sources: for example an opaque
network layer, named feature members, and timing/nuisance features. Every non-empty subset
of the groups is scored by the harness as its own space (``common-<arm>.<g>+<g>``, one
kernel band per group, per-group sums kept), so there are 2^k - 1 fits per subject. This
driver fits nothing: it reads those cached scores and turns them into the 2^k - 1
**commonality terms**, per ROI:

- the unique term of a group G is R²(all) - R²(all without G);
- the term of a set S of groups is the variance all of S explain and no group outside S
  does: ``C(S) = sum over T ⊆ S of (-1)^(|S| - |T| + 1) R²(all without T)``, with
  R²(no groups) = 0. For two groups, C({a, b}) = R²(a) + R²(b) - R²(a, b).

The terms sum to R²(all) exactly. They are partitioned, not residualised: no group is
regressed out of another. A negative shared term means the groups' held-out R² does not
add up as a partition, which happens with correlated predictors and with regularised,
cross-validated fits. It is reported as computed and is not evidence of suppression.

Timing and nuisance features belong in their own group. Otherwise variance a timing signal
explains through both the opaque and the named group is booked as "shared opaque ∩ named".

Verbs:

  spaces  the arm's 2^k - 1 harness spaces, one per line, for a submission loop
  read    per subject and ROI (HCP-MMP1 sections, ``families.familyD``, inside the
          register's reliability mask, as ``rescue.py``): every term's ROI mean with a
          percentile bootstrap interval over the CV groups (one draw of counts per subject,
          shared by every subset fit, so the terms are paired) -> JSON + CSV + a
          median-over-subjects CSV. Descriptive, no verdict.

Usage:
    python commonality.py spaces --arm <arm>
    for s in $(python commonality.py spaces --arm <arm>); do
        sbatch --export=ALL,VERB=score,ARGS="--space $s --sub ##" harness.sbatch; done
    python commonality.py read --arm <arm> --register films --out <dir> [--subs ##,##]

A subset fit is cached by its feature content and its spec, which carries the merge (group
names and their bands). So arms that differ only in the opaque group's source, with the same
group names, share every subset that leaves the opaque group out and refit only those that
hold it.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import harness as h  # noqa: E402  (puts functional_space and src/python on sys.path)
import rescue as rs  # noqa: E402  ROI columns, bootstrap counts, cached-score reader

#: Interval level of the percentile bootstrap intervals.
CI = 0.95
#: Separator in a term's name: ``opaque&named`` is the variance opaque and named share and
#: no other group explains.
TERM_SEP = "&"


# ---------------------------------------------------------------------------
# the partition (pure: synthetic R² and group sums test it)
# ---------------------------------------------------------------------------

def subset_key(groups) -> str:
    return "+".join(groups)


def term_weights(groups: list[str]) -> dict[str, dict[str, int]]:
    """Each commonality term as signed weights on subset fits: term -> {subset_key: coefficient}.

    ``groups`` in arm order. Subset keys name the groups a fit holds (``harness`` order), and
    the empty fit (R² = 0) never appears.
    """
    groups = list(groups)
    if len(groups) < 2 or len(set(groups)) != len(groups):
        raise ValueError(f"need at least two distinct groups, got {groups}")
    out = {}
    for k in range(1, len(groups) + 1):
        for S in itertools.combinations(groups, k):
            w: dict[str, int] = {}
            for j in range(0, k + 1):
                for T in itertools.combinations(S, j):
                    fit = tuple(g for g in groups if g not in T)
                    if not fit:
                        continue
                    key = subset_key(fit)
                    w[key] = w.get(key, 0) + (-1) ** (k - j + 1)
            out[TERM_SEP.join(S)] = {key: c for key, c in w.items() if c}
    return out


def terms_from_r2(r2: dict[str, np.ndarray], weights: dict[str, dict[str, int]]) -> dict[str, np.ndarray]:
    """Apply :func:`term_weights` to R² values (scalars or arrays) keyed by subset."""
    return {t: sum(c * np.asarray(r2[s], np.float64) for s, c in w.items()) for t, w in weights.items()}


def term_stats(groups_by_subset: dict[str, dict], rois: dict[str, np.ndarray], counts: np.ndarray,
               weights: dict[str, dict[str, int]], ci: float = CI) -> dict:
    """Per term and per ROI (column indices): the ROI-mean term and a percentile bootstrap interval.

    ``groups_by_subset`` maps subset key -> ``EncodingScore.groups`` of the same responses; all
    must hold the same CV groups in the same order, so one row of ``counts`` resamples every
    subset fit alike. ``total`` (R² of the fit holding every group) is reported beside the terms.
    """
    from braintwill.encoding import resampled_r2

    keys = list(groups_by_subset)
    need = sorted({s for w in weights.values() for s in w})
    missing = [s for s in need if s not in groups_by_subset]
    if missing:
        raise ValueError(f"subset fits missing: {missing}")
    labels = list(groups_by_subset[keys[0]]["labels"])
    for k in keys[1:]:
        if list(groups_by_subset[k]["labels"]) != labels:
            raise RuntimeError(f"subset {k} holds different CV groups from subset {keys[0]}")
    if not rois:
        return {}
    full = max(need, key=lambda s: len(s.split("+")))
    cols = np.unique(np.concatenate(list(rois.values())))
    at = {c: i for i, c in enumerate(cols)}
    idx = {r: np.array([at[c] for c in v]) for r, v in rois.items()}
    g = {k: rs._restrict(groups_by_subset[k], cols) for k in need}
    names = [*weights, "total"]
    ones = np.ones(len(labels))

    def per_column(r2):
        t = terms_from_r2(r2, weights)
        t["total"] = np.asarray(r2[full], np.float64)
        return t

    obs = per_column({k: resampled_r2(g[k], ones) for k in need})
    boot = {t: {r: [] for r in rois} for t in names}
    for s in range(0, counts.shape[0], rs.DRAW_CHUNK):
        c = counts[s:s + rs.DRAW_CHUNK]
        t = per_column({k: resampled_r2(g[k], c) for k in need})
        for name in names:
            for r, ix in idx.items():
                boot[name][r].append(np.nanmean(t[name][:, ix], axis=1))
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    out = {}
    for name in names:
        out[name] = {}
        for r, ix in idx.items():
            bt = np.concatenate(boot[name][r])
            out[name][r] = {"value": float(np.nanmean(obs[name][ix])), "lo": float(np.quantile(bt, lo)),
                            "hi": float(np.quantile(bt, hi)), "n_cols": int(ix.size)}
    return out


# ---------------------------------------------------------------------------
# verbs
# ---------------------------------------------------------------------------

def cmd_spaces(args) -> None:
    try:
        print("\n".join(h.common_spaces(args.arm)))
    except ValueError as exc:
        raise SystemExit(str(exc))


def cmd_read(args) -> None:
    import families as fam
    import grayordinates as go
    from braintwill.encoding import ScoreCache, resampled_r2

    import cha

    try:
        groups = list(h.check_arm(args.arm))
        spaces = h.common_spaces(args.arm)
    except ValueError as exc:
        raise SystemExit(str(exc))
    weights = term_weights(groups)
    paths = h.Paths()
    cache = ScoreCache(paths.out)
    table = pd.read_csv(go.grayordinates_path(paths.cleaned), sep="\t")
    labels = np.asarray(fam.family_labels(table, cha.Paths().atlases)[rs.RULE["family"]])
    subs = args.subs.split(",") if args.subs else list(h.fm.SUBJECTS)
    rel = paths.out / "reliability" / f"register-{args.register}"
    per, keys, checks = {}, {}, {}
    for sub in subs:
        mask_path = rel / f"sub-{sub}.npz"
        if not mask_path.exists():
            raise SystemExit(f"{mask_path} is missing; run reliability.py for this register")
        with np.load(mask_path) as z:
            mask = z["mask"]
        got, valid = {}, None
        keys[f"sub-{sub}"], checks[f"sub-{sub}"] = {}, {}
        for space in spaces:
            k, s = rs._cached(paths, cache, args.register, space, sub)
            subset = space.split(".", 1)[1]
            keys[f"sub-{sub}"][subset] = {"space": space, "features": k["feature_hash"], "spec": k["spec_hash"]}
            if valid is None:
                valid = s.valid
            elif not np.array_equal(s.valid, valid):
                raise RuntimeError(f"sub-{sub} {space}: valid columns differ from the first subset's")
            back = resampled_r2(rs._restrict(s.groups, np.flatnonzero(s.valid)), np.ones(len(s.groups["labels"])))
            err = float(np.nanmax(np.abs(back - s.r2[s.valid])))
            checks[f"sub-{sub}"][subset] = err
            if err > 1e-4:
                raise RuntimeError(f"sub-{sub} {space}: per-group sums miss the stored R² by {err:.2g}")
            got[subset] = s.groups
        rois = rs.roi_columns(labels, mask, valid, rs.RULE["min_cols"])
        rois[rs.ALL_ROI] = np.flatnonzero(np.asarray(mask, bool) & np.asarray(valid, bool))
        counts = rs.draw_counts(len(got[spaces[0].split(".", 1)[1]]["labels"]), rs.RULE["n_boot"], rs.RULE["seed"])
        per[sub] = term_stats(got, rois, counts, weights)
        print(f"sub-{sub}: {len(rois)} ROIs, {len(weights)} terms from {len(spaces)} fits", flush=True)
    rows = [{"term": t, "sub": sub, "roi": roi, **x}
            for sub, d in per.items() for t, rs_ in d.items() for roi, x in rs_.items()]
    df = pd.DataFrame(rows)
    med = (df.groupby(["term", "roi"])["value"].agg(["median", "count"]).reset_index()
           .rename(columns={"median": "median_over_subjects", "count": "n_subjects"}))
    out = {"arm": args.arm, "groups": {g: list(v) for g, v in h.COMMON_ARMS[args.arm].items()},
           "development_arm": args.arm.startswith("dev"),
           "register": args.register, "subjects": subs, "terms": weights,
           "read": {"family": rs.RULE["family"], "min_cols": rs.RULE["min_cols"], "n_boot": rs.RULE["n_boot"],
                    "seed": rs.RULE["seed"], "interval": f"{CI:.0%} percentile bootstrap over CV groups, "
                    "one draw of counts per subject shared by every subset fit (paired)",
                    "whole_mask_roi": rs.ALL_ROI, "verdict": "none: descriptive",
                    "negative_terms": "reported as computed; not evidence of suppression"},
           "per_subject": per, "keys": keys, "group_sums_check_max_abs": checks,
           "provenance": h.provenance("commonality", {}),
           "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    stem = dest / f"commonality_{args.arm}_{args.register}"
    stem.with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n")
    df.to_csv(stem.with_suffix(".csv"), index=False)
    med.to_csv(dest / f"commonality_{args.arm}_{args.register}_median.csv", index=False)
    print(f"wrote {stem}.json, {stem}.csv, {stem}_median.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    p = sub.add_parser("spaces")
    p.add_argument("--arm", required=True)
    r = sub.add_parser("read")
    r.add_argument("--arm", required=True)
    r.add_argument("--register", choices=("films",), default="films")
    r.add_argument("--out", required=True)
    r.add_argument("--subs", help="comma-separated subjects (default: the films pool's)")
    args = ap.parse_args()
    {"spaces": cmd_spaces, "read": cmd_read}[args.verb](args)


if __name__ == "__main__":
    main()
