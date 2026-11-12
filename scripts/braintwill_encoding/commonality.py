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

The weights and the paired bootstrap are braintwill's (``braintwill.encoding.resample``:
``commonality_weights``, ``combination_stats``); this driver resolves arms, spaces, masks
and ROIs. The terms sum to R²(all) exactly. They are partitioned, not residualised: no group is
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
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import harness as h  # noqa: E402  (puts functional_space and src/python on sys.path)
import rescue as rs  # noqa: E402  ROI columns, cached-score reader
from braintwill.encoding import combination_stats, commonality_weights, draw_counts  # noqa: E402

#: Interval level of the percentile bootstrap intervals.
CI = 0.95


def term_stats(groups_by_subset: dict[str, dict], rois: dict[str, np.ndarray], counts: np.ndarray,
               weights: dict[str, dict[str, int]], ci: float = CI) -> dict:
    """Every commonality term per ROI, plus ``total`` (the fit holding every group).

    ``weights`` from braintwill's ``commonality_weights`` (fit names ``a+b`` match the arm's
    subset spaces); the read is braintwill's ``combination_stats``: per column, then the ROI
    mean, one paired bootstrap over the CV groups for every term.
    """
    if not rois:
        return {}
    full = max({s for w in weights.values() for s in w}, key=lambda s: len(s.split("+")))
    return combination_stats(groups_by_subset, rois, counts, {**weights, "total": {full: 1}},
                             ci=ci, chunk=rs.DRAW_CHUNK)


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
    weights = commonality_weights(groups)
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
        counts = draw_counts(len(got[spaces[0].split(".", 1)[1]]["labels"]), rs.RULE["n_boot"], rs.RULE["seed"])
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
