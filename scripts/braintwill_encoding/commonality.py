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

Two engines fit the subsets (``--engine``). ``joint`` (the default): the ``common-`` spaces,
each subset its own banded-ridge fit, one kernel band per group, equal kernel trace per source
within a group. ``stack``: the ``stack-`` spaces, every source its own ridge and each subset a
convex per-column combination of its sources (braintwill ``encoding.stack``, Lin et al. 2024),
fitted by one ``harness.py stack`` job per subject; its out-of-fold route ``--oof`` has no
default. Stacked subsets score only the reliability mask's columns, which are the columns this
reader reads. The terms and the bootstrap are the same for both engines.

Timing and nuisance features belong in their own group. Otherwise variance a timing signal
explains through both the opaque and the named group is booked as "shared opaque ∩ named".

Verbs:

  spaces  the arm's 2^k - 1 harness spaces, one per line, for a submission loop (``--engine
          stack``: its stacked spaces, solo sources included, for ``harness.py plan``)
  read    per subject and ROI (HCP-MMP1 sections, ``families.familyD``, inside the
          register's reliability mask, as ``rescue.py``): every term's ROI mean with a
          percentile bootstrap interval over the CV groups (one draw of counts per subject,
          shared by every subset fit, so the terms are paired) -> JSON + CSV + a
          median-over-subjects CSV. Descriptive, no verdict. With ``--draws A:B`` (an arm whose
          one ``ctrl:<pool>.<set>`` source is a control set): the same read of each draw arm
          ``<arm>@dNNN``, and per subject, ROI and draw-dependent term the real value's place
          among the draws' (percentile, the draws' 95th percentile, a draw p) -> a ``draws``
          JSON entry + ``*_draws.csv``. The named share C(o∩n) / R²(o) is reported beside it.
          Every draw is bootstrapped on the real read's counts, so C(o∩n)'s excess over the
          draws' mean and over their 95th percentile are paired (SE, interval;
          ``*_excess.csv``, arrays in ``*_boot.npz``). Those cells are then pooled over ROIs x
          subjects (braintwill ``pooled_effects``: ROIs partially pooled, subjects fixed) ->
          a ``pooled`` JSON entry + ``*_pooled.csv``, with the ROIs the screen selects. Every
          draw's value per term and ROI is kept (``*_draw_values.csv``).

Usage:
    python commonality.py spaces --arm <arm>
    for s in $(python commonality.py spaces --arm <arm>); do
        sbatch --export=ALL,VERB=score,ARGS="--space $s --sub ##" harness.sbatch; done
    python commonality.py read --arm <arm> --register films --out <dir> [--subs ##,##]
    sbatch --export=ALL,VERB=stack,ARGS="--arm <arm> --sub ## --oof reuse" harness.sbatch
    python commonality.py read --arm <arm> --engine stack --oof reuse --register films --out <dir>
    sbatch --export=ALL,VERB=stack,ARGS="--arm <arm> --sub ## --oof reuse --draws 0:20" harness.sbatch
    python commonality.py read --arm <arm> --engine stack --oof reuse --draws 0:100 --out <dir>

A subset fit is cached by its feature content and its spec, which carries the merge (group
names and their bands). So arms that differ only in the opaque group's source, with the same
group names, share every subset that leaves the opaque group out and refit only those that
hold it. A draw arm is such an arm: it shares the subsets without its control set with the
base arm (checked by key on every read), so a draw costs only the subsets that hold the set.
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
#: How the subset fits were made (module docstring).
ENGINES = ("joint", "stack")


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


def draw_groups(arm: str) -> tuple[str, str]:
    """(the group holding the arm's control set, the other group): a draw read needs two groups."""
    decl = h.check_arm(arm)
    if len(decl) != 2:
        raise ValueError(f"arm {arm!r}: a draw read needs two groups (opaque, control set); it has {list(decl)}")
    drawn = h.check_arm(f"{arm}@d000")  # refuses an arm without exactly one control set
    ctrl = next(g for g in decl if decl[g] != drawn[g])
    return ctrl, next(g for g in decl if g != ctrl)


def draw_rank(value: float, draws: np.ndarray) -> dict:
    """``value``'s place among control draws: percentile (ties count half), the draws' median
    and 95th percentile (numpy's linear quantile), ``exceeds_q95`` (strictly above it), and
    ``p_draws`` = (1 + #draws >= value) / (n + 1)."""
    d = np.asarray(draws, np.float64)
    if d.size == 0 or not np.all(np.isfinite(d)) or not np.isfinite(value):
        raise ValueError(f"a draw rank needs finite values and >= 1 draw; got {value!r} and {d.size} draws")
    q95 = float(np.quantile(d, 0.95))
    return {"value": float(value), "percentile": float(100 * (np.sum(d < value) + 0.5 * np.sum(d == value)) / d.size),
            "draw_median": float(np.median(d)), "draw_q95": q95, "exceeds_q95": bool(value > q95),
            "p_draws": float((1 + np.sum(d >= value)) / (d.size + 1)), "n_draws": int(d.size)}


def draw_read(real: dict[str, dict[str, float]], draws: list[dict[str, dict[str, float]]],
              ctrl: str, other: str) -> dict[str, dict[str, dict]]:
    """Per draw-dependent term (every term holding ``ctrl``, and ``total``) and ROI: the real
    value ranked among the draws'. The shared term also carries the named share C(o∩n) / R²(o)
    of the real set and the draws' median and 95th-percentile shares (R²(o) is the same for every
    draw, so the share ranks as the shared term does; it is None where R²(o) <= 0)."""
    from braintwill.encoding.resample import TERM_SEP

    shared = f"{other}{TERM_SEP}{ctrl}" if f"{other}{TERM_SEP}{ctrl}" in real else f"{ctrl}{TERM_SEP}{other}"
    terms = [t for t in real if t == "total" or ctrl in t.split(TERM_SEP)]
    out = {}
    for t in terms:
        out[t] = {}
        for roi, v in real[t].items():
            x = draw_rank(v, np.array([d[t][roi] for d in draws]))
            if t == shared:
                r2_o = real[other][roi] + real[shared][roi]
                ok = r2_o > 0
                x.update({"r2_other": r2_o, "share": v / r2_o if ok else None,
                          "share_draw_median": x["draw_median"] / r2_o if ok else None,
                          "share_draw_q95": x["draw_q95"] / r2_o if ok else None})
            out[t][roi] = x
    return out


#: Interval level of the pooled read over ROIs x subjects (a screen for the confirmation
#: addendum, model-readout prereg R17), and its CSV/JSON label.
POOL_CI = 0.90


def excess_stats(real: float, real_boot: np.ndarray, draws: np.ndarray, draws_boot: np.ndarray,
                 ci: float = CI) -> dict[str, dict]:
    """The real value's excess over the control draws, paired: over their mean and over their
    95th percentile (numpy linear), each with its bootstrap SE and percentile interval.

    ``draws`` (n,) are the draws' observed values and ``draws_boot`` (n, B) their values on the
    same B bootstrap draws as ``real_boot`` (B,), so each excess is formed draw by draw."""
    draws_boot = np.asarray(draws_boot, np.float64)
    real_boot = np.asarray(real_boot, np.float64)
    if draws_boot.shape != (np.size(draws), real_boot.size):
        raise ValueError(f"draws_boot {draws_boot.shape} does not match {np.size(draws)} draws x {real_boot.size}")
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    out = {}
    for name, stat in (("mean", lambda a, ax: np.mean(a, axis=ax)), ("q95", lambda a, ax: np.quantile(a, 0.95, axis=ax))):
        boot = real_boot - stat(draws_boot, 0)
        out[name] = {"value": float(real - stat(np.asarray(draws, np.float64), 0)), "se": float(np.std(boot, ddof=1)),
                     "lo": float(np.quantile(boot, lo)), "hi": float(np.quantile(boot, hi))}
    return out


def pooled_read(excess: dict[str, dict[str, dict]], ci: float = POOL_CI) -> dict:
    """Partial pooling of the per-cell excess over ROIs x subjects (braintwill
    ``pooled_effects``: ROIs pooled, subjects fixed), for each excess kind. The whole-mask ROI
    overlaps every section, so it is not a cell. ``selected``: ROIs whose pooled excess over the
    draws' 95th percentile has its ``ci`` interval above 0 (the screen, prereg R17)."""
    from braintwill.encoding import pooled_effects

    cells = [(sub, roi, x) for sub, by in excess.items() for roi, x in by.items() if roi != rs.ALL_ROI]
    out = {}
    for kind in ("mean", "q95"):
        out[f"excess_{kind}"] = pooled_effects([x[kind]["value"] for _, _, x in cells],
                                               [x[kind]["se"] for _, _, x in cells],
                                               [roi for _, roi, _ in cells], [sub for sub, _, _ in cells], ci=ci)
    out["selected"] = sorted(r for r, v in out["excess_q95"]["units"].items() if v["lo"] > 0)
    out["rule"] = (f"selected: pooled excess of C(o∩n) over the draws' 95th percentile, {ci:.0%} interval "
                   "above 0 (a screen for the confirmation addendum; no multiplicity correction: R17)")
    return out


# ---------------------------------------------------------------------------
# verbs
# ---------------------------------------------------------------------------

def engine_spaces(arm: str, engine: str, solo: bool = False) -> list[str]:
    """The subset spaces an engine reads (``stack`` adds its solo sources with ``solo``)."""
    if engine == "stack":
        return h.stack_spaces(arm, solo=solo)
    return h.common_spaces(arm)


def _output_stem(arm: str, register: str, engine: str, oof: str | None) -> str:
    tag = "" if engine == "joint" else f"_stack-{oof}"
    return f"commonality_{arm}_{register}{tag}"


def cmd_spaces(args) -> None:
    try:
        print("\n".join(engine_spaces(args.arm, args.engine, solo=True)))
    except ValueError as exc:
        raise SystemExit(str(exc))


def load_subsets(paths, cache, register: str, spaces: list[str], sub: str, oof: str | None,
                 engine: str) -> tuple[dict, np.ndarray, dict, dict, dict]:
    """One subject's cached subset scores, checked: (per-group sums by subset, valid columns,
    keys by subset, group-sums check by subset, stacking summary by subset (``stack`` only))."""
    from braintwill.encoding import resampled_r2

    got, valid, keys, checks, stacking = {}, None, {}, {}, {}
    for space in spaces:
        k, s = rs._cached(paths, cache, register, space, sub, oof=oof)
        subset = space.split(".", 1)[1]
        keys[subset] = {"space": space, "features": k["feature_hash"], "spec": k["spec_hash"]}
        if valid is None:
            valid = s.valid
        elif not np.array_equal(s.valid, valid):
            raise RuntimeError(f"sub-{sub} {space}: valid columns differ from the first subset's")
        back = resampled_r2(rs._restrict(s.groups, np.flatnonzero(s.valid)), np.ones(len(s.groups["labels"])))
        err = float(np.nanmax(np.abs(back - s.r2[s.valid])))
        checks[subset] = err
        if err > 1e-4:
            raise RuntimeError(f"sub-{sub} {space}: per-group sums miss the stored R² by {err:.2g}")
        got[subset] = s.groups
        if engine == "stack":
            stacking[subset] = {"weight_median": s.full_fit["weight_median"], "qp_max_gap": s.summary["qp_max_gap"]}
    return got, valid, keys, checks, stacking


def read_draws(paths, cache, args, sub: str, draws: range, ctrl: str, other: str, got: dict,
               valid: np.ndarray, keys: dict, rois: dict, weights: dict, counts: np.ndarray) -> dict:
    """Every draw arm's terms for one subject, ranked against the real arm's (``draw_read``),
    and the shared term's paired excess over the draws (``excess_stats``).

    Every draw is bootstrapped on the real read's ``counts``, so the excess is paired. A draw
    arm's subsets without the control set hold the base arm's sources, so they are the base
    arm's units (a key is its sources' content, the spec and the register) and are reused, not
    reloaded. Each draw's own subsets must score the base arm's valid columns and carry
    features other than the real set's. ``_boot`` (popped before the JSON) holds the bootstrap
    arrays the reader writes to ``*_boot.npz``."""
    from braintwill.encoding import combination_draws
    from braintwill.encoding.resample import TERM_SEP

    base = h.check_arm(args.arm)
    shared = TERM_SEP.join(base)
    terms = {t: w for t, w in {**weights, "total": {"+".join(base): 1}}.items()  # total, as term_stats reads it
             if t == "total" or ctrl in t.split(TERM_SEP)}
    terms[other] = weights[other]  # the share's denominator needs C(other)
    real, real_boot = combination_draws(got, rois, counts, terms, chunk=rs.DRAW_CHUNK)
    vals, shared_boot, qp = [], {r: [] for r in rois}, []
    for i in draws:
        arm = f"{args.arm}@d{i:03d}"
        decl = h.check_arm(arm)
        if any(decl[g] != base[g] for g in decl if g != ctrl):
            raise RuntimeError(f"{arm}: a group other than {ctrl!r} differs from the base arm's")
        mine = [s for s in engine_spaces(arm, args.engine) if ctrl in s.split(".", 1)[1].split("+")]
        g, v, k, _, st = load_subsets(paths, cache, args.register, mine, sub, args.oof, args.engine)
        if not np.array_equal(v, valid):
            raise RuntimeError(f"sub-{sub} {arm}: valid columns differ from the base arm's")
        same = [x for x in k if k[x]["features"] == keys[x]["features"]]
        if same:
            raise RuntimeError(f"sub-{sub} {arm}: subsets {same} have the base arm's features, not the draw's")
        obs, boot = combination_draws({**got, **g}, rois, counts, terms, chunk=rs.DRAW_CHUNK)
        vals.append(obs)
        for r in rois:
            shared_boot[r].append(boot[shared][r])
        qp += [x["qp_max_gap"] for x in st.values()]
    excess, ex_boot = {}, {}
    for r in rois:
        db = np.stack(shared_boot[r])
        dv = np.array([d[shared][r] for d in vals])
        excess[r] = excess_stats(real[shared][r], real_boot[shared][r], dv, db)
        ex_boot[r] = {"real": real_boot[shared][r], "mean": real_boot[shared][r] - db.mean(0),
                      "q95": real_boot[shared][r] - np.quantile(db, 0.95, axis=0)}
    return {"read": draw_read(real, vals, ctrl, other), "excess": excess,
            "values": {t: {r: [d[t][r] for d in vals] for r in rois} for t in terms if t != other},
            **({"qp_max_gap_max": float(max(qp))} if qp else {}), "_boot": ex_boot}


def draws_entry(args, paths, draws: range, ctrl: str, other: str, drawn: dict) -> dict:
    """The JSON ``draws`` entry: what was drawn, how it was read, and the per-subject reads."""
    src = h.check_arm(args.arm)[ctrl]
    pool, name = h.CTRL_SOURCE.fullmatch(next(s for s in src if s.startswith("ctrl:"))).group(1, 2)
    cs = h.ctrl_set(paths.controls, pool, name)
    with np.load(cs["path"]) as z:
        target = float(z["share"]) if "share" in z else None
    return {"range": [draws.start, draws.stop], "n": len(draws), "control_group": ctrl, "other_group": other,
            "pool": pool, "set": name, "k": cs["k"], "target_share": target, "set_file": str(cs["path"]),
            "rank": "percentile (ties half) of the real term among the draws; draw_q95 numpy linear quantile; "
                    "exceeds_q95 strict; p_draws = (1 + #draws >= real) / (n + 1)",
            "share": "C(other & control) / R²(other), R²(other) = C(other) + C(other & control): the same "
                     "for every draw, so the share ranks as the shared term; None where R²(other) <= 0",
            "per_subject": drawn}


def cmd_read(args) -> None:
    import families as fam
    import grayordinates as go
    from braintwill.encoding import ScoreCache

    import cha

    if args.engine == "stack" and args.oof is None:
        raise SystemExit(f"--engine stack needs --oof, one of {h.STACK_OOF} (no default: model-readout prereg R2)")
    try:
        groups = list(h.check_arm(args.arm))
        spaces = engine_spaces(args.arm, args.engine)
        draws, ctrl, other = None, None, None
        if args.draws is not None:
            if h.DRAW_ARM.fullmatch(args.arm):
                raise ValueError(f"--draws takes the base arm, not a draw arm ({args.arm!r})")
            draws = h.draw_range(args.draws)
            ctrl, other = draw_groups(args.arm)
    except ValueError as exc:
        raise SystemExit(str(exc))
    weights = commonality_weights(groups)
    paths = h.Paths()
    cache = ScoreCache(paths.out)
    table = pd.read_csv(go.grayordinates_path(paths.cleaned), sep="\t")
    labels = np.asarray(fam.family_labels(table, cha.Paths().atlases)[rs.RULE["family"]])
    subs = args.subs.split(",") if args.subs else list(h.fm.SUBJECTS)
    rel = paths.out / "reliability" / f"register-{args.register}"
    per, keys, checks, stacking, drawn, boots = {}, {}, {}, {}, {}, {}
    for sub in subs:
        mask_path = rel / f"sub-{sub}.npz"
        if not mask_path.exists():
            raise SystemExit(f"{mask_path} is missing; run reliability.py for this register")
        with np.load(mask_path) as z:
            mask = z["mask"]
        got, valid, keys[f"sub-{sub}"], checks[f"sub-{sub}"], st = load_subsets(
            paths, cache, args.register, spaces, sub, args.oof, args.engine)
        if args.engine == "stack":
            stacking[f"sub-{sub}"] = st
        rois = rs.roi_columns(labels, mask, valid, rs.RULE["min_cols"])
        rois[rs.ALL_ROI] = np.flatnonzero(np.asarray(mask, bool) & np.asarray(valid, bool))
        counts = draw_counts(len(got[spaces[0].split(".", 1)[1]]["labels"]), rs.RULE["n_boot"], rs.RULE["seed"])
        per[sub] = term_stats(got, rois, counts, weights)
        print(f"sub-{sub}: {len(rois)} ROIs, {len(weights)} terms from {len(spaces)} fits", flush=True)
        if draws is not None:
            drawn[sub] = read_draws(paths, cache, args, sub, draws, ctrl, other, got, valid,
                                    keys[f"sub-{sub}"], rois, weights, counts)
            boots[sub] = drawn[sub].pop("_boot")
            print(f"sub-{sub}: {len(draws)} control draws read", flush=True)
    rows = [{"term": t, "sub": sub, "roi": roi, **x}
            for sub, d in per.items() for t, rs_ in d.items() for roi, x in rs_.items()]
    df = pd.DataFrame(rows)
    med = (df.groupby(["term", "roi"])["value"].agg(["median", "count"]).reset_index()
           .rename(columns={"median": "median_over_subjects", "count": "n_subjects"}))
    out = {"arm": args.arm, "engine": args.engine, "oof": args.oof,
           "groups": {g: list(v) for g, v in h.COMMON_ARMS[args.arm].items()},
           "development_arm": args.arm.startswith("dev"),
           "register": args.register, "subjects": subs, "terms": weights,
           "read": {"family": rs.RULE["family"], "min_cols": rs.RULE["min_cols"], "n_boot": rs.RULE["n_boot"],
                    "seed": rs.RULE["seed"], "interval": f"{CI:.0%} percentile bootstrap over CV groups, "
                    "one draw of counts per subject shared by every subset fit (paired)",
                    "whole_mask_roi": rs.ALL_ROI, "verdict": "none: descriptive",
                    "negative_terms": "reported as computed; not evidence of suppression"},
           "per_subject": per, "keys": keys, "group_sums_check_max_abs": checks,
           **({"stacking": stacking} if args.engine == "stack" else {}),
           **({"draws": draws_entry(args, paths, draws, ctrl, other, drawn),
               "pooled": pooled_read({s: d["excess"] for s, d in drawn.items()})} if draws is not None else {}),
           "provenance": h.provenance("commonality", {}),
           "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    stem = dest / _output_stem(args.arm, args.register, args.engine, args.oof)
    stem.with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n")
    df.to_csv(stem.with_suffix(".csv"), index=False)
    med.to_csv(dest / f"{stem.name}_median.csv", index=False)
    print(f"wrote {stem}.json, {stem}.csv, {stem}_median.csv")
    if draws is not None:
        dr = pd.DataFrame([{"term": t, "sub": sub, "roi": roi, **x}
                           for sub, d in drawn.items() for t, by in d["read"].items() for roi, x in by.items()])
        dr.to_csv(dest / f"{stem.name}_draws.csv", index=False)
        pd.DataFrame([{"term": t, "sub": sub, "roi": roi, "draw": draws.start + j, "value": v}
                      for sub, d in drawn.items() for t, by in d["values"].items() for roi, vs in by.items()
                      for j, v in enumerate(vs)]).to_csv(dest / f"{stem.name}_draw_values.csv", index=False)
        pd.DataFrame([{"sub": sub, "roi": roi, "excess_over": kind, **y}
                      for sub, d in drawn.items() for roi, x in d["excess"].items() for kind, y in x.items()]
                     ).to_csv(dest / f"{stem.name}_excess.csv", index=False)
        pl = out["pooled"]
        pd.DataFrame([{"excess_over": k[len("excess_"):], "roi": roi, **v, "selected": k == "excess_q95" and roi in pl["selected"]}
                      for k in ("excess_mean", "excess_q95") for roi, v in pl[k]["units"].items()]
                     ).to_csv(dest / f"{stem.name}_pooled.csv", index=False)
        np.savez_compressed(dest / f"{stem.name}_boot.npz", **{f"sub-{sub}|{roi}|{kind}": a
                                                              for sub, by in boots.items() for roi, x in by.items()
                                                              for kind, a in x.items()})
        print(f"wrote {stem}_draws.csv, _draw_values.csv, _excess.csv, _pooled.csv, _boot.npz ({len(draws)} draws; "
              f"{len(pl['selected'])} ROIs selected)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    p = sub.add_parser("spaces")
    p.add_argument("--arm", required=True)
    p.add_argument("--engine", choices=ENGINES, default="joint")
    r = sub.add_parser("read")
    r.add_argument("--arm", required=True)
    r.add_argument("--engine", choices=ENGINES, default="joint")
    r.add_argument("--oof", choices=h.STACK_OOF, help="--engine stack: the stacking route (required)")
    r.add_argument("--register", choices=("films",), default="films")
    r.add_argument("--out", required=True)
    r.add_argument("--subs", help="comma-separated subjects (default: the films pool's)")
    r.add_argument("--draws", help="A:B: also read control draws A … B-1 of the arm's one control set")
    args = ap.parse_args()
    {"spaces": cmd_spaces, "read": cmd_read}[args.verb](args)


if __name__ == "__main__":
    main()
