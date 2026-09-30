#!/usr/bin/env python3
"""Scoring for the functional-space study: metrics M1–M4, gains, and the H1/H2 decision rule.

Pre-registration §9; M2b and M4 details DECIDED 2026-09-30 in mmmdata-agents
``docs/workbench/functional-space/``. **Before the freeze this module runs on
synthetic data only** (§11.6): nothing here reads the held-out sessions, TB
betas or localizer maps, and the ``selftest`` verb plants its own data.

Space. Every score is computed in the target's space (§9): a template
subject's data go into the template through its own transform, then into the
target through the transpose of the target's (``project``). The template side
of every metric is the mean over the template subjects.

Metrics (per network; a network is a set of grayordinate columns):

  M1   localizer map prediction: Pearson over the network's valid vertices
       (``m1_map``, as ``localizer_ceiling.py``); pRF polar angle by circular
       correlation within each hemisphere, vertex-weighted (``m1_angle``). A
       pRF map is projected as its Cartesian components (ecc·cos, ecc·sin),
       never as an angle: an orthogonal map mixes columns linearly, which is
       meaningless for a circular quantity.
  M2a  film ISC: per-column temporal r on a held-out film, averaged over the
       network's columns (``m2a``).
  M2b  film segment identification (primary): non-overlapping 10-TR segments
       of each film (remainder dropped), spatiotemporal patterns (10 TRs ×
       columns, flattened), each target segment ranked by correlation against
       every held-out segment of the template side; score = rank accuracy
       (chance .5), top-1 reported alongside (``m2b``).
  M3   TB item identification: spatial patterns per item, same rank accuracy
       (``m3``).
  M4   M2a on residuals: each subject's own encoder prediction removed from
       its own held-out series before projection (``residualize``, then
       ``m2a``).

Inference (§9, the H1/H2 rule; the same functions as ``h1_power_sim.py``):
per-film gains (route − reference, film means of the segment scores,
averaged over draws), one-sided exact sign-flip p per target × network, Holm
across networks within a target, go if a network survives in ≥ 2 targets.
The anatomical reference per (target, network) is the stronger of the MNI and
fsaverage6 baselines (``reference_scores``).

Verbs:

  selftest  planted synthetic data at real sizes: checks every metric ranks
            a planted alignment above anatomy, times each metric, and writes
            <derivatives>/functional_space/dryrun/scoring_selftest.json. No
            dataset file is read.

Usage:
    python scoring.py selftest [--n-columns 82835] [--out <json>]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT / "src" / "python", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import h1_power_sim as h1  # noqa: E402
import procrustes as pr  # noqa: E402

SEGMENT_TRS = 10  # 15 s at TR 1.5 s (§9; DECIDED 2026-09-27)


# ---------------------------------------------------------------------------
# space
# ---------------------------------------------------------------------------

def project(data: dict[str, np.ndarray], transforms: dict[str, pr.PiecewiseTransform], target: str
            ) -> np.ndarray:
    """Mean over the template subjects of their data carried into the target's space.

    ``data`` holds the template subjects' rows (same rows for each);
    ``transforms`` every subject's transform into the template. Columns
    outside the target's transform come out NaN.
    """
    back = transforms[target].inverse()
    subs = [s for s in data if s != target]
    if not subs:
        raise ValueError("no template subject to project")
    return np.mean(np.stack([back.apply(transforms[s].apply(data[s])) for s in subs]), axis=0)


def zscore_columns(y: np.ndarray) -> np.ndarray:
    """Per-column z-score; a constant or non-finite column is NaN (never filled)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (y - y.mean(axis=0)) / y.std(axis=0)


def residualize(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """M4: a subject's held-out series minus its own encoder's prediction of it."""
    if y.shape != pred.shape:
        raise ValueError(f"series {y.shape} and prediction {pred.shape} differ")
    return y - pred


def network_columns(networks: np.ndarray, names=None) -> dict[str, np.ndarray]:
    """``{network: columns}`` from a per-column label array ('' or None = no network)."""
    networks = np.asarray(networks, dtype=object)
    names = names or sorted({n for n in networks if n not in ("", None)})
    return {n: np.flatnonzero(networks == n) for n in names}


def _valid(*arrays: np.ndarray) -> np.ndarray:
    ok = np.ones(arrays[0].shape[-1], dtype=bool)
    for a in arrays:
        ok &= np.isfinite(a).reshape(-1, a.shape[-1]).all(axis=0)
    return ok


# ---------------------------------------------------------------------------
# M2a / M4
# ---------------------------------------------------------------------------

def column_r(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-column Pearson r over rows."""
    pc = pred - pred.mean(axis=0)
    yc = y - y.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (pc * yc).sum(axis=0) / np.sqrt((pc ** 2).sum(axis=0) * (yc ** 2).sum(axis=0))


def m2a(target: np.ndarray, template: np.ndarray, nets: dict[str, np.ndarray]) -> dict[str, float]:
    """Film ISC per network: mean over valid columns of the per-column temporal r."""
    r = column_r(template, target)
    return {n: float(np.nanmean(r[c])) if np.isfinite(r[c]).any() else np.nan for n, c in nets.items()}


# ---------------------------------------------------------------------------
# identification (M2b, M3)
# ---------------------------------------------------------------------------

def identification(target: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row i of ``target`` against every row of ``template``: (rank accuracy, top-1 hit) per row.

    Rank accuracy = share of foils (j ≠ i) whose correlation is below the true
    one's (chance .5; a tie counts half). Patterns are rows; correlation is
    Pearson over their entries.
    """
    if target.shape != template.shape:
        raise ValueError(f"target {target.shape} and template {template.shape} differ")
    n = target.shape[0]
    if n < 2:
        raise ValueError("identification needs at least two items")
    a = zscore_columns(target.T).T
    b = zscore_columns(template.T).T
    c = a @ b.T / a.shape[1]
    true = np.diag(c)[:, None]
    foil = ~np.eye(n, dtype=bool)
    rank = (((c < true) & foil).sum(1) + 0.5 * ((c == true) & foil).sum(1)) / (n - 1)
    top1 = c.argmax(axis=1) == np.arange(n)
    return rank, top1


def segment_slices(n_rows: int, length: int = SEGMENT_TRS) -> list[slice]:
    """Consecutive non-overlapping segments; a remainder shorter than ``length`` is dropped."""
    return [slice(i, i + length) for i in range(0, n_rows - length + 1, length)]


def m2b(target_films: dict[str, np.ndarray], template_films: dict[str, np.ndarray],
        nets: dict[str, np.ndarray], length: int = SEGMENT_TRS) -> pd.DataFrame:
    """Segment identification over all films together, per network.

    ``*_films`` map film id to that film's paired rows (target grid as
    reference), the same rows on both sides. Each film is z-scored per column
    before segmenting. Columns non-finite on either side in any film are left
    out of that network. Returns one row per (network, film, segment).
    """
    if target_films.keys() != template_films.keys():
        raise ValueError("target and template cover different films")
    tz = {f: zscore_columns(x) for f, x in target_films.items()}
    pz = {f: zscore_columns(x) for f, x in template_films.items()}
    segs = [(f, k, sl) for f in tz for k, sl in enumerate(segment_slices(tz[f].shape[0], length))]
    rows = []
    for net, cols in nets.items():
        ok = _valid(*(tz[f][:, cols] for f in tz), *(pz[f][:, cols] for f in pz))
        c = cols[ok]
        if c.size == 0 or len(segs) < 2:
            continue
        a = np.stack([tz[f][sl][:, c].ravel() for f, _, sl in segs])
        b = np.stack([pz[f][sl][:, c].ravel() for f, _, sl in segs])
        rank, top1 = identification(a, b)
        rows += [{"network": net, "film": f, "segment": k, "rank_acc": float(r), "top1": bool(t),
                  "n_columns": int(c.size), "n_segments": len(segs)}
                 for (f, k, _), r, t in zip(segs, rank, top1)]
    return pd.DataFrame(rows)


def per_film(scores: pd.DataFrame, value: str = "rank_acc") -> pd.DataFrame:
    """Film means of segment scores: the unit the inference uses (network × film)."""
    return scores.groupby(["network", "film"], as_index=False)[value].mean()


def m3(target_items: np.ndarray, template_items: np.ndarray, nets: dict[str, np.ndarray]) -> pd.DataFrame:
    """TB item identification per network: rank accuracy per item (and top-1)."""
    rows = []
    for net, cols in nets.items():
        ok = _valid(target_items[:, cols], template_items[:, cols])
        c = cols[ok]
        if c.size < 2:
            continue
        rank, top1 = identification(target_items[:, c], template_items[:, c])
        rows.append({"network": net, "rank_acc": float(rank.mean()), "top1": float(top1.mean()),
                     "n_items": int(rank.size), "n_columns": int(c.size)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# M1
# ---------------------------------------------------------------------------

def m1_map(target_map: np.ndarray, template_map: np.ndarray, nets: dict[str, np.ndarray]) -> dict[str, float]:
    """Localizer contrast map prediction: Pearson over each network's valid vertices."""
    out = {}
    for net, cols in nets.items():
        a, b = target_map[cols], template_map[cols]
        ok = np.isfinite(a) & np.isfinite(b)
        out[net] = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan
    return out


def circ_corr(a_deg: np.ndarray, b_deg: np.ndarray) -> float:
    """Circular correlation (Jammalamadaka & SenGupta 2001), as ``localizer_ceiling.circ_corr``."""
    a, b = np.radians(a_deg), np.radians(b_deg)
    sa = np.sin(a - np.arctan2(np.sin(a).mean(), np.cos(a).mean()))
    sb = np.sin(b - np.arctan2(np.sin(b).mean(), np.cos(b).mean()))
    return float((sa * sb).sum() / np.sqrt((sa ** 2).sum() * (sb ** 2).sum()))


def cartesian(angle_deg: np.ndarray, ecc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pRF centre as (x, y): the form a pRF map is projected in."""
    a = np.radians(angle_deg)
    return ecc * np.cos(a), ecc * np.sin(a)


def m1_angle(target_xy: tuple[np.ndarray, np.ndarray], template_xy: tuple[np.ndarray, np.ndarray],
             hemi: np.ndarray, keep: np.ndarray, nets: dict[str, np.ndarray]) -> dict[str, float]:
    """pRF polar angle prediction: circular r within each hemisphere, vertex-weighted over the two.

    ``keep`` marks the vertices the pre-registration scores (target R² floor,
    within the stimulus radius). Angles come from the (projected) Cartesian
    components.
    """
    ta = np.degrees(np.arctan2(target_xy[1], target_xy[0]))
    pa = np.degrees(np.arctan2(template_xy[1], template_xy[0]))
    out = {}
    for net, cols in nets.items():
        num = den = 0.0
        for h in np.unique(hemi[cols]):
            c = cols[(hemi[cols] == h) & keep[cols] & np.isfinite(ta[cols]) & np.isfinite(pa[cols])]
            if c.size > 2:
                num += c.size * circ_corr(ta[c], pa[c])
                den += c.size
        out[net] = num / den if den else np.nan
    return out


# ---------------------------------------------------------------------------
# gains and the decision rule
# ---------------------------------------------------------------------------

def reference_scores(baselines: dict[str, pd.DataFrame], value: str = "rank_acc") -> pd.DataFrame:
    """Per (target, network), the baseline with the higher mean over films (§9's conservative reference).

    Each frame has columns target, network, film, ``value``. Returns the
    chosen baseline's rows with a ``baseline`` column.
    """
    means = pd.concat([df.groupby(["target", "network"])[value].mean().rename(name)
                       for name, df in baselines.items()], axis=1)
    pick = means.idxmax(axis=1).rename("baseline").reset_index()
    rows = []
    for r in pick.itertuples(index=False):
        df = baselines[r.baseline]
        rows.append(df[(df["target"] == r.target) & (df["network"] == r.network)].assign(baseline=r.baseline))
    return pd.concat(rows, ignore_index=True)


def film_gains(route: pd.DataFrame, reference: pd.DataFrame, value: str = "rank_acc") -> pd.DataFrame:
    """Per (target, network, film): route − reference, each already averaged over segments and draws."""
    key = ["target", "network", "film"]
    m = route[key + [value]].merge(reference[key + [value]], on=key, suffixes=("_route", "_ref"), validate="1:1")
    if len(m) != len(route) or len(m) != len(reference):
        raise ValueError("route and reference do not cover the same (target, network, film) cells")
    return m.assign(gain=m[f"{value}_route"] - m[f"{value}_ref"])[key + ["gain"]]


def decide(gains: pd.DataFrame, alpha: float = 0.05, min_targets: int = 2) -> dict:
    """The §9 rule on per-film gains: exact one-sided sign-flip p, Holm within target, go if a
    network survives in ≥ ``min_targets`` targets."""
    targets = sorted(gains["target"].unique())
    networks = sorted(gains["network"].unique())
    films = sorted(gains["film"].unique())
    cube = (gains.set_index(["target", "network", "film"])["gain"]
            .reindex(pd.MultiIndex.from_product([targets, networks, films])).to_numpy()
            .reshape(len(targets), len(networks), len(films)))
    if not np.isfinite(cube).all():
        raise ValueError("gains are missing for some (target, network, film) cells")
    signs = h1.sign_matrix(len(films))
    p = h1.signflip_p(cube.reshape(-1, len(films)), signs).reshape(len(targets), len(networks))
    reject = h1.holm_reject(p, alpha)
    counted = [n for j, n in enumerate(networks) if reject[:, j].sum() >= min_targets]
    return {"targets": targets, "networks": networks, "n_films": len(films),
            "p": {t: dict(zip(networks, map(float, p[i]))) for i, t in enumerate(targets)},
            "reject": {t: dict(zip(networks, map(bool, reject[i]))) for i, t in enumerate(targets)},
            "networks_counted": counted, "go": bool(counted), "alpha": alpha, "min_targets": min_targets}


# ---------------------------------------------------------------------------
# selftest (synthetic, real sizes)
# ---------------------------------------------------------------------------

def _planted_transforms(labels: np.ndarray, subs: list[str], rng) -> dict[str, pr.PiecewiseTransform]:
    n = labels.size
    tfs = {}
    for s in subs:
        tf = pr.PiecewiseTransform(n)
        for lab in np.unique(labels):
            cols = np.flatnonzero(labels == lab)
            q, _ = np.linalg.qr(rng.standard_normal((cols.size, cols.size)))
            tf.pieces[lab] = (cols, q)
        tfs[s] = tf
    return tfs


def selftest(n_columns: int, piece: int, n_films: int, film_trs: int, n_items: int, noise: float, seed: int = 0,
             log=print) -> dict:
    """Plant a shared signal in a common frame, give each subject its own per-piece rotation, and check
    that every metric scores the true alignment above anatomy (identity). Times each metric."""
    rng = np.random.default_rng(seed)
    subs, target = ["A", "B", "T"], "T"
    labels = np.arange(n_columns) // piece
    nets = network_columns(np.array([f"net{int(v) % 7}" for v in labels], dtype=object))
    tfs = _planted_transforms(labels, subs, rng)
    ident = {s: pr.PiecewiseTransform(n_columns, {lab: (c, np.eye(c.size)) for lab, (c, _) in tfs[s].pieces.items()})
             for s in subs}

    def subject_view(common, s):  # common frame -> subject s's columns: x R_s' (R_s maps s -> common)
        return tfs[s].inverse().apply(common) + noise * rng.standard_normal(common.shape)

    out, seconds = {}, {}
    films = {f"f{i:02d}": rng.standard_normal((film_trs, n_columns)).astype(np.float32) for i in range(n_films)}
    data = {s: {f: subject_view(x, s).astype(np.float32) for f, x in films.items()} for s in subs}
    for name, tf in (("aligned", tfs), ("anatomical", ident)):
        t0 = time.time()
        tpl = {f: project({s: data[s][f] for s in subs if s != target}, tf, target) for f in films}
        seconds[f"project_{name}"] = round(time.time() - t0, 2)
        t0 = time.time()
        s2a = [m2a(data[target][f], tpl[f], nets) for f in films]
        seconds[f"m2a_{name}"] = round(time.time() - t0, 2)
        t0 = time.time()
        s2b = m2b({f: data[target][f] for f in films}, tpl, nets)
        seconds[f"m2b_{name}"] = round(time.time() - t0, 2)
        out[name] = {"m2a": float(np.mean([np.nanmean(list(d.values())) for d in s2a])),
                     "m2b_rank": float(s2b["rank_acc"].mean()), "m2b_top1": float(s2b["top1"].mean()),
                     "n_segments": int(s2b["n_segments"].iat[0])}
    items = rng.standard_normal((n_items, n_columns)).astype(np.float32)
    it = {s: subject_view(items, s) for s in subs}
    for name, tf in (("aligned", tfs), ("anatomical", ident)):
        t0 = time.time()
        s3 = m3(it[target], project({s: it[s] for s in subs if s != target}, tf, target), nets)
        seconds[f"m3_{name}"] = round(time.time() - t0, 2)
        out[name]["m3_rank"] = float(s3["rank_acc"].mean())
    for name in ("aligned", "anatomical"):
        log(f"{name}: " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in out[name].items()))
    checks = {m: out["aligned"][m] > out["anatomical"][m] for m in ("m2a", "m2b_rank", "m3_rank")}
    return {"sizes": {"n_columns": n_columns, "piece": piece, "n_films": n_films, "film_trs": film_trs,
                      "n_items": n_items, "noise": noise, "seed": seed},
            "scores": out, "aligned_beats_anatomical": checks, "passed": all(checks.values()),
            "seconds": seconds}


def cmd_selftest(args: argparse.Namespace) -> None:
    t0 = time.time()
    res = selftest(args.n_columns, args.piece, args.n_films, args.film_trs, args.n_items, args.noise)
    if args.out:
        out = Path(args.out)
    else:
        from core.config import load_config

        out = Path(load_config()["paths"]["output_dir"]) / "functional_space" / "dryrun" / "scoring_selftest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.update({"description": "scoring.py selftest: planted synthetic data at real sizes; no dataset file read; "
                               "checks each metric ranks a planted alignment above identity, and times it",
                "code_version": _code_version(), "total_s": round(time.time() - t0, 1),
                "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")})
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"{'PASSED' if res['passed'] else 'FAILED'}; wrote {out}")
    if not res["passed"]:
        sys.exit(1)


def _code_version() -> str:
    from neuroimaging import data_quality as dq

    return dq.code_version(REPO_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--n-columns", type=int, default=82835)
    s.add_argument("--piece", type=int, default=200, help="columns per synthetic piece (real median 182)")
    s.add_argument("--n-films", type=int, default=12)
    s.add_argument("--film-trs", type=int, default=150)
    s.add_argument("--n-items", type=int, default=658)
    s.add_argument("--noise", type=float, default=3.0)
    s.add_argument("--out", default=None)
    args = ap.parse_args()
    {"selftest": cmd_selftest}[args.verb](args)


if __name__ == "__main__":
    main()
