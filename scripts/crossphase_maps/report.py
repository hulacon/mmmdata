#!/usr/bin/env python3
"""Concatenate the per-cell tables of fit_pair.py into the design record's
tables and write a Vega-Lite spec beside each, in the mmmview results
contract (src/python/resultsview/page.py: <stem>.tsv + <stem>.vl.json,
usermeta.mmmview.schema_version 1, err_over named).

Reads every ``*_<table>.tsv`` under ``<in>/fits/sub-*/``; writes to ``<out>``:

  transformation_class.tsv   every fold x rank x class row
  rank_sweep.tsv             mean +- stderr over folds per cell x class x rank
  class_summary.tsv          per cell x direction x class at the nested-CV
                             rank: mean accuracy, mean gain with its fold CI,
                             ceiling, null p (median over folds), gate
  class_winners.tsv          per cell: the winning class and whether its CI
                             clears the runner-up (the design record's rule)
  class_verdict.tsv          per rung x roi x pair across subjects: a winner
                             needs 2 of 3 subjects and sub-03 not alone
  rotation_metrics.tsv / rotation_summary.tsv
  geometry.tsv, delay_slopes.tsv, plane_spectrum.tsv, anchor_drift.tsv,
  anchor_regression.tsv, block_energy.tsv, composition.tsv,
  cue_control.tsv (transformation_class rows of enc:ret-image in HeschlsGyrus)
  identity_chance.tsv        per cell x direction x variant (all / reCon1 /
                             reCon2 / ntf = non-triplet foils / ntf_reCon1 /
                             ntf_reCon2): the identity class's 2AFC against
                             chance over folds (the plain same>different
                             test; mean, se, t, p, folds above 0.5)

Cells fitted with reCon strata (2026-09-23 on) carry *_reCon1 / *_reCon2
columns; class_summary reports them where present, older cells give n/a.

Usage:
    python report.py --in <out_root> --out <dir> [--headline-pair enc:ret-word]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import score  # noqa: E402

TABLES = ["transformation_class", "rotation_metrics", "geometry", "item_residuals",
          "delay_slopes", "plane_spectrum", "anchor_drift", "anchor_regression",
          "block_energy", "composition"]
CUE_CONTROL_ROI = "HeschlsGyrus"
CUE_CONTROL_PAIR = "enc:ret-image"


NA = ["n/a", ""]          # only these are missing; a literal "null" is a label (keep_default_na=False)


def read_tsv(path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", na_values=NA, keep_default_na=False)


def gather(in_root: Path, table: str) -> pd.DataFrame:
    files = sorted((in_root / "fits").glob(f"sub-*/*_{table}.tsv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([read_tsv(f) for f in files], ignore_index=True)
    if table == "rotation_metrics" and "arm" in df:
        df["arm"] = df["arm"].replace({"null": "permuted"})       # cells written before 2026-09-23
    return df


def class_summary(tc: pd.DataFrame, seed: int) -> pd.DataFrame:
    keys = ["subject", "rung", "roi", "pair", "beta_type", "direction", "class"]
    sel = tc[tc["rank_selected"].astype(bool)]
    rows = []
    for k, g in sel.groupby(keys):
        mean, lo, hi = score.fold_ci(g["gain"].to_numpy(), seed=seed)
        p = g["null_p"].dropna()
        row = dict(zip(keys, k), rank=int(g["rank"].median()), n_folds=len(g),
                   acc_2afc=g["acc_2afc"].mean(), acc_rank=g["acc_rank"].mean(),
                   gain=mean, ci_lo=lo, ci_hi=hi, ceiling=g["ceiling"].mean(),
                   gain_frac_ceiling=score.gain_fraction(mean, g["ceiling"].mean()),
                   null_p_median=p.median() if len(p) else np.nan,
                   frac_folds_p05=(p < 0.05).mean() if len(p) else np.nan,
                   n_vox=g["n_vox"].median(),
                   gate_passed=bool(lo > 0 and len(p) and p.median() < 0.05))
        for s in VARIANTS:
            if f"gain{s}" not in g or g[f"gain{s}"].isna().all():
                continue
            v = g[f"gain{s}"].dropna().to_numpy()
            m_s, lo_s, hi_s = score.fold_ci(v, seed=seed) if len(v) else (np.nan, np.nan, np.nan)
            p_s = g[f"null_p{s}"].dropna() if f"null_p{s}" in g else pd.Series(dtype=float)
            row.update({f"acc_2afc{s}": g[f"acc_2afc{s}"].mean(), f"gain{s}": m_s,
                        f"ci_lo{s}": lo_s, f"ci_hi{s}": hi_s,
                        f"null_p_median{s}": p_s.median() if len(p_s) else np.nan,
                        f"gate_passed{s}": bool(np.isfinite(lo_s) and lo_s > 0 and len(p_s) and p_s.median() < 0.05)})
        rows.append(row)
    return pd.DataFrame(rows)


# scoring variants written by fit_pair (suffix -> label): reCon strata, the
# non-triplet foil pool, and their crossing; older cells lack some or all
VARIANTS = ("_reCon1", "_reCon2", "_ntf", "_ntf_reCon1", "_ntf_reCon2")


def identity_chance(tc: pd.DataFrame) -> pd.DataFrame:
    """The identity class against chance, per cell x direction x stratum: the
    plain same>different item test the gate never runs (the gate compares
    maps WITH identity). One-sample t over folds at the identity class's
    nested-CV rank; uncorrected."""
    from scipy import stats
    keys = ["subject", "rung", "roi", "pair", "beta_type", "direction"]
    sel = tc[(tc["class"] == "identity") & tc["rank_selected"].astype(bool)]
    rows = []
    for k, g in sel.groupby(keys):
        for s, col in (("all", "acc_2afc"), *[(s.lstrip("_"), f"acc_2afc{s}") for s in VARIANTS]):
            if col not in g:
                continue
            v = g[col].dropna().to_numpy(float)
            if len(v) < 2:
                continue
            t, p = stats.ttest_1samp(v, 0.5)
            rows.append(dict(zip(keys, k), stratum=s, rank=int(g["rank"].median()), n_folds=len(v),
                             n_items=g[f"n_items_{s}"].sum() if s != "all" and f"n_items_{s}" in g else g["n_test_items"].sum(),
                             acc_2afc=v.mean(), se=v.std(ddof=1) / np.sqrt(len(v)), t=t, p=p,
                             folds_above=int((v > 0.5).sum()), n_vox=g["n_vox"].median()))
    return pd.DataFrame(rows)


def spec_identity_chance(headline_pair: str) -> dict:
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "title": f"Identity (same > different item) against chance, {headline_pair}, forward, by reCon stratum",
        "usermeta": _meta("folds", "Mean held-out run-matched 2AFC of the identity map over folds, "
                                   "+- stderr; chance 0.5. reCon1 = retrieved in the encoding session, "
                                   "reCon2 = a later session; candidates are shared across strata. "
                                   "Filled = uncorrected fold-wise p < .05."),
        "transform": [{"filter": f"datum.pair == '{headline_pair}' && datum.direction == 'forward'"},
                      {"calculate": "datum.acc_2afc - datum.se", "as": "lo"},
                      {"calculate": "datum.acc_2afc + datum.se", "as": "hi"}],
        "facet": {"column": {"field": "subject", "type": "nominal"}},
        "spec": {"width": 240, "height": {"step": 12},
                 "layer": [
                     {"mark": {"type": "rule", "strokeDash": [4, 4]}, "encoding": {"x": {"datum": 0.5}}},
                     {"mark": {"type": "errorbar"},
                      "encoding": {"y": {"field": "roi", "type": "nominal", "sort": {"field": "rung"}},
                                   "x": {"field": "lo", "type": "quantitative", "title": "2AFC"},
                                   "x2": {"field": "hi"}, "color": {"field": "stratum", "type": "nominal"}}},
                     {"mark": {"type": "point", "filled": True, "size": 40},
                      "encoding": {"y": {"field": "roi", "type": "nominal"},
                                   "x": {"field": "acc_2afc", "type": "quantitative"},
                                   "color": {"field": "stratum", "type": "nominal"},
                                   "opacity": {"condition": {"test": "datum.p < 0.05", "value": 1}, "value": 0.35},
                                   "tooltip": [{"field": "roi"}, {"field": "stratum"},
                                               {"field": "acc_2afc", "format": ".3f"}, {"field": "p", "format": ".3f"},
                                               {"field": "n_items"}]}}]}}


def class_winners(cs: pd.DataFrame) -> pd.DataFrame:
    keys = ["subject", "rung", "roi", "pair", "beta_type"]
    rows = []
    fwd = cs[(cs["direction"] == "forward") & (cs["class"] != "semantic_oracle")]
    for k, g in fwd.groupby(keys):
        g = g.sort_values("gain", ascending=False)
        top, second = g.iloc[0], (g.iloc[1] if len(g) > 1 else None)
        clears = bool(second is not None and top["ci_lo"] > second["gain"])
        # identity family ties: a "win" for scaled identity over identity is not a map win
        rows.append(dict(zip(keys, k), winner=top["class"], winner_gain=top["gain"],
                         winner_ci_lo=top["ci_lo"], runner_up=second["class"] if second is not None else "n/a",
                         runner_up_gain=second["gain"] if second is not None else np.nan,
                         ci_clears_runner_up=clears, winner_gate=bool(top["gate_passed"]),
                         ceiling=top["ceiling"]))
    return pd.DataFrame(rows)


def class_verdict(cw: pd.DataFrame) -> pd.DataFrame:
    keys = ["rung", "roi", "pair", "beta_type"]
    rows = []
    for k, g in cw.groupby(keys):
        wins = g[g["ci_clears_runner_up"] & g["winner_gate"]]
        counts = wins["winner"].value_counts()
        n_sub = g["subject"].nunique()
        verdict, carrier = "no class beats identity", ""
        if len(counts):
            best, n = counts.index[0], int(counts.iloc[0])
            subs = sorted(wins.loc[wins["winner"] == best, "subject"])
            carrier = "+".join(subs)
            if n >= 2 and not (subs == ["sub-03"]):
                verdict = best
            elif n == 1:
                verdict = f"{best} in one subject only"
        rows.append(dict(zip(keys, k), n_subjects=n_sub, verdict=verdict, carriers=carrier,
                         mean_ceiling=g["ceiling"].mean()))
    return pd.DataFrame(rows)


def rotation_summary(rm: pd.DataFrame, cs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per cell x metric: fitted value (mean over folds), the encoding
    split-half floor, the 2.5-97.5 % band of the item-permuted null, and the
    CELL-level gate of the orthogonal map (from class_summary: fold CI of
    gain > 0 and median permutation p < .05) -- not any-fold, which passes
    by chance in ~half the cells over 15 folds."""
    keys = ["subject", "rung", "roi", "pair", "beta_type", "metric", "subspace_m"]
    gate = {}
    if cs is not None and len(cs):
        g = cs[(cs["direction"] == "forward") & (cs["class"] == "procrustes")]
        gate = {tuple(r[k] for k in keys[:5]): bool(r["gate_passed"]) for _, r in g.iterrows()}
    rows = []
    for k, g in rm.groupby(keys, dropna=False):
        fit = g[g["arm"] == "fit"]["value"]
        floor = g[g["arm"] == "floor_enc"]["value"]
        null = g[g["arm"] == "permuted"]["value"]
        rows.append(dict(zip(keys, k), value=fit.mean(), value_sd_folds=fit.std(),
                         floor_enc=floor.mean() if len(floor) else np.nan,
                         null_lo=null.quantile(0.025) if len(null) else np.nan,
                         null_hi=null.quantile(0.975) if len(null) else np.nan,
                         null_n=int(len(null)), rank=g["rank"].median(),
                         gate_passed=gate.get(k[:5], bool(g["gate_passed"].any()))))
    return pd.DataFrame(rows)


# ── specs ────────────────────────────────────────────────────────────────────

def _meta(err_over: str, caption: str, table: str | None = None) -> dict:
    m = {"schema_version": 1, "err_over": err_over, "caption": caption}
    if table:
        m["table"] = table
    return {"mmmview": m}


def spec_class_summary(headline_pair: str) -> dict:
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "title": f"Alignment gain over identity, {headline_pair}, nested-CV rank, forward",
        "usermeta": _meta("folds", "Held-out run-matched 2AFC accuracy minus identity, as a "
                                   "fraction of the encoding ceiling's excess over chance; bars = "
                                   "bootstrap CI over folds. Filled = gate passed (CI > 0 and "
                                   "median permutation p < .05)."),
        "transform": [{"filter": f"datum.pair == '{headline_pair}' && datum.direction == 'forward'"},
                      {"calculate": "datum.gain_frac_ceiling", "as": "gf"}],
        "facet": {"column": {"field": "subject", "type": "nominal"}},
        "spec": {
            "width": 240, "height": {"step": 12},
            "layer": [
                {"mark": {"type": "rule", "strokeDash": [4, 4]}, "encoding": {"x": {"datum": 0}}},
                {"mark": {"type": "point", "filled": True, "size": 40},
                 "encoding": {
                     "y": {"field": "roi", "type": "nominal", "title": "ROI", "sort": {"field": "rung"}},
                     "x": {"field": "gf", "type": "quantitative", "title": "gain / ceiling"},
                     "color": {"field": "class", "type": "nominal"},
                     "opacity": {"condition": {"test": "datum.gate_passed == true || datum.gate_passed == 'True'", "value": 1}, "value": 0.35},
                     "tooltip": [{"field": "roi"}, {"field": "class"}, {"field": "rank"},
                                 {"field": "gain", "format": ".3f"}, {"field": "ceiling", "format": ".3f"},
                                 {"field": "null_p_median", "format": ".3f"}]}}]}}


def spec_rank_sweep(headline_pair: str) -> dict:
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "title": f"Rank sweep, {headline_pair}, forward: accuracy by class and basis rank (rungs ii-iv)",
        "usermeta": _meta("folds", "Mean +- stderr of held-out 2AFC accuracy over folds at every basis rank; "
                                   "the nested-CV pick is the headline row of class_summary. "
                                   "Rung (i) rows are omitted here for legibility (they are in the table)."),
        "transform": [{"filter": f"datum.pair == '{headline_pair}' && datum.direction == 'forward' && datum.rung != 'i'"},
                      {"calculate": "datum.acc_2afc", "as": "acc"},
                      {"calculate": "datum.acc - datum.se", "as": "lo"},
                      {"calculate": "datum.acc + datum.se", "as": "hi"}],
        "facet": {"row": {"field": "roi", "type": "nominal"}, "column": {"field": "subject", "type": "nominal"}},
        "spec": {"width": 200, "height": 120,
                 "layer": [
                     {"mark": {"type": "line", "point": True},
                      "encoding": {"x": {"field": "rank", "type": "quantitative", "scale": {"type": "log"}},
                                   "y": {"field": "acc", "type": "quantitative", "scale": {"zero": False}, "title": "2AFC"},
                                   "color": {"field": "class", "type": "nominal"}}},
                     {"mark": {"type": "errorbar"},
                      "encoding": {"x": {"field": "rank", "type": "quantitative"},
                                   "y": {"field": "lo", "type": "quantitative"},
                                   "y2": {"field": "hi"},
                                   "color": {"field": "class", "type": "nominal"}}}]}}


def spec_rotation(headline_pair: str) -> dict:
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "title": f"Rotation magnitude, {headline_pair}: variance-weighted mean plane angle vs floor and null",
        "usermeta": _meta("null draws", "Point = fitted map (mean over folds); tick = encoding "
                                        "split-half floor; bar = 2.5-97.5 % of the item-permuted null. "
                                        "Read only where the gate passed (filled)."),
        "transform": [{"filter": f"datum.pair == '{headline_pair}' && datum.metric == 'mean_plane_angle_deg'"}],
        "facet": {"column": {"field": "subject", "type": "nominal"}},
        "spec": {"width": 240, "height": {"step": 12},
                 "layer": [
                     {"mark": {"type": "rule", "color": "#6a6a7e"},
                      "encoding": {"y": {"field": "roi", "type": "nominal", "sort": {"field": "rung"}},
                                   "x": {"field": "null_lo", "type": "quantitative", "title": "degrees"},
                                   "x2": {"field": "null_hi"}}},
                     {"mark": {"type": "tick", "color": "#eda100"},
                      "encoding": {"y": {"field": "roi", "type": "nominal"},
                                   "x": {"field": "floor_enc", "type": "quantitative"}}},
                     {"mark": {"type": "point", "filled": True, "size": 45},
                      "encoding": {"y": {"field": "roi", "type": "nominal"},
                                   "x": {"field": "value", "type": "quantitative"},
                                   "opacity": {"condition": {"test": "datum.gate_passed == true || datum.gate_passed == 'True'", "value": 1}, "value": 0.35},
                                   "tooltip": [{"field": "roi"}, {"field": "value", "format": ".1f"},
                                               {"field": "floor_enc", "format": ".1f"}, {"field": "rank"}]}}]}}


def spec_delay(headline_pair: str) -> dict:
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "title": f"Delay: slope of per-item angle on item age (single-exposure items), {headline_pair}",
        "usermeta": _meta("items", "OLS slope in degrees per day with a bootstrap-over-items CI; "
                                   "top-3 planes of the pooled map (plane_angle) and the held-out "
                                   "alignment residual. A CI spanning zero is a result."),
        "transform": [{"filter": f"datum.pair == '{headline_pair}' && datum.exposure_group == 'single'"},
                      {"calculate": "datum.measure + (datum.plane_rank == null ? '' : ' p' + datum.plane_rank)", "as": "series"}],
        "facet": {"column": {"field": "subject", "type": "nominal"}},
        "spec": {"width": 240, "height": {"step": 12},
                 "layer": [
                     {"mark": {"type": "rule", "strokeDash": [4, 4]}, "encoding": {"x": {"datum": 0}}},
                     {"mark": {"type": "errorbar", "ticks": True},
                      "encoding": {"y": {"field": "roi", "type": "nominal"},
                                   "x": {"field": "ci_lo", "type": "quantitative", "title": "deg / day"},
                                   "x2": {"field": "ci_hi"},
                                   "color": {"field": "series", "type": "nominal"}}},
                     {"mark": {"type": "point", "filled": True},
                      "encoding": {"y": {"field": "roi", "type": "nominal"},
                                   "x": {"field": "slope_per_day", "type": "quantitative"},
                                   "color": {"field": "series", "type": "nominal"},
                                   "tooltip": [{"field": "roi"}, {"field": "series"},
                                               {"field": "slope_per_day", "format": ".4f"},
                                               {"field": "n_items"}, {"field": "age_dependent"}]}}]}}


def spec_geometry(headline_pair: str) -> dict:
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "title": f"Geometry preservation, {headline_pair}: RDM correlation vs encoding split-half ceiling",
        "usermeta": _meta("folds", "Spearman correlation of held-out item RDMs between the two phases "
                                   "(point, mean over folds) against the encoding split-half RDM "
                                   "correlation (tick), at rank 50 or the largest rank below it."),
        "transform": [{"filter": f"datum.pair == '{headline_pair}' && datum.rank <= 50"},
                      {"joinaggregate": [{"op": "max", "field": "rank", "as": "rmax"}], "groupby": ["subject", "roi"]},
                      {"filter": "datum.rank == datum.rmax"},
                      {"aggregate": [{"op": "mean", "field": "rdm_corr", "as": "rdm"},
                                     {"op": "mean", "field": "ceiling_rdm", "as": "ceil"}],
                       "groupby": ["subject", "roi", "rung"]}],
        "facet": {"column": {"field": "subject", "type": "nominal"}},
        "spec": {"width": 240, "height": {"step": 12},
                 "layer": [
                     {"mark": {"type": "tick", "color": "#eda100"},
                      "encoding": {"y": {"field": "roi", "type": "nominal", "sort": {"field": "rung"}},
                                   "x": {"field": "ceil", "type": "quantitative", "title": "Spearman rho"}}},
                     {"mark": {"type": "point", "filled": True},
                      "encoding": {"y": {"field": "roi", "type": "nominal"},
                                   "x": {"field": "rdm", "type": "quantitative"},
                                   "tooltip": [{"field": "roi"}, {"field": "rdm", "format": ".3f"},
                                               {"field": "ceil", "format": ".3f"}]}}]}}


def spec_composition() -> dict:
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "title": "Composition test: E->R_word->R_image composed vs direct E->R_image (held-out)",
        "usermeta": _meta("folds", "Shortfall = direct minus composed accuracy at each rank, mean and "
                                   "stderr over folds; a shortfall inside the CI says the three maps "
                                   "are consistent with one transformation."),
        "transform": [{"aggregate": [{"op": "mean", "field": "shortfall", "as": "sf"},
                                     {"op": "stderr", "field": "shortfall", "as": "se"}],
                       "groupby": ["subject", "roi", "rank"]},
                      {"calculate": "datum.sf - 1.96 * datum.se", "as": "lo"},
                      {"calculate": "datum.sf + 1.96 * datum.se", "as": "hi"}],
        "facet": {"column": {"field": "subject", "type": "nominal"}},
        "spec": {"width": 240, "height": {"step": 12},
                 "layer": [
                     {"mark": {"type": "rule", "strokeDash": [4, 4]}, "encoding": {"x": {"datum": 0}}},
                     {"mark": {"type": "errorbar", "ticks": True},
                      "encoding": {"y": {"field": "roi", "type": "nominal"},
                                   "x": {"field": "lo", "type": "quantitative", "title": "shortfall (direct - composed)"},
                                   "x2": {"field": "hi"}, "color": {"field": "rank", "type": "ordinal"}}},
                     {"mark": {"type": "point", "filled": True},
                      "encoding": {"y": {"field": "roi", "type": "nominal"},
                                   "x": {"field": "sf", "type": "quantitative"},
                                   "color": {"field": "rank", "type": "ordinal"}}}]}}


def spec_block_energy(headline_pair: str) -> dict:
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "title": f"Block energy of the orthogonal map at the union rungs, {headline_pair}",
        "usermeta": _meta("none", "Row-normalised Frobenius energy of the fitted map from source "
                                  "block to target block (mean over folds): diagonal mass = rotation "
                                  "within, off-diagonal = movement across."),
        "transform": [{"filter": f"datum.pair == '{headline_pair}'"},
                      {"aggregate": [{"op": "mean", "field": "energy_frac", "as": "e"}],
                       "groupby": ["subject", "roi", "source_block", "target_block"]}],
        "facet": {"row": {"field": "roi", "type": "nominal"}, "column": {"field": "subject", "type": "nominal"}},
        "spec": {"mark": "rect",
                 "encoding": {"y": {"field": "source_block", "type": "nominal"},
                              "x": {"field": "target_block", "type": "nominal"},
                              "color": {"field": "e", "type": "quantitative", "scale": {"scheme": "blues", "domain": [0, 1]}, "title": "energy"},
                              "tooltip": [{"field": "source_block"}, {"field": "target_block"}, {"field": "e", "format": ".3f"}]}}}


def rank_sweep(tc: pd.DataFrame) -> pd.DataFrame:
    """Mean and stderr of held-out 2AFC over folds per cell x direction x class x
    rank -- the table the rank-sweep chart reads (the per-fold table is too
    large to inline in a page)."""
    keys = ["subject", "rung", "roi", "pair", "beta_type", "direction", "class", "rank"]
    g = tc.groupby(keys)["acc_2afc"]
    out = g.agg(acc_2afc="mean", se=lambda v: v.std(ddof=1) / np.sqrt(max(len(v), 1)), n_folds="count").reset_index()
    return out


SPECS = {
    "class_summary": spec_class_summary,
    "rank_sweep": spec_rank_sweep,
    "rotation_summary": spec_rotation,
    "delay_slopes": spec_delay,
    "geometry": spec_geometry,
    "composition": lambda pair: spec_composition(),
    "block_energy": spec_block_energy,
    "identity_chance": spec_identity_chance,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_root", required=True, help="fit_pair's --out-root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--headline-pair", default="enc:ret-word")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    in_root, out = Path(args.in_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tables = {t: gather(in_root, t) for t in TABLES}
    if tables["transformation_class"].empty and tables["composition"].empty:
        sys.exit(f"ERROR: no fit tables under {in_root}/fits/")
    tc = tables["transformation_class"]
    cs = None
    if not tc.empty:
        cs = class_summary(tc, args.seed)
        cw = class_winners(cs)
        tables["class_summary"], tables["class_winners"], tables["class_verdict"] = cs, cw, class_verdict(cw)
        tables["rank_sweep"] = rank_sweep(tc)
        tables["identity_chance"] = identity_chance(tc)
        tables["cue_control"] = tc[(tc["pair"] == CUE_CONTROL_PAIR) & (tc["roi"] == CUE_CONTROL_ROI)]
    if not tables["rotation_metrics"].empty:
        tables["rotation_summary"] = rotation_summary(tables["rotation_metrics"], cs)
    if "item_residuals" in tables:
        tables.pop("item_residuals")               # per-item rows stay in fits/
    written = []
    for name, df in tables.items():
        if df is None or df.empty:
            continue
        df.to_csv(out / f"{name}.tsv", sep="\t", index=False, na_rep="n/a", float_format="%.6g")
        written.append(name)
        if name in SPECS:
            with open(out / f"{name}.vl.json", "w") as f:
                json.dump(SPECS[name](args.headline_pair), f, indent=1)
    print(f"wrote {len(written)} tables to {out}: {', '.join(written)}")
    if "class_verdict" in tables:
        print(tables["class_verdict"].to_string(index=False))


if __name__ == "__main__":
    main()
