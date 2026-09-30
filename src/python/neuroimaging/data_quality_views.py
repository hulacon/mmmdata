"""Views of the data-quality tier-2 tables: Vega-Lite specs for mmmview.

Each tier-2 part directory gets a spec per chart, in the mmmview results
contract (src/python/resultsview/page.py: ``<stem>.tsv`` + ``<stem>.vl.json``,
``usermeta.mmmview.schema_version`` 1, ``err_over`` named; a spec may name
another table in the same directory with ``usermeta.mmmview.table``). With
them in place, ``mmmview <tree>/tier2/<part>`` draws the part as one results
page, and ``mmmview serve`` reports it stale when a rebuild changes a table.

Most charts read the part's own tables. Four tables are too large to inline
in a page (per-parcel rows over films, sessions or lags), so this module
writes a small summary beside each, and the chart reads that:

  naturalistic/wsc_networks.tsv       T2.1  median r over parcels x films
  naturalistic/discriminability_networks.tsv
                                      T2.3  medians over parcels x sessions
  naturalistic/isc_networks.tsv       T2.2  median LOO-ISC over parcels x films
  naturalistic/envelope_lag_summary.tsv
                                      T2.4  median r (and IQR) over showings

A network is the Schaefer 17n name field with its A/B/C variants pooled
(VisCent, VisPeri, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default,
TempPar), as the design record reads them. Summaries are not part of the
tier-2 diff (they are derived from tables it already compares).

Only names come from the data (scope, task and subject lists for the page's
selectors); every number reaches the page through a table.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import pandas as pd

SCHEMA = "https://vega.github.io/schema/vega-lite/v6.json"
SPEC_EXT = ".vl.json"
FLOAT_FORMAT = "%.6f"

# Plotting order, lightest to heaviest nuisance model, then the two that are
# not in that ladder; tests pin it to the registry's regime set.
REGIME_ORDER = ("none", "drift", "base", "basecsfwm", "baseacc6", "base12fd", "base12fdcsfwm",
                "base12fdacc6", "base12fdacc20", "gsr", "reference")
NETWORK_ORDER = ("VisCent", "VisPeri", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default",
                 "TempPar")
SCOPE_ORDER = ("pooled_confirmed", "pooled")      # then sub-## in order

PARTS = ("naturalistic", "connectivity", "snr", "univariate", "glmsingle")
NA = ["n/a", ""]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=False, na_values=NA,
                       dtype={"sub": str, "ses": str, "run": str, "subjects": str})


def network_of(parcel: str) -> str | None:
    """'17Networks_LH_VisCent_ExStr_1' -> 'VisCent'; 'LH_DefaultB' -> 'Default'.
    None for a name that is not a Schaefer 17n parcel or network."""
    parts = str(parcel).split("_")
    if parts[0] == "17Networks":
        parts = parts[1:]
    if len(parts) < 2 or parts[0] not in ("LH", "RH"):
        return None
    net = re.sub(r"[ABC]$", "", parts[1])
    return net if net in NETWORK_ORDER else None


def _scopes(values) -> list[str]:
    vals = set(values)
    return [s for s in SCOPE_ORDER if s in vals] + sorted(vals - set(SCOPE_ORDER))


def _meta(err_over: str, caption: str, table: str | None = None) -> dict:
    m = {"schema_version": 1, "err_over": err_over, "caption": caption}
    if table:
        m["table"] = table
    return {"mmmview": m}


def _spec(title: str, meta: dict, **body) -> dict:
    return {"$schema": SCHEMA, "title": title, "usermeta": meta, **body}


def _select(name: str, options: list[str], default: str, label: str) -> dict:
    if default not in options:
        default = options[0]
    return {"name": name, "value": default,
            "bind": {"input": "select", "options": list(options), "name": f"{label} "}}


REGIME_X = {"field": "regime", "type": "nominal", "sort": list(REGIME_ORDER), "title": "confound regime",
            "axis": {"labelAngle": -45}}
REGIME_Y = {"field": "regime", "type": "nominal", "sort": list(REGIME_ORDER), "title": "confound regime"}


def _dot_by_regime(y: str, title: str, lo: str | None = None, hi: str | None = None,
                   color: dict | None = None, tooltip: list | None = None, zero: bool = False) -> dict:
    """A layered unit: median dot per regime, optional lo..hi rule."""
    yenc = {"field": y, "type": "quantitative", "title": title, "scale": {"zero": zero}}
    dot = {"mark": {"type": "point", "filled": True, "size": 50},
           "encoding": {"x": REGIME_X, "y": yenc, "tooltip": tooltip or [{"field": y, "type": "quantitative"}]}}
    if color:
        dot["encoding"]["color"] = color
        dot["encoding"]["xOffset"] = {"field": color["field"]}
    layers = []
    if lo and hi:
        rule = {"mark": {"type": "rule"},
                "encoding": {"x": REGIME_X, "y": {"field": lo, "type": "quantitative"}, "y2": {"field": hi}}}
        if color:
            rule["encoding"]["color"] = color
            rule["encoding"]["xOffset"] = {"field": color["field"]}
        layers.append(rule)
    layers.append(dot)
    return {"layer": layers}


def _heatmap(value: str, title: str, x: dict, y: dict, fmt: str = ".3f", tooltip: list | None = None,
             width: int | None = None) -> dict:
    enc = {"x": x, "y": y}
    unit = {"layer": [
        {"mark": "rect",
         "encoding": {**enc, "color": {"field": value, "type": "quantitative", "title": title,
                                       "scale": {"scheme": "viridis"}},
                      "tooltip": tooltip or [x, y, {"field": value, "type": "quantitative", "format": fmt}]}},
        {"mark": {"type": "text", "fontSize": 9},
         "encoding": {**enc, "text": {"field": value, "type": "quantitative", "format": fmt},
                      # dark text on the light end of viridis, light on the dark end
                      "color": {"condition": {"test": f"luminance(scale('color', datum['{value}'])) > 0.4",
                                              "value": "#101014"}, "value": "#e8e8ee"}}}]}
    if width:
        unit["width"] = width
    return unit


# ---------------------------------------------------------------------------
# snr (T2.8)
# ---------------------------------------------------------------------------

def snr_views(d: Path) -> dict[str, dict]:
    ts = read_tsv(d / "task_summary.tsv")
    ses = read_tsv(d / "sessions.tsv")
    tasks = ["all"] + sorted(set(ts["task"]) - {"all"})
    scopes = _scopes(ts["scope"])
    regimes = [r for r in REGIME_ORDER if r in set(ses["regime"])]
    specs = {}
    specs["tsnr_by_regime"] = _spec(
        "Median in-mask tSNR by confound regime",
        _meta("runs (interquartile range of the per-run medians)",
              "T2.8. Dot = median over runs of each run's in-mask median tSNR; bar = its interquartile range. "
              "tSNR counts variance removed, signal included, so it cannot rank regimes on its own. "
              "Each panel has its own y axis.", "task_summary.tsv"),
        params=[_select("pick_task", tasks, "all", "task")],
        transform=[{"filter": "datum.task == pick_task"}],
        facet={"field": "scope", "type": "nominal", "sort": scopes, "title": None}, columns=4,
        spec={"width": 190, "height": 170,
              **_dot_by_regime("tsnr_median", "tSNR", "tsnr_q25", "tsnr_q75",
                               tooltip=["regime", "n_runs", "n_absent",
                                        {"field": "tsnr_median", "format": ".1f"},
                                        {"field": "tsnr_q25", "format": ".1f"},
                                        {"field": "tsnr_q75", "format": ".1f"}])},
        resolve={"scale": {"y": "independent"}})
    specs["dof_loss_by_regime"] = _spec(
        "Temporal degrees of freedom lost to the regime",
        _meta("none", "T1.2 summarised. Median over runs of the fraction of volumes spent on nuisance regressors; "
                      "short runs lose the largest fraction. Pick a task with the selector.",
              "task_summary.tsv"),
        params=[_select("pick_task", tasks, "all", "task")],
        transform=[{"filter": "datum.task == pick_task"}],
        facet={"field": "scope", "type": "nominal", "sort": scopes, "title": None}, columns=4,
        spec={"width": 190, "height": 150,
              **_dot_by_regime("dof_loss_median", "fraction of DOF lost", zero=True,
                               tooltip=["regime", "n_runs", {"field": "dof_loss_median", "format": ".3f"},
                                        {"field": "n_regressors_median", "format": ".0f"}])})
    session_x = {"field": "session", "type": "quantitative", "title": "session", "scale": {"zero": False}}
    specs["tsnr_by_session"] = _spec(
        "tSNR over sessions",
        _meta("none", "T2.8 longitudinal view. Median over a session's runs of each run's in-mask median tSNR; "
                      "read dips against the coverage chart below.",
              "sessions.tsv"),
        params=[_select("pick_regime", regimes, "none", "regime")],
        transform=[{"filter": "datum.regime == pick_regime"},
                   {"calculate": "toNumber(datum.ses)", "as": "session"}],
        width=640, height=220,
        mark={"type": "line", "point": True},
        encoding={"x": session_x,
                  "y": {"field": "tsnr_median", "type": "quantitative", "title": "tSNR", "scale": {"zero": False}},
                  "color": {"field": "sub", "type": "nominal", "title": "subject"},
                  "tooltip": ["sub", "ses", "tasks", "n_runs", {"field": "tsnr_median", "format": ".1f"},
                              {"field": "tsnr_rel_median", "format": ".2f"}]})
    specs["fd_by_session"] = _spec(
        "Head motion over sessions: raw and respiration-filtered FD",
        _meta("none", "T1.7 per session. Median over the session's runs of mean framewise displacement, raw "
                      "(solid) and with the subject's breathing band notched out (dashed). Motion does not "
                      "depend on the regime; the rows shown are the `none` rows.", "sessions.tsv"),
        transform=[{"filter": "datum.regime == 'none'"},
                   {"calculate": "toNumber(datum.ses)", "as": "session"},
                   {"fold": ["fd_mean_median", "fdf_mean_median"], "as": ["fd_kind", "fd"]},
                   {"calculate": "datum.fd_kind == 'fd_mean_median' ? 'raw' : 'filtered'", "as": "FD"}],
        width=640, height=220,
        mark={"type": "line", "point": True},
        encoding={"x": session_x,
                  "y": {"field": "fd", "type": "quantitative", "title": "mean FD (mm)"},
                  "color": {"field": "sub", "type": "nominal", "title": "subject"},
                  "strokeDash": {"field": "FD", "type": "nominal", "sort": ["raw", "filtered"]},
                  "tooltip": ["sub", "ses", "tasks", "FD", {"field": "fd", "format": ".3f"},
                              {"field": "fd_frac_gt_0\\.2_max", "format": ".3f", "title": "max frac FD > 0.2"}]})
    specs["coverage_by_session"] = _spec(
        "Brain coverage over sessions",
        _meta("none", "Lowest (over the session's runs) median parcel coverage of the BOLD mask. Hippocampal "
                      "coverage is in the tooltip.", "sessions.tsv"),
        transform=[{"filter": "datum.regime == 'none'"},
                   {"calculate": "toNumber(datum.ses)", "as": "session"}],
        width=640, height=200,
        mark={"type": "line", "point": True},
        encoding={"x": session_x,
                  "y": {"field": "cov_median_min", "type": "quantitative", "title": "median parcel coverage (min over runs)",
                        "scale": {"zero": False}},
                  "color": {"field": "sub", "type": "nominal", "title": "subject"},
                  "tooltip": ["sub", "ses", {"field": "cov_median_min", "format": ".3f"},
                              {"field": "n_cov_below_floor_max", "title": "parcels below floor (max)"},
                              {"field": "hipp_cov_min", "format": ".3f"}]})
    return specs


# ---------------------------------------------------------------------------
# connectivity (T2.5-T2.7)
# ---------------------------------------------------------------------------

def connectivity_views(d: Path) -> dict[str, dict]:
    q = read_tsv(d / "qcfc.tsv")
    hs = read_tsv(d / "hipp_similarity.tsv")
    scopes = _scopes(q["scope"])
    fd_color = {"field": "fd_kind", "type": "nominal", "title": "FD", "sort": ["raw", "filtered"]}
    specs = {}
    specs["qcfc"] = _spec(
        "QC-FC: fraction of edges whose FC tracks head motion",
        _meta("none", "T2.5, rest runs, within subject. Fraction of Schaefer-400 edges whose correlation with "
                      "run mean FD has p < .05 (null .05, dashed), for raw and respiration-filtered FD.",
              "qcfc.tsv"),
        facet={"field": "scope", "type": "nominal", "sort": scopes, "title": None}, columns=5,
        spec={"width": 170, "height": 170,
              "layer": [{"mark": {"type": "rule", "strokeDash": [4, 3]}, "encoding": {"y": {"datum": 0.05}}},
                        *_dot_by_regime("frac_p05", "fraction of edges p < .05", zero=True, color=fd_color,
                                        tooltip=["scope", "regime", "fd_kind", "n_runs",
                                                 {"field": "frac_p05", "format": ".3f"},
                                                 {"field": "median_abs_qcfc", "format": ".3f"}])["layer"]]})
    specs["qcfc_distance"] = _spec(
        "QC-FC distance dependence",
        _meta("none", "T2.5. Spearman rho between an edge's QC-FC and the distance between its parcel "
                      "centroids; negative = motion inflates short-range FC more.", "qcfc.tsv"),
        facet={"field": "scope", "type": "nominal", "sort": scopes, "title": None}, columns=5,
        spec={"width": 170, "height": 170,
              "layer": [{"mark": {"type": "rule", "strokeDash": [4, 3]}, "encoding": {"y": {"datum": 0}}},
                        *_dot_by_regime("distance_dependence", "rho(QC-FC, distance)", zero=True, color=fd_color,
                                        tooltip=["scope", "regime", "fd_kind",
                                                 {"field": "distance_dependence", "format": ".3f"}])["layer"]]})
    specs["fingerprint"] = _spec(
        "Fingerprinting: identification rate from rest FC",
        _meta("none", "T2.7. Identification of subject, and of session by split halves, from rest FC; ticks = "
                      "chance. A rate near ceiling says little on its own: read the margin in the tooltip. "
                      "Session ID is descriptive, not a noise fingerprint.", "fingerprint.tsv"),
        transform=[{"filter": "isValid(datum.id_rate)"}],
        facet={"field": "test", "type": "nominal", "sort": ["subject", "session"], "title": None},
        spec={"width": 330, "height": 200,
              "layer": [
                  {"mark": {"type": "tick", "thickness": 1, "opacity": 0.6},
                   "encoding": {"x": REGIME_X, "y": {"field": "chance", "type": "quantitative"},
                                "color": {"field": "scope", "type": "nominal", "sort": scopes},
                                "xOffset": {"field": "scope", "sort": scopes}}},
                  {"mark": {"type": "point", "filled": True, "size": 50},
                   "encoding": {"x": REGIME_X,
                                "y": {"field": "id_rate", "type": "quantitative", "title": "identification rate",
                                      "scale": {"domain": [0, 1]}},
                                "color": {"field": "scope", "type": "nominal", "sort": scopes, "title": "scope"},
                                "xOffset": {"field": "scope", "sort": scopes},
                                "tooltip": ["scope", "regime", "n_targets", {"field": "id_rate", "format": ".3f"},
                                            {"field": "chance", "format": ".3f"},
                                            {"field": "margin_median", "format": ".3f"}]}}]})
    seeds = [s for s in ("both", "left", "right") if s in set(hs["seed"])] or sorted(set(hs["seed"]))
    specs["hipp_similarity"] = _spec(
        "Hippocampal FC profile: agreement between regimes",
        _meta("none", "T2.6. r between two regimes' hippocampal seed-to-parcel FC profiles.", "hipp_similarity.tsv"),
        params=[_select("pick_seed", seeds, "both", "seed")],
        transform=[{"filter": "datum.seed == pick_seed"}],
        facet={"field": "sub", "type": "nominal", "title": "subject"}, columns=3,
        spec=_heatmap("r", "r", {**REGIME_X, "field": "regime_a", "title": None},
                      {**REGIME_Y, "field": "regime_b", "title": None}, fmt=".2f",
                      tooltip=["sub", "seed", "regime_a", "regime_b", "n_parcels", {"field": "r", "format": ".3f"}]))
    return specs


# ---------------------------------------------------------------------------
# naturalistic (T2.1-T2.4)
# ---------------------------------------------------------------------------

def _with_network(df: pd.DataFrame, col: str = "parcel") -> pd.DataFrame:
    out = df.assign(network=df[col].map(network_of))
    return out[out["network"].notna()]


def _by_scope(df: pd.DataFrame, keys: list[str], agg: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
    """agg over every sub-## and over all subjects pooled, as a `scope` column."""
    parts = [df.groupby(keys).apply(agg, include_groups=False).reset_index().assign(scope="pooled")]
    for sub, g in df.groupby("sub"):
        parts.append(g.groupby(keys).apply(agg, include_groups=False).reset_index().assign(scope=f"sub-{sub}"))
    out = pd.concat(parts, ignore_index=True)
    return out[["scope"] + [c for c in out.columns if c != "scope"]]


def _sorted(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    order = {"regime": REGIME_ORDER, "network": NETWORK_ORDER}
    sort_keys = []
    for k in keys:
        if k in order:
            df = df.assign(**{f"_{k}": df[k].map({v: i for i, v in enumerate(order[k])})})
            sort_keys.append(f"_{k}")
        else:
            sort_keys.append(k)
    return df.sort_values(sort_keys).drop(columns=[c for c in df.columns if c.startswith("_")]).reset_index(drop=True)


def wsc_networks(wsc: pd.DataFrame) -> pd.DataFrame:
    """T2.1 by network: median r over parcels x films (x subjects when pooled)."""
    df = _with_network(wsc)
    out = _by_scope(df, ["regime", "network"],
                    lambda g: pd.Series({"r_median": g["r"].median(), "n": len(g),
                                         "films": g["stimulus_id"].nunique()}))
    return _sorted(out, ["scope", "regime", "network"])


def discriminability_networks(disc: pd.DataFrame) -> pd.DataFrame:
    """T2.3 by network: medians over parcels x sessions."""
    df = _with_network(disc)
    out = df.groupby(["regime", "network"]).agg(
        within_r_median=("within_r", "median"), between_r_median=("between_r", "median"),
        discriminability_median=("discriminability", "median"), n=("parcel", "size"),
        sessions=("ses", "nunique")).reset_index()
    return _sorted(out, ["regime", "network"])


def isc_networks(isc: pd.DataFrame) -> pd.DataFrame:
    """T2.2 by network: median LOO-ISC over parcels x films (x subjects when pooled)."""
    df = _with_network(isc)
    out = _by_scope(df, ["regime", "network"],
                    lambda g: pd.Series({"isc_median": g["isc"].median(), "n": len(g),
                                         "films": g["stimulus_id"].nunique()}))
    return _sorted(out, ["scope", "regime", "network"])


def envelope_lag_summary(lag: pd.DataFrame) -> pd.DataFrame:
    """T2.4 across films: median and IQR of r at each lag over showings."""
    def agg(g):
        return pd.Series({"r_median": g["r"].median(), "r_q25": g["r"].quantile(0.25),
                          "r_q75": g["r"].quantile(0.75), "n_showings": len(g)})
    out = _by_scope(lag, ["roi", "envelope", "regime", "lag"], agg)
    return _sorted(out, ["scope", "roi", "envelope", "regime", "lag"])


NATURALISTIC_SUMMARIES = {
    "wsc_networks": ("wsc.tsv", wsc_networks),
    "discriminability_networks": ("discriminability.tsv", discriminability_networks),
    "isc_networks": ("loo_isc.tsv", isc_networks),
    "envelope_lag_summary": ("envelope_lag.tsv", envelope_lag_summary),
}


def naturalistic_views(d: Path) -> dict[str, dict]:
    tables = {}
    for name, (source, fn) in NATURALISTIC_SUMMARIES.items():
        table = fn(read_tsv(d / source))
        table.to_csv(d / f"{name}.tsv", sep="\t", index=False, na_rep="n/a", float_format=FLOAT_FORMAT)
        tables[name] = table
    pairs = read_tsv(d / "wsc_pairs.tsv")
    regimes = [r for r in REGIME_ORDER if r in set(pairs["regime"])]
    net_x = {"field": "network", "type": "nominal", "sort": list(NETWORK_ORDER), "title": None,
             "axis": {"labelAngle": -45}}
    specs = {}
    specs["wsc_networks"] = _spec(
        "Repeat-viewing reliability (WSC) by network",
        _meta("none", "T2.1: films a subject saw in more than one session. Median over parcels x films (x "
                      "subjects when pooled) of the per-parcel r between showings."),
        params=[_select("pick_scope", _scopes(tables["wsc_networks"]["scope"]), "pooled", "scope")],
        transform=[{"filter": "datum.scope == pick_scope"}],
        width=460, height=280,
        **_heatmap("r_median", "median r", net_x, REGIME_Y,
                   tooltip=["scope", "regime", "network", "n", "films", {"field": "r_median", "format": ".3f"}]))
    specs["wsc_by_gap"] = _spec(
        "Repeat-viewing reliability against the gap between showings",
        _meta("none", "T2.1. Each dot is one pair of showings: median r over parcels. Line = median at each gap.", "wsc_pairs.tsv"),
        params=[_select("pick_regime", regimes, "none", "regime")],
        transform=[{"filter": "datum.regime == pick_regime"},
                   {"calculate": "toNumber(datum.ses_b) - toNumber(datum.ses_a)", "as": "gap"}],
        facet={"field": "stimulus_id", "type": "nominal", "title": None},
        spec={"width": 300, "height": 200, "layer": [
            {"mark": {"type": "point", "opacity": 0.6},
             "encoding": {"x": {"field": "gap", "type": "quantitative", "title": "sessions apart"},
                          "y": {"field": "r_median", "type": "quantitative", "title": "median r over parcels"},
                          "color": {"field": "sub", "type": "nominal", "title": "subject"},
                          "tooltip": ["sub", "ses_a", "ses_b", {"field": "r_median", "format": ".3f"}]}},
            {"mark": "line",
             "encoding": {"x": {"field": "gap", "type": "quantitative"},
                          "y": {"field": "r_median", "type": "quantitative", "aggregate": "median"},
                          "color": {"field": "sub", "type": "nominal"}}}]})
    specs["discriminability_networks"] = _spec(
        "Film discriminability by network",
        _meta("none", "T2.3. Per session, same-film cross-subject r minus different-film cross-subject r, per "
                      "parcel; median over parcels x sessions.", "discriminability_networks.tsv"),
        width=460, height=280,
        **_heatmap("discriminability_median", "median discriminability", net_x, REGIME_Y,
                   tooltip=["regime", "network", "n", "sessions",
                            {"field": "discriminability_median", "format": ".3f"},
                            {"field": "within_r_median", "format": ".3f"},
                            {"field": "between_r_median", "format": ".3f"}]))
    specs["discriminability_within"] = _spec(
        "Same-film cross-subject r by network",
        _meta("none", "T2.3, the within term alone: median over parcels x sessions of the same-film r between "
                      "subjects.",
              "discriminability_networks.tsv"),
        width=460, height=280,
        **_heatmap("within_r_median", "median same-film r", net_x, REGIME_Y))
    specs["isc_networks"] = _spec(
        "Leave-one-out ISC by network (first viewings)",
        _meta("none", "T2.2. Each subject against the mean of the others, every film's first viewing; median "
                      "over parcels x films (x subjects when pooled).", "isc_networks.tsv"),
        params=[_select("pick_scope", _scopes(tables["isc_networks"]["scope"]), "pooled", "scope")],
        transform=[{"filter": "datum.scope == pick_scope"}],
        width=460, height=280,
        **_heatmap("isc_median", "median LOO-ISC", net_x, REGIME_Y,
                   tooltip=["scope", "regime", "network", "n", "films", {"field": "isc_median", "format": ".3f"}]))
    env = tables["envelope_lag_summary"]
    specs["envelope_lag_summary"] = _spec(
        "Auditory cortex against the film's audio envelope, by lag",
        _meta("none", "T2.4 over every film's first viewing: median r (over showings) between the SomMotB "
                      "auditory ROI and the audio envelope at each lag in TRs; negative = audio leads BOLD. "
                      "Shaded = the plausible haemodynamic window. Click a legend entry to pick out a regime.",
              "envelope_lag_summary.tsv"),
        params=[_select("pick_scope", _scopes(env["scope"]), "pooled", "scope")],
        transform=[{"filter": "datum.scope == pick_scope"}],
        facet={"field": "envelope", "type": "nominal", "title": None},
        spec={"width": 330, "height": 220, "layer": [
            {"mark": {"type": "rect", "opacity": 0.12, "color": "#9a9aae"},
             "encoding": {"x": {"datum": -5}, "x2": {"datum": -2}}},
            {"mark": {"type": "line"},
             "params": [{"name": "hl", "select": {"type": "point", "fields": ["regime"]}, "bind": "legend"}],
             "encoding": {"x": {"field": "lag", "type": "quantitative", "title": "lag (TR)"},
                          "y": {"field": "r_median", "type": "quantitative", "title": "median r"},
                          "color": {"field": "regime", "type": "nominal", "sort": list(REGIME_ORDER),
                                    "scale": {"scheme": "tableau20"}},
                          "opacity": {"condition": {"param": "hl", "value": 1}, "value": 0.12},
                          "tooltip": ["regime", "lag", "n_showings", {"field": "r_median", "format": ".3f"},
                                      {"field": "r_q25", "format": ".3f"}, {"field": "r_q75", "format": ".3f"}]}}]})
    return specs


# ---------------------------------------------------------------------------
# univariate (T1.5, T1.8, T2.13)
# ---------------------------------------------------------------------------

def univariate_views(d: Path) -> dict[str, dict]:
    r2 = read_tsv(d / "task_r2.tsv")
    scopes = _scopes(r2["scope"])
    tasks = ["all"] + sorted(set(r2["task"]) - {"all"})
    scope_param = _select("pick_scope", scopes, "pooled_confirmed", "scope")
    task_facet = {"field": "task", "type": "nominal", "sort": tasks, "title": None}
    specs = {}
    specs["task_f_fraction"] = _spec(
        "Task signal: fraction of voxels with task F p < .001",
        _meta("runs (interquartile range of the per-run fractions)",
              "T1.5 headline. Dot = median over runs; bar = IQR; dashed = the .001 null. The F test assumes "
              "independent residuals, so regimes without drift terms (`none`, `gsr`) can read high from "
              "autocorrelation alone; a cosine high-pass also removes blocks longer than its cutoff.", "task_r2.tsv"),
        params=[scope_param],
        transform=[{"filter": "datum.scope == pick_scope"}],
        facet=task_facet, columns=4,
        spec={"width": 190, "height": 160,
              "layer": [{"mark": {"type": "rule", "strokeDash": [4, 3]}, "encoding": {"y": {"datum": 0.001}}},
                        *_dot_by_regime("frac_p001_median", "fraction p < .001", "frac_p001_q25", "frac_p001_q75",
                                        zero=True,
                                        tooltip=["task", "regime", "n_runs",
                                                 {"field": "frac_p001_median", "format": ".4f"},
                                                 {"field": "frac_p001_q25", "format": ".4f"},
                                                 {"field": "frac_p001_q75", "format": ".4f"}])["layer"]]},
        resolve={"scale": {"y": "independent"}})
    specs["task_r2adj"] = _spec(
        "Task signal: adjusted R²",
        _meta("runs (interquartile range of the per-run medians)",
              "T1.5. Median over runs of each run's median voxel adjusted R² for the task columns. Still "
              "dof-skewed on short runs; read the F fraction first.", "task_r2.tsv"),
        params=[scope_param],
        transform=[{"filter": "datum.scope == pick_scope"}],
        facet=task_facet, columns=4,
        spec={"width": 190, "height": 160,
              **_dot_by_regime("r2adj_median", "adjusted R²", "r2adj_median_q25", "r2adj_median_q75",
                               tooltip=["task", "regime", "n_runs", {"field": "r2adj_median", "format": ".4f"},
                                        {"field": "r2raw_median", "format": ".4f"}])},
        resolve={"scale": {"y": "independent"}})
    specs["motion_task_r"] = _spec(
        "Motion–task correlation by task",
        _meta("none", "T1.8. Median (bar) and max (tick) over runs of the largest |r| between a motion "
                      "parameter and a task regressor. Short runs reach high values by chance, so this is a "
                      "warning, not evidence of task-locked motion. Regime-independent; the `none` rows are shown.",
              "task_r2.tsv"),
        params=[scope_param],
        transform=[{"filter": "datum.scope == pick_scope && datum.regime == 'none' && datum.task != 'all'"}],
        width=520, height=220,
        layer=[{"mark": "bar",
                "encoding": {"x": {"field": "task", "type": "nominal", "title": None, "axis": {"labelAngle": -45}},
                             "y": {"field": "motion_task_r_median", "type": "quantitative", "title": "max |r|",
                                   "scale": {"domain": [0, 1]}},
                             "tooltip": ["task", "n_runs", {"field": "motion_task_r_median", "format": ".3f"},
                                         {"field": "motion_task_r_max", "format": ".3f"}]}},
               {"mark": {"type": "tick", "color": "#eb6834"},
                "encoding": {"x": {"field": "task", "type": "nominal"},
                             "y": {"field": "motion_task_r_max", "type": "quantitative"}}}])
    for stem, field, title, caption in (
            ("split_half_dice", "dice@1000", "Dice of the top 1,000 voxels",
             "T2.13, unsmoothed. Localizer runs split in two halves; Dice overlap of the 1,000 highest-effect "
             "voxels per half. Faint dots = contrasts; line = median over contrasts. Tasks with no Dice value "
             "(n/a) are left out."),
            ("split_half_r", "r", "whole-mask r",
             "T2.13, unsmoothed. Correlation of the two halves' effect maps over the whole mask. Faint dots = "
             "contrasts; line = median over contrasts.")):
        specs[stem] = _spec(
            f"Localizer split-half reliability: {title}",
            _meta("none", caption, "split_half.tsv"),
            transform=[{"filter": f"isValid(datum['{field}'])"}],
            facet={"field": "task", "type": "nominal", "title": None},
            spec={"width": 260, "height": 200, "layer": [
                {"mark": {"type": "point", "opacity": 0.35},
                 "encoding": {"x": REGIME_X, "y": {"field": field, "type": "quantitative", "title": title},
                              "color": {"field": "sub", "type": "nominal", "title": "subject"},
                              "xOffset": {"field": "sub"},
                              "tooltip": ["sub", "contrast", "regime", "n_runs",
                                          {"field": field, "format": ".3f"}]}},
                {"mark": {"type": "line", "point": True},
                 "encoding": {"x": REGIME_X,
                              "y": {"field": field, "type": "quantitative", "aggregate": "median"},
                              "color": {"field": "sub", "type": "nominal"}}}]},
            resolve={"scale": {"y": "independent"}})
    return specs


# ---------------------------------------------------------------------------
# glmsingle (T2.10-T2.12)
# ---------------------------------------------------------------------------

def glmsingle_views(d: Path) -> dict[str, dict]:
    nets = read_tsv(d / "networks.tsv")
    arms = ["enc", "ret-image", "ret-word"]
    arm_facet = {"field": "arm", "type": "nominal", "sort": arms, "title": None}
    beta_x = {"field": "beta_type", "type": "nominal", "title": "GLMsingle beta type", "sort": ["B", "C", "D"]}
    sub_color = {"field": "sub", "type": "nominal", "title": "subject"}
    specs = {}
    specs["repeat_frac_exceed"] = _spec(
        "Noise ceiling: fraction of voxels above the shuffle null",
        _meta("none", "T2.12 headline. Fraction of floored voxels whose NSD ncsnr exceeds the 95th percentile "
                      "of 19 within-run label shuffles (null .05, dashed); raw ncsnr is not read because it has "
                      "a positive floor without signal. The repeats' scope (within or across sessions) is in the "
                      "tooltip. Rows without a fit are left out.", "fit_summary.tsv"),
        transform=[{"filter": "isValid(datum.repeat_frac_exceed)"}],
        facet=arm_facet,
        spec={"width": 180, "height": 200, "layer": [
            {"mark": {"type": "rule", "strokeDash": [4, 3]}, "encoding": {"y": {"datum": 0.05}}},
            {"mark": {"type": "point", "filled": True, "size": 60},
             "encoding": {"x": beta_x, "xOffset": {"field": "sub"}, "color": sub_color,
                          "y": {"field": "repeat_frac_exceed", "type": "quantitative",
                                "title": "fraction above null", "scale": {"zero": True}},
                          "tooltip": ["sub", "arm", "beta_type", "repeat_rep_scope", "repeat_n_conditions",
                                      {"field": "repeat_frac_exceed", "format": ".3f"},
                                      {"field": "anchor_frac_exceed", "format": ".3f"}]}}]})
    specs["fit_r2"] = _spec(
        "GLMsingle fit quality: median voxel R²",
        _meta("none", "T2.11. Median over floored voxels of each fit's R² (%).", "fit_summary.tsv"),
        transform=[{"filter": "isValid(datum.r2_median)"}],
        facet=arm_facet,
        spec={"width": 180, "height": 200,
              "mark": {"type": "point", "filled": True, "size": 60},
              "encoding": {"x": beta_x, "xOffset": {"field": "sub"}, "color": sub_color,
                           "y": {"field": "r2_median", "type": "quantitative", "title": "median R² (%)"},
                           "tooltip": ["sub", "arm", "beta_type", {"field": "r2_median", "format": ".1f"},
                                       {"field": "r2_p90", "format": ".1f"}]}})
    betas = [b for b in ("B", "C", "D") if b in set(nets["beta_type"])]
    # cortex (LH then RH, network order) above the subcortical structures
    cortex = sorted({r for r in nets.loc[nets["atlas"] != "HOSPA", "region"]},
                    key=lambda r: (r[:2], NETWORK_ORDER.index(network_of(r)) if network_of(r) else 99, r))
    regions = cortex + sorted(set(nets.loc[nets["atlas"] == "HOSPA", "region"]) - set(cortex))
    specs["networks_repeat_frac_exceed"] = _spec(
        "Noise ceiling by network",
        _meta("none", "T2.12 by region: fraction of the region's floored voxels above the shuffle null (.05).", "networks.tsv"),
        params=[_select("pick_beta", betas, "D", "beta type")],
        transform=[{"filter": "datum.beta_type == pick_beta"}],
        facet=arm_facet,
        spec=_heatmap("repeat_frac_exceed", "fraction above null",
                      {"field": "sub", "type": "nominal", "title": "subject"},
                      {"field": "region", "type": "nominal", "title": None, "sort": regions},
                      fmt=".2f", width=110,
                      tooltip=["sub", "arm", "beta_type", "atlas", "region", "n_voxels_floor",
                               {"field": "repeat_frac_exceed", "format": ".3f"},
                               {"field": "r2_median", "format": ".1f"}]))
    return specs


BUILDERS = {"snr": snr_views, "connectivity": connectivity_views, "naturalistic": naturalistic_views,
            "univariate": univariate_views, "glmsingle": glmsingle_views}


def _normalise(node):
    """Tooltip shorthand ("sub") -> a field definition ({"field": "sub"})."""
    if isinstance(node, dict):
        return {k: ([{"field": t} if isinstance(t, str) else _normalise(t) for t in v]
                    if k == "tooltip" and isinstance(v, list) else _normalise(v))
                for k, v in node.items()}
    if isinstance(node, list):
        return [_normalise(v) for v in node]
    return node


def write_views(part: str, part_dir: Path) -> list[Path]:
    """Write a part's specs (and any summary tables) into its directory.
    Returns the spec paths, in page order. Every *.vl.json in the directory
    is this function's: one it no longer writes is removed."""
    part_dir = Path(part_dir)
    specs = {stem: _normalise(spec) for stem, spec in BUILDERS[part](part_dir).items()}
    written = []
    for i, (stem, spec) in enumerate(specs.items()):
        # the page orders charts by file name; a numeric prefix keeps the
        # order above, and the table link goes through usermeta.mmmview.table
        meta = spec["usermeta"]["mmmview"]
        meta.setdefault("table", f"{stem}.tsv")
        path = part_dir / f"{i:02d}_{stem}{SPEC_EXT}"
        path.write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n")
        written.append(path)
    for stale in part_dir.glob(f"*{SPEC_EXT}"):
        if stale not in written:
            stale.unlink()
    return written
