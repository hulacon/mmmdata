#!/usr/bin/env python3
"""Phase 7 — pattern-similarity report (figures + stats + standalone HTML).

Consumes results/similarity_summary.tsv (Phase 6) and produces:
  figures/cells_TB.png, cells_NAT.png   — 4-cell bars per ROI x timeseries,
                                          both pipelines side by side
  figures/pipeline_delta.png            — paired nordic - original deltas
  results/stats_contrasts.tsv           — paired Wilcoxon: same vs diff
                                          (within + across), within vs across
  results/stats_pipeline.tsv            — paired Wilcoxon: nordic vs original
                                          per cell
  report/index.html                     — self-contained (figures embedded)

Pairing conventions (unit-item rows, session_scope=all):
  - same vs diff: within_same/within_diff (and across_same/across_diff)
    joined on (unit, item).
  - within vs across: each within_same row (subject, item) is paired with
    the mean of the across_same rows for subject pairs containing that
    subject and the same item (TB triplet items + both NAT movies).
  - nordic vs original: joined on (unit, item) per cell.

Plan: docs/doc/pattern-similarity-plan.md, Phase 7.

Usage:
    python report.py
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from shared import (  # noqa: E402
    FIGURES_DIR, PATTERN_ROI_NAMES, PS_ROOT, RESULTS_DIR,
)

REPORT_DIR = PS_ROOT / "report"
CELLS = ["within_same", "within_diff", "across_same", "across_diff"]
CELL_LABELS = {"within_same": "within/same", "within_diff": "within/diff",
               "across_same": "across/same", "across_diff": "across/diff"}
CELL_COLORS = {"within_same": "#1b6ca8", "within_diff": "#8ab6d6",
               "across_same": "#c0392b", "across_diff": "#e6a09a"}
PIPELINES = ["original", "nordic"]
TIMESERIES = ["rawtr", "glmsingle"]
PARADIGMS = ["TB", "NAT"]


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "similarity_summary.tsv", sep="\t",
                     dtype={"item": str})
    return df[df["session_scope"].isin(["all"])].copy()


# ── paired contrasts ─────────────────────────────────────────────────────

def _wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:  # all-zero differences / too few pairs
        return np.nan


def contrast_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Same-vs-diff (within, across) + within-vs-across per stream x ROI."""
    rows = []
    grp_cols = ["paradigm", "pipeline", "timeseries", "roi"]
    for keys, g in df.groupby(grp_cols):
        cell = {c: g[g["cell"] == c].set_index(["unit", "item"])["r"]
                for c in CELLS}
        base = dict(zip(grp_cols, keys))

        for scope_name, s_key, d_key in (
                ("within_same_vs_diff", "within_same", "within_diff"),
                ("across_same_vs_diff", "across_same", "across_diff")):
            joined = pd.concat([cell[s_key], cell[d_key]], axis=1,
                               join="inner", keys=["same", "diff"])
            rows.append(base | dict(
                contrast=scope_name, n=len(joined),
                mean_a=joined["same"].mean(), mean_b=joined["diff"].mean(),
                delta=(joined["same"] - joined["diff"]).mean(),
                p=_wilcoxon(joined["same"], joined["diff"])))

        # within vs across (same-item cells): pair each (subject, item)
        # within row with the mean of across rows containing that subject
        w = cell["within_same"].reset_index()
        a = cell["across_same"].reset_index()
        a_pairs = a["unit"].str.split("+")
        pairs_w, pairs_a = [], []
        for _, row in w.iterrows():
            sel = a[(a["item"] == row["item"]) &
                    a_pairs.apply(lambda p: row["unit"] in p)]
            if len(sel):
                pairs_w.append(row["r"])
                pairs_a.append(sel["r"].mean())
        pairs_w, pairs_a = np.asarray(pairs_w), np.asarray(pairs_a)
        rows.append(base | dict(
            contrast="within_vs_across_same", n=len(pairs_w),
            mean_a=pairs_w.mean(), mean_b=pairs_a.mean(),
            delta=(pairs_w - pairs_a).mean(),
            p=_wilcoxon(pairs_w, pairs_a)))
    return pd.DataFrame(rows)


def pipeline_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Paired nordic - original per (paradigm, timeseries, roi, cell)."""
    rows = []
    grp_cols = ["paradigm", "timeseries", "roi", "cell"]
    for keys, g in df.groupby(grp_cols):
        piv = g.pivot_table(index=["unit", "item"], columns="pipeline",
                            values="r")
        if not set(PIPELINES) <= set(piv.columns):
            continue
        piv = piv.dropna()
        rows.append(dict(zip(grp_cols, keys)) | dict(
            n=len(piv), original=piv["original"].mean(),
            nordic=piv["nordic"].mean(),
            delta=(piv["nordic"] - piv["original"]).mean(),
            p=_wilcoxon(piv["nordic"], piv["original"])))
    return pd.DataFrame(rows)


def context_table(df: pd.DataFrame) -> pd.DataFrame:
    """TB across-subject cells split by triplet/single context match."""
    sel = df[(df["paradigm"] == "TB") & df["cell"].str.startswith("across")]
    return (sel.groupby(["pipeline", "timeseries", "cell", "context_match"])
            .agg(r=("r", "mean"), n_rows=("r", "size")).reset_index())


# ── figures ──────────────────────────────────────────────────────────────

def fig_cells(df: pd.DataFrame, paradigm: str) -> Path:
    """Bars: rows = 6 ROIs, cols = timeseries; 4 cells x 2 pipelines."""
    sub = df[df["paradigm"] == paradigm]
    fig, axes = plt.subplots(len(PATTERN_ROI_NAMES), len(TIMESERIES),
                             figsize=(10, 16), sharex=True)
    width = 0.09
    for i, roi in enumerate(PATTERN_ROI_NAMES):
        for j, ts in enumerate(TIMESERIES):
            ax = axes[i, j]
            for ci, cell in enumerate(CELLS):
                for pi, pipe in enumerate(PIPELINES):
                    g = sub[(sub.roi == roi) & (sub.timeseries == ts) &
                            (sub.cell == cell) & (sub.pipeline == pipe)]
                    x = ci * 0.25 + pi * width
                    ax.bar(x, g["r"].mean(), width=width * 0.9,
                           color=CELL_COLORS[cell],
                           alpha=1.0 if pipe == "original" else 0.55,
                           edgecolor="black", linewidth=0.4)
                    ax.errorbar(x, g["r"].mean(), yerr=g["r"].std(),
                                fmt="none", ecolor="black",
                                elinewidth=0.7, capsize=2)
            ax.axhline(0, color="gray", linewidth=0.6)
            ax.set_xticks([ci * 0.25 + width / 2 for ci in range(len(CELLS))])
            ax.set_xticklabels([CELL_LABELS[c] for c in CELLS],
                               rotation=30, ha="right", fontsize=7)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(f"{roi}\nmean r", fontsize=8)
            if i == 0:
                ax.set_title(f"{ts}", fontsize=10)
    fig.suptitle(f"{paradigm}: 4-cell pattern similarity "
                 f"(solid = original, faded = nordic; error = SD over units)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    p = FIGURES_DIR / f"cells_{paradigm}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_pipeline_delta(stats: pd.DataFrame) -> Path:
    """Nordic - original delta per cell, panel per paradigm x timeseries."""
    fig, axes = plt.subplots(len(PARADIGMS), len(TIMESERIES),
                             figsize=(11, 7), sharex=True)
    xs = np.arange(len(PATTERN_ROI_NAMES))
    for i, par in enumerate(PARADIGMS):
        for j, ts in enumerate(TIMESERIES):
            ax = axes[i, j]
            for ci, cell in enumerate(CELLS):
                g = (stats[(stats.paradigm == par) & (stats.timeseries == ts)
                           & (stats.cell == cell)]
                     .set_index("roi").reindex(PATTERN_ROI_NAMES))
                ax.plot(xs + (ci - 1.5) * 0.12, g["delta"], "o",
                        color=CELL_COLORS[cell], markersize=5,
                        label=CELL_LABELS[cell] if (i, j) == (0, 0) else None)
            ax.axhline(0, color="gray", linewidth=0.6)
            ax.set_xticks(xs)
            ax.set_xticklabels(PATTERN_ROI_NAMES, rotation=30, ha="right",
                               fontsize=8)
            ax.tick_params(labelsize=8)
            ax.set_title(f"{par} / {ts}", fontsize=10)
            if j == 0:
                ax.set_ylabel("nordic − original (Δr)", fontsize=9)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Pipeline effect on pattern similarity (paired unit-item "
                 "deltas)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = FIGURES_DIR / "pipeline_delta.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ── HTML ─────────────────────────────────────────────────────────────────

CSS = """
body { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
       margin: 2rem auto; max-width: 1100px; color: #222; padding: 0 1rem; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.2rem; margin-top: 2rem;
     border-bottom: 1px solid #ccc; padding-bottom: 0.3rem; }
table { border-collapse: collapse; font-size: 0.82rem; margin: 1rem 0; }
th, td { border: 1px solid #ddd; padding: 0.25rem 0.5rem; text-align: right; }
th { background: #f0f3f6; } td:first-child, th:first-child { text-align: left; }
img { max-width: 100%; height: auto; border: 1px solid #eee; margin: 0.5rem 0; }
.note { color: #555; font-size: 0.85rem; }
.sig { background: #e8f4e8; }
"""


def _embed(p: Path) -> str:
    return ("data:image/png;base64,"
            + base64.b64encode(p.read_bytes()).decode())


def _table_html(df: pd.DataFrame, float_cols: dict[str, str],
                sig_col: str | None = None) -> str:
    d = df.copy()
    for c, f in float_cols.items():
        if c in d.columns:
            d[c] = d[c].map(lambda v: f % v if pd.notna(v) else "—")
    head = "".join(f"<th>{c}</th>" for c in d.columns)
    body = []
    for _, row in d.iterrows():
        cls = ""
        if sig_col is not None:
            try:
                cls = ' class="sig"' if float(row[sig_col]) < 0.05 else ""
            except (TypeError, ValueError):
                cls = ""
        cells = "".join(f"<td>{v}</td>" for v in row)
        body.append(f"<tr{cls}>{cells}</tr>")
    return (f"<table><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>")


def build_html(df, cells_means, contrasts, pipe_stats, ctx, figs) -> str:
    n_combos = df.groupby(["paradigm", "pipeline", "timeseries",
                           "roi"]).ngroups
    parts = [f"<style>{CSS}</style>",
             "<h1>Pattern-similarity analysis — "
             "preprocessing × timeseries × paradigm</h1>",
             f"<p class='note'>Grid: 2 paradigms × 2 pipelines × 2 timeseries"
             f" × 6 ROIs = {n_combos} combos × 4 cells. Metric: Pearson r "
             "across mutually finite ROI voxels, averaged over matched "
             "trials/TRs/chunks; session_scope=all rows. Generated by "
             "scripts/pattern_similarity/report.py.</p>"]
    parts.append("<h2>4-cell similarity by ROI</h2>")
    for par in PARADIGMS:
        parts.append(f"<h3>{par}</h3><img src='{figs[f'cells_{par}']}' "
                     f"alt='{par} cells figure'>")
    parts.append("<h2>Pipeline effect (NORDIC − original)</h2>")
    parts.append(f"<img src='{figs['pipeline_delta']}' alt='delta figure'>")
    parts.append(_table_html(
        pipe_stats, {"original": "%.4f", "nordic": "%.4f",
                     "delta": "%+.4f", "p": "%.4g"}, sig_col="p"))
    parts.append("<h2>Contrasts (paired Wilcoxon)</h2>")
    parts.append("<p class='note'>mean_a / mean_b = first / second member "
                 "of the contrast; green rows p &lt; 0.05 (uncorrected).</p>")
    parts.append(_table_html(
        contrasts, {"mean_a": "%.4f", "mean_b": "%.4f", "delta": "%+.4f",
                    "p": "%.4g"}, sig_col="p"))
    parts.append("<h2>TB across-subject context match</h2>")
    parts.append("<p class='note'>An item can be triplet-context in one "
                 "subject and single-context in the other (plan risk #6).</p>")
    parts.append(_table_html(ctx, {"r": "%.4f"}))
    parts.append("<h2>Cell means (collapsed)</h2>")
    parts.append(_table_html(cells_means, {"r": "%.4f", "n_voxels": "%.0f"}))
    return "\n".join(parts)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    df = load_summary()
    cells_means = pd.read_csv(RESULTS_DIR / "similarity_cells.tsv", sep="\t")

    contrasts = contrast_stats(df)
    pipe_stats = pipeline_stats(df)
    ctx = context_table(df)
    contrasts.to_csv(RESULTS_DIR / "stats_contrasts.tsv", sep="\t",
                     index=False, float_format="%.6g")
    pipe_stats.to_csv(RESULTS_DIR / "stats_pipeline.tsv", sep="\t",
                      index=False, float_format="%.6g")
    print(f"Wrote stats: {len(contrasts)} contrast rows, "
          f"{len(pipe_stats)} pipeline rows")

    figs = {f"cells_{par}": _embed(fig_cells(df, par)) for par in PARADIGMS}
    figs["pipeline_delta"] = _embed(fig_pipeline_delta(pipe_stats))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "index.html"
    out.write_text(build_html(df, cells_means, contrasts, pipe_stats, ctx,
                              figs))
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    n_contrast = len(contrasts)
    expect = 2 * 2 * 2 * 6 * 3
    assert n_contrast == expect, f"{n_contrast} contrast rows != {expect}"
    assert len(pipe_stats) == 2 * 2 * 6 * 4, "pipeline stats rows off"
    print("Row-count checks passed.\nDone.")


if __name__ == "__main__":
    main()
