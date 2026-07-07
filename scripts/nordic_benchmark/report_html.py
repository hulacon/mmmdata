#!/usr/bin/env python3
"""Render the NORDIC benchmark HTML report from on-disk results.

Reproducible replacement for the previously hand-built
`derivatives/nordic/benchmark/benchmark_report.html`. Every number in the
tables, stat row, and statistics section is computed from the TSVs written by
`report.py`; the four figures are embedded as base64 PNGs from `figures/`. The
editorial prose is templated here, so re-running after a data refresh keeps the
narrative but refreshes all values.

Run `report.py` first (it writes benchmark_summary.tsv, within_between_by_roi.tsv
and the figures), then this script.

Inputs (under --output-root, default derivatives/nordic/benchmark/):
  benchmark_summary.tsv         unified within_r/between_r/discriminability
  nat/pair_correlations.tsv     NAT rows incl. confound_model column
  figures/discriminability_per_roi.png
  figures/nordic_effect_paired.png
  figures/within_between_scatter.png
  figures/within_between_bars.png

Outputs (both written by default; see --mode):
  benchmark_report.html          linked  — references figures/*.png (lightweight)
  benchmark_report.bundled.html  bundled — figures base64-embedded, fully
                                 self-contained for sharing (email, upload, etc.)

Both are otherwise identical. The linked version is smaller and convenient for
local viewing next to figures/; the bundled version has no external references
and can travel as a single file.
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from shared import BENCHMARK_OUT, ROI_KEYS  # noqa: E402

ROI_ORDER = [f"{r}_{h}" for r, h in ROI_KEYS]
GENERATED = "2026-07-07"  # stamp; passed explicitly to stay deterministic

# ── data + derived quantities ────────────────────────────────────────────


def load_inputs(root: Path):
    summ = pd.read_csv(root / "benchmark_summary.tsv", sep="\t")
    nat = pd.read_csv(root / "nat" / "pair_correlations.tsv", sep="\t")
    return summ, nat


def mean_by_roi(df: pd.DataFrame, stream: str) -> pd.DataFrame:
    """Mean within/between/discriminability per ROI × pipeline for one stream."""
    s = df[df["stream"] == stream]
    g = (s.groupby(["roi", "pipeline"])[["within_r", "between_r", "discriminability"]]
           .mean())
    return g


def disc_table(df: pd.DataFrame, stream: str) -> list[dict]:
    """One row per ROI: original/NORDIC discriminability, delta, delta%."""
    g = mean_by_roi(df, stream)
    rows = []
    for roi in ROI_ORDER:
        try:
            o = g.loc[(roi, "original"), "discriminability"]
            n = g.loc[(roi, "nordic"), "discriminability"]
        except KeyError:
            continue
        d = n - o
        pct = 100 * d / abs(o) if o != 0 else float("nan")
        rows.append({"roi": roi, "orig": o, "nord": n, "delta": d, "pct": pct})
    return rows


def components_table(df: pd.DataFrame, stream: str) -> list[dict]:
    """One row per ROI: within/between r for each pipeline (the components)."""
    g = mean_by_roi(df, stream)
    rows = []
    for roi in ROI_ORDER:
        try:
            o = g.loc[(roi, "original")]
            n = g.loc[(roi, "nordic")]
        except KeyError:
            continue
        rows.append({
            "roi": roi,
            "o_within": o["within_r"], "o_between": o["between_r"],
            "n_within": n["within_r"], "n_between": n["between_r"],
        })
    return rows


def confound_table(nat: pd.DataFrame) -> list[dict]:
    """NAT discriminability per ROI × pipeline × {none, model-08}."""
    g = (nat.groupby(["roi", "pipeline", "confound_model"])["discriminability"]
            .mean())
    rows = []
    for roi in ROI_ORDER:
        try:
            row = {
                "roi": roi,
                "o_none": g.loc[(roi, "original", "none")],
                "o_m8": g.loc[(roi, "original", "model-08")],
                "n_none": g.loc[(roi, "nordic", "none")],
                "n_m8": g.loc[(roi, "nordic", "model-08")],
            }
        except KeyError:
            continue
        rows.append(row)
    return rows


def wilcoxon_stats(df: pd.DataFrame, stream: str) -> dict:
    s = df[df["stream"] == stream]
    piv = s.pivot_table(index=["subject", "session", "roi"],
                        columns="pipeline", values="discriminability").dropna()
    d = (piv["nordic"] - piv["original"]).values
    w = wilcoxon(d)
    return {"n": len(d), "mean": float(d.mean()),
            "pct_pos": 100 * float((d > 0).mean()), "p": float(w.pvalue)}


def headline(df: pd.DataFrame) -> dict:
    disc = df.groupby(["stream", "roi", "pipeline"])["discriminability"].mean().unstack()
    won = int((disc["nordic"] > disc["original"]).sum())
    tot = int(len(disc))
    # largest relative NAT gain
    nat = disc.loc["NAT"]
    rel = ((nat["nordic"] - nat["original"]) / nat["original"].abs())
    roi_best = rel.idxmax()
    return {"won": won, "tot": tot,
            "nat_best_roi": roi_best, "nat_best_pct": 100 * rel.max()}


# ── html helpers ─────────────────────────────────────────────────────────


def _cls(x: float) -> str:
    return "pos" if x >= 0 else "neg"


def _sd(x: float, dp: int = 4) -> str:
    """Signed delta with a Unicode minus for negatives."""
    s = f"{x:+.{dp}f}"
    return s.replace("-", "−")


def _spct(x: float) -> str:
    if not np.isfinite(x):
        return "&mdash;"
    return f"{x:+.0f}%".replace("-", "−")


def fmt_p(p: float) -> str:
    """Format p as m.m×10^e HTML."""
    exp = int(np.floor(np.log10(p)))
    mant = p / (10 ** exp)
    return f"{mant:.1f}&times;10<sup>&minus;{-exp}</sup>"


def disc_rows_html(rows: list[dict]) -> str:
    out = []
    for r in rows:
        out.append(
            f'<tr><td>{r["roi"]}</td><td>{r["orig"]:.4f}</td>'
            f'<td>{r["nord"]:.4f}</td>'
            f'<td class="{_cls(r["delta"])}">{_sd(r["delta"])}</td>'
            f'<td class="{_cls(r["delta"])}">{_spct(r["pct"])}</td></tr>')
    return "\n".join(out)


def components_rows_html(rows: list[dict]) -> str:
    out = []
    for r in rows:
        out.append(
            f'<tr><td>{r["roi"]}</td>'
            f'<td>{r["o_within"]:.4f}</td><td>{r["o_between"]:.4f}</td>'
            f'<td>{r["n_within"]:.4f}</td><td>{r["n_between"]:.4f}</td></tr>')
    return "\n".join(out)


def confound_rows_html(rows: list[dict]) -> str:
    out = []
    for r in rows:
        out.append(
            f'<tr><td>{r["roi"]}</td>'
            f'<td>{r["o_none"]:.4f}</td><td>{r["o_m8"]:.4f}</td>'
            f'<td>{r["n_none"]:.4f}</td><td>{r["n_m8"]:.4f}</td></tr>')
    return "\n".join(out)


def embed_png(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


CSS = """
  :root {
    --paper:#f6f7f9; --surface:#ffffff; --surface-2:#fbfcfd;
    --ink:#1a1e26; --ink-soft:#485162; --ink-faint:#727b8a;
    --rule:#e3e6ec; --rule-strong:#cdd2db;
    --accent:#17629b; --accent-soft:#e9f1f8;
    --original:#8a8f98; --nordic:#17629b;
    --pos:#2f8f63; --neg:#c2503d;
    --serif: Georgia, "Iowan Old Style", "Times New Roman", serif;
    --sans: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    --mono: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
    --maxw: 44rem;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --paper:#0e1116; --surface:#161b22; --surface-2:#1b212a;
      --ink:#e7eaef; --ink-soft:#aab2bd; --ink-faint:#7e8793;
      --rule:#242b34; --rule-strong:#343d48;
      --accent:#5aa6da; --accent-soft:#14222e;
      --original:#7c828c; --nordic:#5aa6da;
      --pos:#4fbe8b; --neg:#e07a68;
    }
  }
  :root[data-theme="light"] {
    --paper:#f6f7f9; --surface:#ffffff; --surface-2:#fbfcfd;
    --ink:#1a1e26; --ink-soft:#485162; --ink-faint:#727b8a;
    --rule:#e3e6ec; --rule-strong:#cdd2db;
    --accent:#17629b; --accent-soft:#e9f1f8;
    --original:#8a8f98; --nordic:#17629b; --pos:#2f8f63; --neg:#c2503d;
  }
  :root[data-theme="dark"] {
    --paper:#0e1116; --surface:#161b22; --surface-2:#1b212a;
    --ink:#e7eaef; --ink-soft:#aab2bd; --ink-faint:#7e8793;
    --rule:#242b34; --rule-strong:#343d48;
    --accent:#5aa6da; --accent-soft:#14222e;
    --original:#7c828c; --nordic:#5aa6da; --pos:#4fbe8b; --neg:#e07a68;
  }

  * { box-sizing: border-box; }
  body {
    margin:0; background:var(--paper); color:var(--ink);
    font-family:var(--sans); font-size:17px; line-height:1.65;
    -webkit-font-smoothing:antialiased;
  }
  .wrap { max-width:var(--maxw); margin:0 auto; padding:0 1.4rem; }

  /* ── masthead ── */
  header.masthead {
    border-bottom:1px solid var(--rule);
    background:linear-gradient(180deg, var(--surface-2), var(--paper));
  }
  .masthead .wrap { padding-top:3.2rem; padding-bottom:2.2rem; }
  .eyebrow {
    font-family:var(--mono); font-size:.72rem; letter-spacing:.18em;
    text-transform:uppercase; color:var(--accent); margin:0 0 1rem;
    display:flex; align-items:center; gap:.7rem;
  }
  .eyebrow::before { content:""; width:1.6rem; height:2px; background:var(--accent); display:inline-block; }
  h1 {
    font-family:var(--serif); font-weight:600; font-size:2.55rem;
    line-height:1.08; letter-spacing:-.01em; margin:0 0 .6rem;
    text-wrap:balance;
  }
  .dek { font-size:1.12rem; color:var(--ink-soft); margin:0 0 1.6rem; max-width:38rem; text-wrap:pretty; }
  .meta { display:flex; flex-wrap:wrap; gap:.5rem 1.6rem; font-size:.86rem; color:var(--ink-faint); font-family:var(--mono); }
  .meta b { color:var(--ink-soft); font-weight:600; }

  /* ── stat row ── */
  .stats { display:grid; grid-template-columns:repeat(4,1fr); gap:1px; background:var(--rule); border:1px solid var(--rule); border-radius:10px; overflow:hidden; margin:2.4rem 0 .4rem; }
  .stat { background:var(--surface); padding:1.1rem 1.15rem 1.2rem; }
  .stat .n { font-family:var(--serif); font-size:1.85rem; font-weight:600; line-height:1; letter-spacing:-.01em; font-variant-numeric:tabular-nums; }
  .stat .n small { font-size:.9rem; color:var(--ink-faint); font-weight:400; }
  .stat .l { font-size:.76rem; color:var(--ink-faint); margin-top:.5rem; line-height:1.35; }
  @media (max-width:640px){ .stats{ grid-template-columns:repeat(2,1fr);} }

  /* ── sections ── */
  main .wrap { padding-top:.5rem; padding-bottom:4rem; }
  section { padding-top:2.9rem; }
  h2 {
    font-family:var(--serif); font-weight:600; font-size:1.62rem; letter-spacing:-.01em;
    margin:0 0 .4rem; padding-bottom:.55rem; border-bottom:1px solid var(--rule);
    display:flex; align-items:baseline; gap:.7rem; text-wrap:balance;
  }
  h2 .num { font-family:var(--mono); font-size:.85rem; color:var(--accent); font-weight:400; }
  h3 { font-family:var(--sans); font-weight:650; font-size:1.02rem; margin:1.8rem 0 .3rem; letter-spacing:.005em; }
  p { margin:.75rem 0; }
  a { color:var(--accent); text-decoration:none; border-bottom:1px solid color-mix(in srgb, var(--accent) 35%, transparent); }
  a:hover { border-bottom-color:var(--accent); }
  strong { font-weight:650; }
  .lead { font-size:1.06rem; color:var(--ink-soft); }

  code, .mono { font-family:var(--mono); font-size:.82em; }
  p code, li code { background:var(--surface); border:1px solid var(--rule); border-radius:4px; padding:.05em .38em; color:var(--ink); }

  ul.clean { margin:.6rem 0; padding-left:1.15rem; }
  ul.clean li { margin:.4rem 0; }
  ul.clean li::marker { color:var(--accent); }

  /* pipeline chips */
  .chip { display:inline-flex; align-items:center; gap:.4em; font-family:var(--mono); font-size:.78em; font-weight:600; padding:.1em .55em; border-radius:20px; border:1px solid var(--rule-strong); white-space:nowrap; }
  .chip::before { content:""; width:.6em; height:.6em; border-radius:50%; }
  .chip.orig::before { background:var(--original); }
  .chip.nord::before { background:var(--nordic); }

  /* ── callout ── */
  .callout { background:var(--accent-soft); border:1px solid color-mix(in srgb, var(--accent) 30%, var(--rule)); border-radius:10px; padding:1.1rem 1.25rem; margin:1.6rem 0; }
  .callout .k { font-family:var(--mono); font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:var(--accent); margin:0 0 .35rem; }
  .callout p { margin:.3rem 0; font-size:.96rem; }

  /* ── tables ── */
  .tblwrap { overflow-x:auto; margin:1.3rem 0; border:1px solid var(--rule); border-radius:10px; background:var(--surface); }
  table { border-collapse:collapse; width:100%; font-size:.86rem; }
  caption { caption-side:top; text-align:left; font-size:.8rem; color:var(--ink-faint); padding:.7rem .95rem .1rem; font-family:var(--mono); }
  th, td { padding:.5rem .85rem; text-align:right; white-space:nowrap; }
  th:first-child, td:first-child { text-align:left; font-family:var(--mono); font-size:.92em; }
  thead th { font-family:var(--sans); font-weight:650; font-size:.76rem; color:var(--ink-soft); border-bottom:1.5px solid var(--rule-strong); text-transform:uppercase; letter-spacing:.04em; }
  tbody td { border-bottom:1px solid var(--rule); font-variant-numeric:tabular-nums; font-family:var(--mono); }
  tbody tr:last-child td { border-bottom:none; }
  tbody tr:hover td { background:var(--surface-2); }
  td.pos { color:var(--pos); }
  td.neg { color:var(--neg); }
  .sub-h th { background:var(--surface-2); font-family:var(--mono); font-size:.7rem; letter-spacing:.06em; color:var(--ink-faint); border-bottom:1px solid var(--rule); }

  /* ── figures ── */
  figure { margin:1.7rem 0; }
  .figcard { background:#ffffff; border:1px solid var(--rule-strong); border-radius:10px; padding:1rem; }
  .figcard img { display:block; width:100%; height:auto; border-radius:4px; }
  figcaption { font-size:.83rem; color:var(--ink-faint); margin-top:.7rem; line-height:1.5; }
  figcaption b { color:var(--ink-soft); font-family:var(--sans); }

  /* legend inline */
  .legend { display:flex; flex-wrap:wrap; gap:.4rem .9rem; font-size:.82rem; color:var(--ink-soft); margin:.4rem 0 0; }

  /* footer */
  footer { border-top:1px solid var(--rule); margin-top:2rem; }
  footer .wrap { padding:1.8rem 1.4rem 3rem; font-size:.82rem; color:var(--ink-faint); }
  footer code { font-family:var(--mono); color:var(--ink-soft); }

  hr.soft { border:none; border-top:1px solid var(--rule); margin:2.6rem 0 0; }
  .fnote { font-size:.82rem; color:var(--ink-faint); }
"""


FIG_NAMES = ("discriminability_per_roi", "nordic_effect_paired",
             "within_between_scatter", "within_between_bars")


def figure_srcs(root: Path, embed: bool) -> dict[str, str]:
    """Map figure name -> img src. Bundled = base64 data URI; linked = rel path."""
    if embed:
        return {n: embed_png(root / "figures" / f"{n}.png") for n in FIG_NAMES}
    return {n: f"figures/{n}.png" for n in FIG_NAMES}


def build_html(root: Path, embed: bool = True) -> str:
    summ, nat = load_inputs(root)
    hl = headline(summ)
    tb_w = wilcoxon_stats(summ, "TB")
    nat_w = wilcoxon_stats(summ, "NAT")

    tb_disc = disc_table(summ, "TB")
    nat_disc = disc_table(summ, "NAT")
    tb_comp = components_table(summ, "TB")
    nat_comp = components_table(summ, "NAT")
    conf = confound_table(nat)

    tb_won = sum(1 for r in tb_disc if r["delta"] > 0)
    nat_won = sum(1 for r in nat_disc if r["delta"] > 0)

    figs = figure_srcs(root, embed)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NORDIC Preprocessing Benchmark — MMMData</title>
<style>{CSS}</style>
</head>
<body>

<header class="masthead">
  <div class="wrap">
    <p class="eyebrow">MMMData · Preprocessing Benchmark</p>
    <h1>Does NORDIC denoising improve neural discriminability?</h1>
    <p class="dek">A within-dataset comparison of standard fMRIPrep against a NORDIC thermal-noise&ndash;denoised variant, measured by how well each pipeline separates repeated stimuli from different ones across two independent data streams.</p>
    <div class="meta">
      <span><b>Dataset</b> Multi-Modal Memory (sub-03/04/05)</span>
      <span><b>Compared</b> <span class="chip orig">original</span> vs <span class="chip nord">NORDIC</span></span>
      <span><b>Generated</b> {GENERATED}</span>
    </div>

    <div class="stats">
      <div class="stat">
        <div class="n">{hl['won']}&thinsp;<small>/&thinsp;{hl['tot']}</small></div>
        <div class="l">ROI&times;stream cells where NORDIC beats original</div>
      </div>
      <div class="stat">
        <div class="n">{hl['nat_best_pct']:+.0f}<small>%</small></div>
        <div class="l">NAT {hl['nat_best_roi'].replace('_', ' ')} gain (largest relative)</div>
      </div>
      <div class="stat">
        <div class="n">{fmt_p(tb_w['p'])}</div>
        <div class="l">TB paired effect (Wilcoxon, n={tb_w['n']})</div>
      </div>
      <div class="stat">
        <div class="n">{fmt_p(nat_w['p'])}</div>
        <div class="l">NAT paired effect (Wilcoxon, n={nat_w['n']})</div>
      </div>
    </div>
  </div>
</header>

<main>
  <div class="wrap">

    <section id="overview">
      <h2><span class="num">01</span> What was compared</h2>
      <p class="lead">Two preprocessing pipelines are run over the identical raw data, and each is scored by a single shared metric on two different task streams.</p>
      <p>The dataset has exactly two fMRI preprocessing variants in <code>derivatives/</code>, each complete across all 3&nbsp;subjects &times; 29&nbsp;sessions (609 preprocessed BOLD runs apiece):</p>
      <ul class="clean">
        <li><span class="chip orig">original</span> &mdash; standard fMRIPrep, no denoising (<code>derivatives/fmriprep/</code>).</li>
        <li><span class="chip nord">NORDIC</span> &mdash; Marchenko&ndash;Pastur PCA thermal-noise removal applied to the BOLD timeseries <em>before</em> fMRIPrep (<code>derivatives/fmriprep_nordic/</code>). Magnitude-only (no phase data exists for this dataset); anatomical/FreeSurfer derivatives are shared, not recomputed.</li>
      </ul>

      <div class="callout">
        <p class="k">The shared metric — discriminability</p>
        <p><strong>discriminability = within-item similarity &minus; between-item similarity.</strong> A pipeline discriminates well when the neural response to <em>repeats of the same stimulus</em> is more similar than the response to <em>different stimuli</em>. Higher is better; it is reported in correlation units (<span class="mono">r</span>).</p>
      </div>

      <p>The same idea is applied two independent ways, so an improvement that shows up in both is unlikely to be an artifact of one analysis choice:</p>
      <div class="tblwrap">
        <table>
          <caption>Two streams, one metric</caption>
          <thead>
            <tr><th>&nbsp;</th><th>TB &mdash; task-based</th><th>NAT &mdash; naturalistic</th></tr>
          </thead>
          <tbody>
            <tr><td>Signal</td><td>GLMsingle &beta; patterns</td><td>ROI mean timeseries</td></tr>
            <tr><td>Item</td><td>NSD image identity</td><td>movie clip</td></tr>
            <tr><td>Similarity</td><td>across runs, same subject</td><td>across subjects (ISC)</td></tr>
            <tr><td>Sessions</td><td>ses-04&ndash;17 (14&times;3 runs)</td><td>ses-19&ndash;28 (10&times;2 runs)</td></tr>
            <tr><td>ROIs</td><td colspan="2" style="text-align:center">Harvard&ndash;Oxford V1, EAC, MT+, IFG, hippocampus (L/R)</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <section id="tb">
      <h2><span class="num">02</span> TB &mdash; task-based encoding</h2>
      <p class="lead">Can each pipeline tell apart the brain-activity pattern evoked by one image from another, when the same images recur across runs?</p>

      <h3>What was done</h3>
      <p>During TBencoding runs, subjects viewed NSD <span class="mono">shared1000</span> images (3&nbsp;s each, image+spoken-word trials; only image rows are used here). Rather than refitting, the benchmark reads the <strong>canonical multi-session GLMsingle fit</strong> already on disk for each subject &times; pipeline (<code>derivatives/glmsingle{{,_nordic}}/</code>) &mdash; the TYPED (FITHRF + GLMdenoise + Ridge) single-trial &beta; maps. A pilot refit was verified bit-for-bit identical (&beta; correlation 1.000000) to these, so the existing fits are reused directly.</p>
      <p>For each subject, session, and ROI, single-trial &beta; patterns are correlated across every trial pair, then split by pair type:</p>
      <ul class="clean">
        <li><strong>Within-item:</strong> same image, <em>different runs</em>, same session.</li>
        <li><strong>Between-item:</strong> different images, <em>different runs</em>, same session.</li>
        <li><strong>Excluded:</strong> same-run pairs (shared run-level noise) and cross-session pairs (scanner drift).</li>
      </ul>
      <p>No external confound regression is applied &mdash; GLMdenoise already learns noise regressors from the data, and adding motion/aCompCor on top biases the fit. Discriminability is the mean within-item&nbsp;<span class="mono">r</span> minus mean between-item&nbsp;<span class="mono">r</span>, averaged over 3&nbsp;subjects &times; 14&nbsp;sessions.</p>

      <h3>Results</h3>
      <p>NORDIC improves discriminability in <strong>{tb_won} of 10 ROIs</strong>. Gains are clearest in visual and motor-language regions (V1, MT+, IFG); auditory cortex (EAC) sits near zero because TB is an image-dominated task, and hippocampus is near zero in both pipelines &mdash; single-trial episodic patterns are hard to decode at this resolution.</p>
      <div class="tblwrap">
        <table>
          <caption>TB mean discriminability (r), across 3 subjects &times; 14 sessions</caption>
          <thead>
            <tr><th>ROI</th><th>original</th><th>NORDIC</th><th>&Delta;</th><th>&Delta; %</th></tr>
          </thead>
          <tbody>
{disc_rows_html(tb_disc)}
          </tbody>
        </table>
      </div>
      <p class="fnote">Where both pipelines hover at essentially zero discriminability (hippocampus), the sign of &Delta; is not meaningful.</p>
    </section>

    <section id="nat">
      <h2><span class="num">03</span> NAT &mdash; naturalistic movie watching</h2>
      <p class="lead">Do different subjects' brains track the <em>same movie</em> more similarly than they track <em>different movies</em> &mdash; and does NORDIC sharpen that agreement?</p>

      <h3>What was done</h3>
      <p>During NATencoding runs all three subjects watched a shared set of film clips. For each session, pipeline, and ROI, the mean timeseries of the ROI is extracted per subject, segmented into per-movie blocks using the events file, and z-scored. Discriminability here is a cross-subject inter-subject correlation (ISC):</p>
      <ul class="clean">
        <li><strong>Within-movie:</strong> correlation between <em>two different subjects</em> watching the <em>same</em> movie (8 movies &times; 3 subject pairs = 24 pairs/session).</li>
        <li><strong>Between-movie:</strong> two different subjects on <em>different</em> movies, truncated to the shorter clip (168 pairs/session).</li>
      </ul>
      <p>Two confound conditions are computed. The <strong>baseline (&ldquo;none&rdquo;)</strong> applies no regression. <strong>Model&nbsp;8</strong> &mdash; the winner of a separate 21-model ISC confound comparison &mdash; residualizes each run's ROI timeseries against <code>6&nbsp;head-motion + WM + CSF + global signal + cosine</code> drift terms before segmentation, reusing the Nastase&nbsp;&amp;&nbsp;Hasson <code>extract_confounds</code> utilities. The unified summary below uses the baseline; the Model&nbsp;8 contrast is broken out separately.</p>

      <h3>Results</h3>
      <p>NORDIC improves discriminability in <strong>{'all 10' if nat_won == 10 else f'{nat_won} of 10'} ROIs</strong>. Auditory cortex (EAC) is the strongest ISC region overall &mdash; movie audio is a powerful shared driver &mdash; and posts the largest absolute gain. Hippocampus shows the largest <em>relative</em> gain ({hl['nat_best_pct']:+.0f}%), echoing the ~3&times; ISFC increase seen in earlier NORDIC validation.</p>
      <div class="tblwrap">
        <table>
          <caption>NAT mean discriminability (r), across 10 sessions &middot; baseline (no confound regression)</caption>
          <thead>
            <tr><th>ROI</th><th>original</th><th>NORDIC</th><th>&Delta;</th><th>&Delta; %</th></tr>
          </thead>
          <tbody>
{disc_rows_html(nat_disc)}
          </tbody>
        </table>
      </div>

      <h3>Confound Model 8 vs baseline</h3>
      <p>Model&nbsp;8's global-signal regression <strong>sharpens cortical discriminability</strong> (most strongly in V1, where it removes shared noise from the between-movie baseline) but <strong>suppresses hippocampus</strong>, whose ISC signal partly rides on the global component. NORDIC keeps its edge over original under both confound conditions.</p>
      <div class="tblwrap">
        <table>
          <caption>NAT discriminability (r) &middot; confound Model 8 vs baseline, per pipeline</caption>
          <thead>
            <tr>
              <th rowspan="2">ROI</th>
              <th colspan="2" style="text-align:center">original</th>
              <th colspan="2" style="text-align:center">NORDIC</th>
            </tr>
          </thead>
          <tbody>
            <tr class="sub-h"><th>ROI</th><th>none</th><th>model-08</th><th>none</th><th>model-08</th></tr>
{confound_rows_html(conf)}
          </tbody>
        </table>
      </div>
    </section>

    <section id="components">
      <h2><span class="num">04</span> Same vs different: the components behind the metric</h2>
      <p class="lead">Discriminability is a <em>difference</em> of two correlations. Splitting it back into its within-item (&ldquo;same&rdquo;) and between-item (&ldquo;different&rdquo;) parts shows the two streams are built on very different footings.</p>
      <p>The single discriminability number can hide how it arises: a small gap between two large correlations looks the same as a large gap over a near-zero floor. The figure and tables below report the raw <span class="mono">within_r</span> and <span class="mono">between_r</span> that get subtracted, for each ROI and pipeline.</p>

      <figure>
        <div class="figcard"><img alt="Grouped bars of within-item and between-item correlation per ROI, original vs NORDIC, TB and NAT panels" src="{figs['within_between_bars']}"></div>
        <figcaption><b>Figure 1 &mdash; Within vs between components per ROI.</b> Solid bars are within-item (&ldquo;same&rdquo;) <span class="mono">r</span>; hatched bars are between-item (&ldquo;different&rdquo;) <span class="mono">r</span>. Grey is <span class="chip orig">original</span>, blue is <span class="chip nord">NORDIC</span>. The visible gap between a solid bar and its hatched neighbour <em>is</em> the discriminability. Two distinct regimes are visible: in <b>NAT</b> (left) the within bar towers over a near-zero between floor, so discriminability &asymp; the raw ISC; in <b>TB</b> (right) within and between are both large and close together, and NORDIC lifts <em>both</em> &mdash; the discriminability gain is a smaller residual on top of an overall rise in pattern correlation.</figcaption>
      </figure>

      <h3>TB &mdash; within and between components (r)</h3>
      <div class="tblwrap">
        <table>
          <caption>TB mean within_r / between_r, across 3 subjects &times; 14 sessions</caption>
          <thead>
            <tr>
              <th rowspan="2">ROI</th>
              <th colspan="2" style="text-align:center">original</th>
              <th colspan="2" style="text-align:center">NORDIC</th>
            </tr>
          </thead>
          <tbody>
            <tr class="sub-h"><th>ROI</th><th>within</th><th>between</th><th>within</th><th>between</th></tr>
{components_rows_html(tb_comp)}
          </tbody>
        </table>
      </div>

      <h3>NAT &mdash; within and between components (r)</h3>
      <div class="tblwrap">
        <table>
          <caption>NAT mean within_r / between_r, across 10 sessions &middot; baseline (no confound regression)</caption>
          <thead>
            <tr>
              <th rowspan="2">ROI</th>
              <th colspan="2" style="text-align:center">original</th>
              <th colspan="2" style="text-align:center">NORDIC</th>
            </tr>
          </thead>
          <tbody>
            <tr class="sub-h"><th>ROI</th><th>within</th><th>between</th><th>within</th><th>between</th></tr>
{components_rows_html(nat_comp)}
          </tbody>
        </table>
      </div>
      <p class="fnote">In NAT the between-movie floor sits near zero (occasionally slightly negative), confirming the naturalistic signal is genuinely stimulus-locked. In TB, even &ldquo;different&rdquo; images share substantial spatial-pattern structure, so both components are elevated.</p>
    </section>

    <section id="figures">
      <h2><span class="num">05</span> Key results, visualized</h2>
      <p class="lead">Each figure places the task-based (TB) and naturalistic (NAT) streams side by side. Throughout, <span class="chip orig">original</span> is grey and <span class="chip nord">NORDIC</span> is blue.</p>

      <figure>
        <div class="figcard"><img alt="Per-ROI bar chart of mean discriminability for original vs NORDIC, TB and NAT panels" src="{figs['discriminability_per_roi']}"></div>
        <figcaption><b>Figure 2 &mdash; Discriminability per ROI.</b> Mean within&minus;between correlation for each Harvard&ndash;Oxford ROI, error bars &plusmn;1&nbsp;SEM across observations. In nearly every ROI the blue (NORDIC) bar sits above the grey (original) one. Note the scale difference between streams: NAT ISC is an order of magnitude larger than TB single-trial &beta; discriminability, with auditory cortex dominating the naturalistic signal.</figcaption>
      </figure>

      <figure>
        <div class="figcard"><img alt="Box plots of paired NORDIC-minus-original discriminability differences per ROI" src="{figs['nordic_effect_paired']}"></div>
        <figcaption><b>Figure 3 &mdash; Paired NORDIC &minus; original effect.</b> Distribution of the per-observation difference (each subject&times;session for TB, each session for NAT), boxes with mean markers. Mass sitting above the zero line means NORDIC helps on that ROI. The naturalistic stream shows the effect is consistently positive; the task-based stream is positive on average with more spread.</figcaption>
      </figure>

      <figure>
        <div class="figcard"><img alt="Scatter of within-item versus between-item correlation colored by pipeline" src="{figs['within_between_scatter']}"></div>
        <figcaption><b>Figure 4 &mdash; Within vs between similarity.</b> Every point is one ROI&times;session (&times;subject for TB); the dashed line is <span class="mono">within = between</span> (zero discriminability). Points above the line have positive discriminability. NORDIC (blue) points sit slightly higher above the diagonal than original (grey), i.e. the same between-item baseline with a stronger within-item signal.</figcaption>
      </figure>
    </section>

    <section id="stats">
      <h2><span class="num">06</span> Is the effect statistically reliable?</h2>
      <p class="lead">Yes &mdash; on both streams, by a large margin.</p>
      <p>Pairing each observation (original vs NORDIC on the same subject, session, and ROI) and testing the signed difference with a Wilcoxon signed-rank test:</p>
      <div class="tblwrap">
        <table>
          <caption>Paired NORDIC &minus; original discriminability &middot; Wilcoxon signed-rank</caption>
          <thead>
            <tr><th>Stream</th><th>n obs</th><th>mean &Delta;r</th><th>% positive</th><th>p</th></tr>
          </thead>
          <tbody>
            <tr><td>TB</td><td>{tb_w['n']}</td><td class="{_cls(tb_w['mean'])}">{_sd(tb_w['mean'])}</td><td>{tb_w['pct_pos']:.1f}%</td><td>{fmt_p(tb_w['p'])}</td></tr>
            <tr><td>NAT</td><td>{nat_w['n']}</td><td class="{_cls(nat_w['mean'])}">{_sd(nat_w['mean'])}</td><td>{nat_w['pct_pos']:.1f}%</td><td>{fmt_p(nat_w['p'])}</td></tr>
          </tbody>
        </table>
      </div>
      <p>Both streams reject the null of no NORDIC effect far below any conventional threshold. NAT is the stronger and cleaner effect ({nat_w['pct_pos']:.0f}% of observations improve); TB is smaller and noisier but still highly significant across its larger sample. Taken together with the {hl['won']}/{hl['tot']} ROI-level tally, the denoising helps and essentially never hurts.</p>
    </section>

    <section id="repro">
      <h2><span class="num">07</span> Reproducing this report</h2>
      <p>All code lives in the <strong>mmmdata</strong> repo under <code>scripts/nordic_benchmark/</code>; all outputs under <code>derivatives/nordic/benchmark/</code>.</p>
      <ul class="clean">
        <li><strong>TB eval</strong> &mdash; <code>tb_eval.py --sub &lt;sub&gt; --pipeline &lt;original|nordic&gt;</code> &rarr; <code>tb/&lt;sub&gt;/&lt;pipeline&gt;/pair_correlations.tsv</code></li>
        <li><strong>NAT eval</strong> &mdash; <code>sbatch nat_eval.sbatch</code> (both pipelines &times; both confound models) &rarr; <code>nat/pair_correlations.tsv</code>, <code>nat/confound_comparison.tsv</code></li>
        <li><strong>Aggregate + figures</strong> &mdash; <code>report.py [--nat-confound-model none|model-08]</code> &rarr; <code>benchmark_summary.tsv</code>, <code>within_between_by_roi.tsv</code>, <code>figures/*.png</code></li>
        <li><strong>This document</strong> &mdash; <code>report_html.py</code> &rarr; <code>benchmark_report.html</code> (linked, references <code>figures/</code>) and <code>benchmark_report.bundled.html</code> (figures base64-embedded, self-contained for sharing). All numbers computed from the TSVs.</li>
      </ul>
      <p class="fnote">Data provenance: numbers and figures were generated from the 2026-07-06 NAT re-run (SLURM job 45097451) and the subsequent <code>report.py</code> aggregation. Statistics computed from <code>benchmark_summary.tsv</code> with <code>scipy.stats.wilcoxon</code>.</p>
    </section>

  </div>
</main>

<footer>
  <div class="wrap">
    NORDIC preprocessing benchmark &middot; Multi-Modal Memory Dataset &middot; generated {GENERATED} from <code>derivatives/nordic/benchmark/</code>. Full methodology and lessons-learned: <code>docs/nordic-and-isc-benchmarks.md</code>.
  </div>
</footer>

</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", type=Path, default=BENCHMARK_OUT)
    ap.add_argument("--mode", choices=["linked", "bundled", "both"], default="both",
                    help="linked = benchmark_report.html referencing figures/*.png; "
                         "bundled = benchmark_report.bundled.html with base64-embedded "
                         "figures (self-contained); both (default) writes each.")
    args = ap.parse_args()

    targets = []
    if args.mode in ("linked", "both"):
        targets.append(("benchmark_report.html", False))
    if args.mode in ("bundled", "both"):
        targets.append(("benchmark_report.bundled.html", True))

    for fname, embed in targets:
        html = build_html(args.output_root, embed=embed)
        out = args.output_root / fname
        out.write_text(html)
        kind = "bundled" if embed else "linked"
        print(f"Wrote {out} ({len(html)/1024:.0f} KB, {kind})")


if __name__ == "__main__":
    main()
