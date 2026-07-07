#!/usr/bin/env python3
"""Aggregate TB + NAT pair_correlations.tsv files into the unified benchmark
summary, and produce figures comparing original vs NORDIC pipelines.

Inputs (anything missing is skipped with a warning):
  derivatives/nordic/benchmark/tb/{sub}/{pipeline}/pair_correlations.tsv  [up to 6]
  derivatives/nordic/benchmark/nat/pair_correlations.tsv                  [1]

Outputs:
  derivatives/nordic/benchmark/benchmark_summary.tsv
  derivatives/nordic/benchmark/within_between_by_roi.tsv
  derivatives/nordic/benchmark/figures/discriminability_per_roi.png
  derivatives/nordic/benchmark/figures/nordic_effect_paired.png
  derivatives/nordic/benchmark/figures/within_between_scatter.png
  derivatives/nordic/benchmark/figures/within_between_bars.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from shared import BENCHMARK_OUT, ROI_KEYS, SUBJECTS, SUMMARY_COLS  # noqa: E402

ROI_ORDER = [f"{r}_{h}" for r, h in ROI_KEYS]
PIPELINE_COLORS = {"original": "#888888", "nordic": "#1f77b4"}


def collect(output_root: Path, nat_confound_model: str = "none") -> pd.DataFrame:
    """Concat all available pair_correlations.tsv files into one DataFrame.

    The NAT file may carry a `confound_model` column (multiple models per row);
    to keep the unified summary well-defined we select a single model
    (`nat_confound_model`, default "none") before concatenating.
    """
    frames = []
    # TB: one file per (subject, pipeline)
    for sub in SUBJECTS:
        for pipeline in ("original", "nordic"):
            p = output_root / "tb" / sub / pipeline / "pair_correlations.tsv"
            if p.exists():
                frames.append(pd.read_csv(p, sep="\t"))
            else:
                print(f"  missing TB: {p}")
    # NAT: single aggregate file (may contain multiple confound models)
    nat_p = output_root / "nat" / "pair_correlations.tsv"
    if nat_p.exists():
        nat_df = pd.read_csv(nat_p, sep="\t")
        if "confound_model" in nat_df.columns:
            avail = sorted(nat_df["confound_model"].unique())
            if nat_confound_model not in avail:
                raise ValueError(
                    f"NAT confound model {nat_confound_model!r} not in {avail}; "
                    "pass --nat-confound-model")
            nat_df = nat_df[nat_df["confound_model"] == nat_confound_model].drop(
                columns=["confound_model"])
            print(f"  NAT: using confound_model={nat_confound_model!r} (available: {avail})")
        frames.append(nat_df)
    else:
        print(f"  missing NAT: {nat_p}")

    if not frames:
        raise FileNotFoundError("No pair_correlations.tsv inputs found.")
    df = pd.concat(frames, ignore_index=True)
    df["roi"] = pd.Categorical(df["roi"], categories=ROI_ORDER, ordered=True)
    return df


def write_summary(df: pd.DataFrame, output_root: Path) -> Path:
    out = output_root / "benchmark_summary.tsv"
    df[SUMMARY_COLS].to_csv(out, sep="\t", index=False)
    print(f"Wrote {out} ({len(df)} rows)")
    return out


def within_between_by_roi(df: pd.DataFrame) -> pd.DataFrame:
    """Mean within_r / between_r / discriminability per stream × ROI × pipeline.

    This is the component breakdown behind the discriminability metric: the
    'same-item' (within) and 'different-item' (between) correlations that get
    subtracted. Rows are ordered by (stream, ROI_ORDER, pipeline).
    """
    g = (df.groupby(["stream", "roi", "pipeline"], observed=True)
           [["within_r", "between_r", "discriminability"]]
           .mean().round(5).reset_index())
    g["roi"] = pd.Categorical(g["roi"], categories=ROI_ORDER, ordered=True)
    return g.sort_values(["stream", "roi", "pipeline"]).reset_index(drop=True)


def write_within_between_table(df: pd.DataFrame, output_root: Path) -> Path:
    tbl = within_between_by_roi(df)
    out = output_root / "within_between_by_roi.tsv"
    tbl.to_csv(out, sep="\t", index=False)
    print(f"Wrote {out} ({len(tbl)} rows)")
    return out


# ── figures ─────────────────────────────────────────────────────────────

def figure_per_roi(df: pd.DataFrame, output_root: Path) -> Path:
    """Per-ROI bar plot of mean discriminability {original, nordic} × {TB, NAT}."""
    streams = sorted(df["stream"].unique())
    if not streams:
        return None
    fig, axes = plt.subplots(1, len(streams), figsize=(8 * len(streams), 5),
                             sharey=True, squeeze=False)
    for ax, stream in zip(axes[0], streams):
        sub = df[df["stream"] == stream]
        means = sub.groupby(["roi", "pipeline"], observed=True)["discriminability"].mean().unstack()
        sems = sub.groupby(["roi", "pipeline"], observed=True)["discriminability"].sem().unstack()
        x = np.arange(len(ROI_ORDER))
        w = 0.4
        for i, pipe in enumerate(("original", "nordic")):
            if pipe not in means.columns:
                continue
            vals = means[pipe].reindex(ROI_ORDER).values
            errs = sems[pipe].reindex(ROI_ORDER).values
            ax.bar(x + (i - 0.5) * w, vals, w, yerr=errs, capsize=3,
                   label=pipe, color=PIPELINE_COLORS[pipe])
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(ROI_ORDER, rotation=45, ha="right")
        ax.set_title(f"{stream} discriminability (within − between)")
        ax.set_ylabel("r")
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = output_root / "figures" / "discriminability_per_roi.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def figure_paired_delta(df: pd.DataFrame, output_root: Path) -> Path:
    """Per-ROI paired (NORDIC − original) discriminability across observations."""
    rows = []
    # Pair within (stream, subject, session): one observation per pair
    for keys, g in df.groupby(["stream", "subject", "session", "roi"], observed=True):
        if len(g) != 2:
            continue
        orig = g.loc[g["pipeline"] == "original", "discriminability"].values
        nord = g.loc[g["pipeline"] == "nordic", "discriminability"].values
        if len(orig) != 1 or len(nord) != 1:
            continue
        rows.append({
            "stream": keys[0], "subject": keys[1], "session": keys[2], "roi": keys[3],
            "delta": float(nord[0] - orig[0]),
        })
    deltas = pd.DataFrame(rows)
    if deltas.empty:
        print("  no paired observations for delta plot")
        return None

    streams = sorted(deltas["stream"].unique())
    fig, axes = plt.subplots(1, len(streams), figsize=(8 * len(streams), 5),
                             sharey=True, squeeze=False)
    for ax, stream in zip(axes[0], streams):
        sub = deltas[deltas["stream"] == stream]
        positions = np.arange(len(ROI_ORDER))
        per_roi = [sub.loc[sub["roi"] == r, "delta"].dropna().values for r in ROI_ORDER]
        ax.boxplot(per_roi, positions=positions, widths=0.6, showmeans=True)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(ROI_ORDER, rotation=45, ha="right")
        ax.set_title(f"{stream}: discriminability(NORDIC − original)")
        ax.set_ylabel("Δ r")
    fig.tight_layout()
    out = output_root / "figures" / "nordic_effect_paired.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def figure_within_between_scatter(df: pd.DataFrame, output_root: Path) -> Path:
    """Per-ROI scatter of within_r vs between_r, colored by pipeline."""
    streams = sorted(df["stream"].unique())
    fig, axes = plt.subplots(len(streams), 1,
                             figsize=(10, 5 * len(streams)),
                             squeeze=False)
    for ax, stream in zip(axes[:, 0], streams):
        sub = df[df["stream"] == stream]
        for pipe in ("original", "nordic"):
            d = sub[sub["pipeline"] == pipe]
            ax.scatter(d["between_r"], d["within_r"],
                       label=pipe, color=PIPELINE_COLORS[pipe], alpha=0.6, s=20)
        lo = min(sub["between_r"].min(skipna=True), sub["within_r"].min(skipna=True))
        hi = max(sub["between_r"].max(skipna=True), sub["within_r"].max(skipna=True))
        ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=0.8, alpha=0.6,
                label="x=y")
        ax.set_xlabel("between-item r")
        ax.set_ylabel("within-item r")
        ax.set_title(f"{stream}: within vs between")
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = output_root / "figures" / "within_between_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def figure_within_between_bars(df: pd.DataFrame, output_root: Path) -> Path:
    """Per-ROI grouped bars of the within_r and between_r components.

    For each ROI, four bars: {original, NORDIC} × {between, within}. Within is
    drawn solid and between hatched/lighter, so the visible gap between the two
    bars of a cluster IS the discriminability. Makes explicit that TB has a
    small gap between two large correlations while NAT has a large within over a
    near-zero between floor.
    """
    streams = sorted(df["stream"].unique())
    if not streams:
        return None
    tbl = within_between_by_roi(df)
    fig, axes = plt.subplots(1, len(streams), figsize=(9 * len(streams), 5.2),
                             squeeze=False)
    x = np.arange(len(ROI_ORDER))
    w = 0.2
    # cluster offsets: original-between, original-within, nordic-between, nordic-within
    specs = [
        ("original", "between_r", -1.5 * w, PIPELINE_COLORS["original"], True),
        ("original", "within_r",  -0.5 * w, PIPELINE_COLORS["original"], False),
        ("nordic",   "between_r",  0.5 * w, PIPELINE_COLORS["nordic"],   True),
        ("nordic",   "within_r",   1.5 * w, PIPELINE_COLORS["nordic"],   False),
    ]
    for ax, stream in zip(axes[0], streams):
        sub = tbl[tbl["stream"] == stream]
        for pipe, metric, off, color, is_between in specs:
            vals = (sub[sub["pipeline"] == pipe]
                    .set_index("roi")[metric].reindex(ROI_ORDER).values)
            ax.bar(x + off, vals, w, color=color,
                   alpha=0.45 if is_between else 1.0,
                   hatch="///" if is_between else None,
                   edgecolor=color, linewidth=0.6)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(ROI_ORDER, rotation=45, ha="right")
        ax.set_title(f"{stream}: within vs between r by ROI")
        ax.set_ylabel("r")
    # shared legend describing the 2×2 encoding
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=PIPELINE_COLORS["original"], label="original — within"),
        Patch(facecolor=PIPELINE_COLORS["original"], alpha=0.45, hatch="///",
              edgecolor=PIPELINE_COLORS["original"], label="original — between"),
        Patch(facecolor=PIPELINE_COLORS["nordic"], label="NORDIC — within"),
        Patch(facecolor=PIPELINE_COLORS["nordic"], alpha=0.45, hatch="///",
              edgecolor=PIPELINE_COLORS["nordic"], label="NORDIC — between"),
    ]
    axes[0][0].legend(handles=handles, loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    out = output_root / "figures" / "within_between_bars.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", type=Path, default=BENCHMARK_OUT)
    ap.add_argument("--nat-confound-model", default="none",
                    help="Which NAT confound model to include in the unified "
                         "summary when the NAT file has multiple (default: none).")
    args = ap.parse_args()

    print("Collecting pair_correlations.tsv inputs...")
    df = collect(args.output_root, nat_confound_model=args.nat_confound_model)
    print(f"  {len(df)} rows total")
    print(f"  streams:    {sorted(df['stream'].unique())}")
    print(f"  pipelines:  {sorted(df['pipeline'].unique())}")
    print(f"  subjects:   {sorted(df['subject'].unique())}")

    write_summary(df, args.output_root)
    write_within_between_table(df, args.output_root)

    print("\nFigures:")
    figure_per_roi(df, args.output_root)
    figure_paired_delta(df, args.output_root)
    figure_within_between_scatter(df, args.output_root)
    figure_within_between_bars(df, args.output_root)

    print("\nDone.")


if __name__ == "__main__":
    main()
