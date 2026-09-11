#!/usr/bin/env python3
"""Figures and summary numbers for the GLM estimator bake-off.

Reads what `glm_bakeoff.py collect` wrote (`scores.tsv`, `harness.json`, and the
per-cell `scores.json` files for the z-threshold voxel counts) and renders the
figures the results summary is built on. Separate from the harness so the
figures can be regenerated without touching a statistical map. Nothing here
recomputes a score: every number is read from the frozen tables.

Usage:
    python glm_bakeoff_figures.py --out-dir DIR [--bakeoff-dir DIR]

`--bakeoff-dir` defaults to `<output_dir>/glm_bakeoff` from the dataset config.
Alongside the PNGs the script writes `bakeoff_summary.json` with every marginal,
paired-difference and win-rate the figures show, so a write-up can cite the
same numbers the figures draw.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))

MODELS = ["floc", "motor", "tbrepetition"]
MODEL_LABEL = {"floc": "fLoc (block, 6 runs, odd/even)",
               "motor": "motor (block, 2 runs, run-01/run-02)",
               "tbrepetition": "TB encoding first-vs-later (event, 42 runs)"}
ENGINES = ["nilearn-ols", "nilearn-ar1", "remlfit-arma11"]
ENGINE_LABEL = {"nilearn-ols": "OLS", "nilearn-ar1": "AR(1)",
                "remlfit-arma11": "REMLfit"}
CONFOUNDS = ["motion6", "motion24", "acompcor"]
CONF_LABEL = {"motion6": "motion6", "motion24": "motion24",
              "acompcor": "+aCompCor"}
HRFS = ["spm", "spmderiv", "voxelwise"]
HRF_LABEL = {"spm": "SPM", "spmderiv": "SPM+deriv", "voxelwise": "per-voxel"}
FACTORS = [("engine", ENGINES, ENGINE_LABEL),
           ("confounds", CONFOUNDS, CONF_LABEL),
           ("hrf", HRFS, HRF_LABEL)]
REFERENCE = {"hrf": "spm", "confounds": "acompcor", "engine": "nilearn-ols"}

# Categorical slots 1-3 of the reference palette (validated all-pairs);
# subjects are drawn in neutral ink and labelled directly.
SERIES = {"nilearn-ols": "#2a78d6", "nilearn-ar1": "#eb6834",
          "remlfit-arma11": "#1baf7a"}
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#9a9891"
GRID = "#e6e5e1"
ACCENT = "#2a78d6"
ALT = "#eb6834"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": INK3, "axes.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK2, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 120,
    "savefig.dpi": 160, "savefig.facecolor": "white",
})


def load(bakeoff_dir: Path):
    harness = json.loads((bakeoff_dir / "harness.json").read_text())
    s = pd.read_csv(bakeoff_dir / "scores.tsv", sep="\t",
                    dtype={"subject": str})
    wide = s.pivot_table(index=["subject", "model", "hrf", "confounds",
                                "engine", "cell", "contrast"],
                         columns="metric", values="value").reset_index()
    for m, ns in harness["n_sets"].items():
        cols = [f"dice@{n}" for n in ns]
        sel = wide.model == m
        wide.loc[sel, "dice_family"] = wide.loc[sel, cols].mean(axis=1)
    # z-threshold voxel counts live only in the per-cell scores.json
    rows = []
    for js in bakeoff_dir.glob("sub-*/model-*/scores.json"):
        d = json.loads(js.read_text())
        for c, v in d["contrasts"].items():
            n = v.get("n@z3.1")
            if n is None:
                continue
            rows.append({"subject": d["subject"], "model": d["model"],
                         "cell": d["cell"], "contrast": c,
                         "n_z_half1": n[0], "n_z_half2": n[1]})
    nz = pd.DataFrame(rows)
    wide = wide.merge(nz, on=["subject", "model", "cell", "contrast"],
                      how="left")
    return wide, harness


def factorial(wide):
    return wide[wide.engine.isin(ENGINES)].copy()


def paired_diffs(fac, model, factor, a, b, metric):
    """Per-pair difference a - b over every other-factor x subject x contrast cell."""
    other = [f for f, _, _ in FACTORS if f != factor]
    keys = ["subject", "contrast"] + other
    sub = fac[fac.model == model]
    pa = sub[sub[factor] == a].set_index(keys)[metric]
    pb = sub[sub[factor] == b].set_index(keys)[metric]
    d = (pa - pb).dropna()
    return d


def label_line_end(ax, x, y, text, color, dx=0.06):
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset fontsize",
                va="center", ha="left", fontsize=7.5, color=color)


def label_line_ends(ax, x, items, min_gap_frac=0.055):
    """Direct-label several line ends at the same x, pushing labels apart
    vertically when they would collide. items = [(y, text, color), ...]."""
    ys = np.array([y for y, _, _ in items], dtype=float)
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    order = np.argsort(ys)
    placed = ys[order].copy()
    for k in range(1, len(placed)):
        if placed[k] - placed[k - 1] < gap:
            placed[k] = placed[k - 1] + gap
    # recentre so the block does not drift upward
    placed -= (placed.mean() - ys[order].mean())
    for k, idx in enumerate(order):
        y, text, color = items[idx]
        ax.annotate(text, (x, y), xytext=(x + 0.08, placed[k]),
                    textcoords="data", va="center", ha="left", fontsize=7.5,
                    color=color)


# --------------------------------------------------------------------------
def fig_marginals(fac, metric, ylabel, path, summary):
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.2), sharex=False)
    for j, model in enumerate(MODELS):
        sub = fac[fac.model == model]
        for i, (factor, levels, lab) in enumerate(FACTORS):
            ax = axes[i, j]
            x = np.arange(len(levels))
            per_sub = (sub.groupby(["subject", factor])[metric].mean()
                       .unstack(factor)[levels])
            mean = per_sub.mean(axis=0)
            ends = []
            for subj, row in per_sub.iterrows():
                ax.plot(x, row.values, color=INK3, lw=1.2, marker="o", ms=3.5,
                        zorder=2)
                ends.append((row.values[-1], f"sub-{subj}", INK2))
            ax.plot(x, mean.values, color=ACCENT, lw=2.4, marker="o", ms=5,
                    zorder=3)
            ends.append((mean.values[-1], "mean", ACCENT))
            ax.set_xticks(x, [lab[l] for l in levels])
            ax.set_xlim(-0.35, len(levels) - 1 + 0.9)
            ax.grid(axis="y")
            label_line_ends(ax, x[-1], ends)
            ax.tick_params(axis="x", length=0)
            if i == 0:
                ax.set_title(MODEL_LABEL[model])
            if j == 0:
                ax.set_ylabel(f"{ylabel}\nby {factor}")
            summary.setdefault("marginals", {}).setdefault(metric, {}) \
                .setdefault(model, {})[factor] = {
                    "mean": {l: round(float(mean[l]), 4) for l in levels},
                    "per_subject": {s: {l: round(float(v), 4)
                                        for l, v in r.items()}
                                    for s, r in per_sub.iterrows()}}
    fig.suptitle(f"Marginal {ylabel} by factor level, averaged over the other "
                 "two factors and all contrasts.\nThin lines = subjects, "
                 "thick = mean. Columns share nothing: motor and fLoc "
                 "split-half values are not comparable (motor has one block "
                 "order).", fontsize=9, color=INK2, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fig_engine_pairs(fac, path, summary):
    comps = [("nilearn-ols", "nilearn-ar1"), ("nilearn-ols", "remlfit-arma11"),
             ("nilearn-ar1", "remlfit-arma11")]
    metrics = [("dice_family", "Dice family (top-N set)"),
               ("r", "split-half map r")]
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 8.0))
    rng = np.random.default_rng(0)
    out = summary.setdefault("engine_pairs", {})
    for i, model in enumerate(MODELS):
        for j, (metric, mlab) in enumerate(metrics):
            ax = axes[i, j]
            for k, (a, b) in enumerate(comps):
                d = paired_diffs(fac, model, "engine", a, b, metric)
                y = k + rng.uniform(-0.18, 0.18, len(d))
                ax.scatter(d.values, y, s=12, color=INK3, alpha=0.65,
                           linewidths=0, zorder=2)
                m = float(d.mean())
                win = float((d > 0).mean())
                ax.scatter([m], [k], s=60, color=ACCENT, zorder=4,
                           edgecolor="white", linewidth=1.2)
                ax.annotate(f"{m:+.3f}   {win:.0%} > 0   (n={len(d)})",
                            (1.0, k), xycoords=("axes fraction", "data"),
                            xytext=(-4, 9), textcoords="offset points",
                            ha="right", fontsize=7.5, color=INK2)
                out.setdefault(model, {}).setdefault(metric, {})[
                    f"{ENGINE_LABEL[a]}-{ENGINE_LABEL[b]}"] = {
                        "mean": round(m, 4), "frac_positive": round(win, 3),
                        "n": int(len(d))}
            ax.axvline(0, color=INK2, lw=0.9, ls="--", zorder=1)
            ax.set_yticks(range(3), [f"{ENGINE_LABEL[a]} − {ENGINE_LABEL[b]}"
                                     for a, b in comps])
            ax.set_ylim(-0.6, 2.6)
            ax.invert_yaxis()
            ax.grid(axis="x")
            ax.tick_params(axis="y", length=0)
            if i == 0:
                ax.set_title(f"paired difference in {mlab}")
            if j == 0:
                ax.set_ylabel(MODEL_LABEL[model].split(" (")[0], color=INK)
            if i == 2:
                ax.set_xlabel("difference (positive = first engine more "
                              "reliable)")
    fig.suptitle("Engine comparison: each dot is one other-factor × subject × "
                 "contrast cell in which two engines are compared head to head; "
                 "the blue marker is the mean.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fig_dice_vs_n(fac, harness, path, summary):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    out = summary.setdefault("dice_vs_n", {})
    for j, model in enumerate(MODELS):
        ax = axes[j]
        ns = harness["n_sets"][model]
        sub = fac[fac.model == model]
        for eng in ENGINES:
            g = sub[sub.engine == eng]
            ys = [float(g[f"dice@{n}"].mean()) for n in ns]
            yz = float(g["dice@z3.1"].mean())
            xs = np.arange(len(ns))
            ax.plot(xs, ys, color=SERIES[eng], lw=2, marker="o", ms=5,
                    label=ENGINE_LABEL[eng])
            ax.scatter([len(ns) + 0.6], [yz], color=SERIES[eng], s=42,
                       marker="D", zorder=3)
            label_line_end(ax, len(ns) + 0.6, yz, ENGINE_LABEL[eng],
                           SERIES[eng], dx=0.5)
            out.setdefault(model, {})[ENGINE_LABEL[eng]] = {
                "dice_at_N": dict(zip(map(str, ns), [round(y, 4) for y in ys])),
                "dice_at_z3.1": round(yz, 4)}
        ax.axvline(len(ns) - 0.2, color=GRID, lw=1)
        ax.set_xticks(list(range(len(ns))) + [len(ns) + 0.6],
                      [f"top {n:,}" for n in ns] + ["z > 3.1"])
        ax.tick_params(axis="x", labelrotation=30, length=0)
        ax.set_xlim(-0.4, len(ns) + 2.4)
        ax.grid(axis="y")
        ax.set_title(MODEL_LABEL[model].split(" (")[0])
        if j == 0:
            ax.set_ylabel("split-half Dice (mean over cells)")
            ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("Dice at each pre-registered N (lines) and at the z > 3.1 "
                 "threshold (diamonds), by engine. The top-N ranking holds at "
                 "every N; the thresholded mask ranks fLoc the other way.",
                 fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fig_calibration(fac, path, summary):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    rng = np.random.default_rng(1)
    out = summary.setdefault("z_threshold_counts", {})
    for j, model in enumerate(MODELS):
        ax = axes[j]
        sub = fac[fac.model == model]
        for k, eng in enumerate(ENGINES):
            g = sub[sub.engine == eng]
            n = pd.concat([g.n_z_half1, g.n_z_half2]).dropna().values
            x = k + rng.uniform(-0.22, 0.22, len(n))
            ax.scatter(x, n, s=9, color=SERIES[eng], alpha=0.5, linewidths=0)
            med = float(np.median(n))
            ax.hlines(med, k - 0.3, k + 0.3, color=INK, lw=2, zorder=4)
            ax.annotate(f"median {med:,.0f}", (k, med), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=7.5,
                        color=INK2)
            out.setdefault(model, {})[ENGINE_LABEL[eng]] = {
                "median_voxels_above_z": round(med, 1),
                "mean_voxels_above_z": round(float(n.mean()), 1)}
        ax.set_xticks(range(3), [ENGINE_LABEL[e] for e in ENGINES])
        ax.tick_params(axis="x", length=0)
        ax.set_yscale("log")
        ax.grid(axis="y", which="both")
        ax.set_title(MODEL_LABEL[model].split(" (")[0])
        if j == 0:
            ax.set_ylabel("voxels above z 3.1 per half (log)")
    fig.suptitle("Calibration: how many voxels each engine puts above z 3.1. "
                 "Same data, same design; only the noise model differs. "
                 "OLS z is anticonservative under autocorrelation.",
                 fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fig_standalone(wide, path, summary):
    ref = wide[(wide.hrf == REFERENCE["hrf"]) &
               (wide.confounds == REFERENCE["confounds"]) &
               (wide.engine == REFERENCE["engine"])]
    arms = [("floc", "glmsingle", "GLMsingle on fLoc blocks"),
            ("tbrepetition", "glmsingle-betas",
             "Welch t over GLMsingle per-trial betas")]
    metrics = [("dice_family", "Dice family"), ("r", "split-half r")]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2),
                             gridspec_kw={"height_ratios": [4, 1.4]})
    out = summary.setdefault("standalone_arms", {})
    behind = []  # (arm label, metric, n_behind, n)
    for i, (model, hrf, alab) in enumerate(arms):
        alt = wide[(wide.model == model) & (wide.hrf == hrf)]
        r = ref[ref.model == model]
        keys = ["subject", "contrast"]
        m = (r.set_index(keys)[["dice_family", "r"]]
             .join(alt.set_index(keys)[["dice_family", "r"]],
                   lsuffix="_ref", rsuffix="_arm").dropna().sort_index())
        for j, (metric, mlab) in enumerate(metrics):
            ax = axes[i, j]
            labels = [f"sub-{s}  {c}" for s, c in m.index]
            y = np.arange(len(m))
            a = m[f"{metric}_ref"].values
            b = m[f"{metric}_arm"].values
            ax.hlines(y, b, a, color=INK3, lw=1.4, zorder=1)
            ax.scatter(a, y, color=ACCENT, s=36, zorder=3,
                       label="reference cell (SPM, +aCompCor, OLS)")
            ax.scatter(b, y, color=ALT, s=36, zorder=3, label=alab)
            ax.set_yticks(y, labels)
            ax.invert_yaxis()
            ax.tick_params(axis="y", length=0, labelsize=7.5)
            ax.grid(axis="x")
            ax.set_xlim(0, max(a.max(), b.max()) * 1.12)
            if i == 0:
                ax.set_title(mlab)
            if j == 0:
                ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0 if i else
                                                            1.04),
                          fontsize=7.5, ncol=2)
            behind.append((alab, mlab, int((b < a).sum()), len(a)))
            out.setdefault(alab, {})[metric] = {
                f"sub-{s} {c}": {"reference": round(float(ra), 4),
                                 "arm": round(float(rb), 4)}
                for (s, c), ra, rb in zip(m.index, a, b)}
    short = {"Dice family": "Dice", "split-half r": "r"}
    tally = "\n".join(
        f"{alab}: behind the reference in " + ", ".join(
            f"{nb}/{n} cells on {short[mlab]}"
            for a2, mlab, nb, n in behind if a2 == alab)
        for alab in dict.fromkeys(a for a, _, _, _ in behind))
    summary["standalone_arms_behind"] = [
        {"arm": alab, "metric": mlab, "behind": nb, "n": n}
        for alab, mlab, nb, n in behind]
    fig.suptitle("Standalone GLMsingle arms against the winning factorial cell, "
                 "per subject and contrast.\n" + tally, fontsize=9,
                 color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fig_subject_contrast_heat(fac, path, summary):
    """OLS - AR(1) Dice-family difference, averaged over hrf x confounds, per
    subject x contrast: the 'every subject, every contrast' claim drawn."""
    d = paired_diffs_all(fac, "engine", "nilearn-ols", "nilearn-ar1",
                         "dice_family")
    d2 = paired_diffs_all(fac, "confounds", "acompcor", "motion6",
                          "dice_family")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    out = summary.setdefault("per_subject_contrast", {})
    for ax, (tab, title, key) in zip(axes, [
            (d, "OLS − AR(1), Dice family", "ols_minus_ar1"),
            (d2, "+aCompCor − motion6, Dice family", "acompcor_minus_motion6")]):
        vmax = float(np.abs(tab.values).max())
        im = ax.imshow(tab.values, cmap="RdBu", vmin=-vmax, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(tab.shape[1]), tab.columns, rotation=35,
                      ha="right", fontsize=7.5)
        ax.set_yticks(range(tab.shape[0]), [f"sub-{s}" for s in tab.index])
        ax.tick_params(length=0)
        for (yi, xi), v in np.ndenumerate(tab.values):
            ax.text(xi, yi, f"{v:+.3f}", ha="center", va="center",
                    fontsize=7, color=INK if abs(v) < vmax * 0.6 else "white")
        ax.set_title(title)
        for sp in ax.spines.values():
            sp.set_visible(False)
        out[key] = {f"sub-{s}": {c: round(float(v), 4) for c, v in row.items()}
                    for s, row in tab.iterrows()}
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02).set_label(
        "Dice difference", color=INK2)
    fig.suptitle("Mean paired difference per subject × contrast (over HRF × "
                 "confounds). Blue = the proposed level is more reliable.",
                 fontsize=9, color=INK2, y=1.04)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def paired_diffs_all(fac, factor, a, b, metric):
    other = [f for f, _, _ in FACTORS if f != factor]
    keys = ["subject", "contrast", "model"] + other
    pa = fac[fac[factor] == a].set_index(keys)[metric]
    pb = fac[fac[factor] == b].set_index(keys)[metric]
    d = (pa - pb).dropna().reset_index()
    order = [c for m in MODELS
             for c in sorted(fac[fac.model == m].contrast.unique())]
    return (d.groupby(["subject", "contrast"])[metric].mean().unstack()
            [order])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bakeoff-dir", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if args.bakeoff_dir is None:
        from core.config import load_config
        cfg = load_config()
        bakeoff_dir = Path(cfg["paths"]["output_dir"]) / "glm_bakeoff"
    else:
        bakeoff_dir = Path(args.bakeoff_dir)
    if not (bakeoff_dir / "scores.tsv").exists():
        sys.exit(f"no scores.tsv under {bakeoff_dir}; run "
                 "`glm_bakeoff.py collect` first")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wide, harness = load(bakeoff_dir)
    fac = factorial(wide)
    summary = {"bakeoff_dir": str(bakeoff_dir),
               "harness_sha": harness.get("sha256"),
               "n_cells_scored": int(wide.groupby(["subject", "cell"]).ngroups),
               "n_factorial_rows": int(len(fac))}

    fig_marginals(fac, "dice_family", "Dice family",
                  out_dir / "bakeoff-marginals-dice.png", summary)
    fig_marginals(fac, "r", "split-half r",
                  out_dir / "bakeoff-marginals-r.png", summary)
    fig_engine_pairs(fac, out_dir / "bakeoff-engine-pairs.png", summary)
    fig_subject_contrast_heat(fac, out_dir / "bakeoff-subject-contrast.png",
                              summary)
    fig_dice_vs_n(fac, harness, out_dir / "bakeoff-dice-vs-n.png", summary)
    fig_calibration(fac, out_dir / "bakeoff-calibration.png", summary)
    fig_standalone(wide, out_dir / "bakeoff-standalone-arms.png", summary)

    # confound and HRF paired differences, for the write-up
    for factor, a, b in [("confounds", "acompcor", "motion6"),
                         ("confounds", "motion24", "motion6"),
                         ("hrf", "spmderiv", "spm"),
                         ("hrf", "voxelwise", "spm")]:
        for model in MODELS:
            for metric in ["dice_family", "r"]:
                d = paired_diffs(fac, model, factor, a, b, metric)
                per_sub = d.groupby(level="subject").mean()
                summary.setdefault("factor_pairs", {}).setdefault(
                    f"{a}-{b}", {}).setdefault(model, {})[metric] = {
                        "mean": round(float(d.mean()), 4),
                        "frac_positive": round(float((d > 0).mean()), 3),
                        "n": int(len(d)),
                        "per_subject_mean": {s: round(float(v), 4)
                                             for s, v in per_sub.items()}}
    p = out_dir / "bakeoff_summary.json"
    p.write_text(json.dumps(summary, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
