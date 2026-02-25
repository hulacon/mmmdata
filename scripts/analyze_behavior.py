#!/usr/bin/env python3
"""Run standard behavioral analyses for the MMMData project.

Usage
-----
    python analyze_behavior.py summary
    python analyze_behavior.py accuracy --by enCon reCon
    python analyze_behavior.py dprime --by subject
    python analyze_behavior.py learning --metric accuracy
    python analyze_behavior.py rt
    python analyze_behavior.py sme
    python analyze_behavior.py final
    python analyze_behavior.py all --output-dir derivatives/behavioral_analysis/

Each subcommand produces TSV summary tables and (optionally) figures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src/python to path so we can import behavioral package
_CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE_ROOT / "src" / "python"))

from behavioral import io, preprocessing, accuracy, rt, learning, encoding, final_session, plotting
from behavioral.constants import ENCON_LABELS, RECON_LABELS, DERIVATIVES_DIR

DEFAULT_BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")


def _ensure_output_dir(output_dir: Path) -> None:
    """Create output directories if they don't exist."""
    (output_dir / "group").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    for sub in ("03", "04", "05"):
        (output_dir / f"sub-{sub}").mkdir(parents=True, exist_ok=True)


def _save_tsv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to TSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, na_rep="n/a")
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_summary(args):
    """Print dataset overview: file counts, trial counts, session coverage."""
    root = Path(args.bids_root)
    print("=== MMMData Behavioral Data Summary ===\n")

    tb = io.find_tb2afc_files(root, args.subjects)
    enc = io.find_encoding_files(root, args.subjects)
    ret = io.find_retrieval_files(root, args.subjects)
    fin = io.find_fin2afc_files(root, args.subjects)
    tl = io.find_fintimeline_files(root, args.subjects)

    print(f"TB2AFC files:      {len(tb)}")
    print(f"TBencoding files:  {len(enc)}")
    print(f"TBretrieval files: {len(ret)}")
    print(f"FIN2AFC files:     {len(fin)}")
    print(f"FINtimeline files: {len(tl)}")

    if tb:
        df = io.load_tb2afc(root, args.subjects)
        print(f"\nTB2AFC total trials: {len(df)}")
        print(f"  Subjects: {sorted(df['subject'].unique())}")
        print(f"  Sessions: {sorted(df['session'].unique())}")
        print(f"  Overall accuracy: {df['trial_accuracy'].astype(float).mean():.3f}")


def cmd_accuracy(args):
    """Compute accuracy by condition."""
    root = Path(args.bids_root)
    output_dir = Path(args.output_dir)
    _ensure_output_dir(output_dir)

    df = io.load_tb2afc(root, args.subjects)
    group_cols = ["subject"] + (args.by or [])

    print(f"Computing accuracy grouped by: {group_cols}")
    result = accuracy.accuracy_by_condition(df, group_cols=group_cols)
    print(result.to_string(index=False))

    suffix = "_".join(args.by) if args.by else "overall"
    _save_tsv(result, output_dir / "group" / f"accuracy_by_{suffix}.tsv")

    if args.plot:
        x = args.by[0] if args.by else "subject"
        hue = args.by[1] if args.by and len(args.by) > 1 else None
        fig_path = str(output_dir / "figures" / f"accuracy_by_{suffix}.png")
        plotting.plot_accuracy_by_condition(
            result, x=x, hue=hue, backend=args.plot_backend,
            save_path=fig_path,
        )
        print(f"  Figure: {fig_path}")


def cmd_dprime(args):
    """Compute d' by condition."""
    root = Path(args.bids_root)
    output_dir = Path(args.output_dir)
    _ensure_output_dir(output_dir)

    df = io.load_tb2afc(root, args.subjects)
    group_cols = ["subject"] + (args.by or [])

    print(f"Computing d' grouped by: {group_cols}")
    result = accuracy.compute_sdt_2afc(df, group_cols=group_cols)
    print(result.to_string(index=False))

    suffix = "_".join(args.by) if args.by else "subject"
    _save_tsv(result, output_dir / "group" / f"dprime_by_{suffix}.tsv")


def cmd_learning(args):
    """Compute learning curves across sessions."""
    root = Path(args.bids_root)
    output_dir = Path(args.output_dir)
    _ensure_output_dir(output_dir)

    df = io.load_tb2afc(root, args.subjects)
    metric = args.metric

    if metric == "dprime":
        result = learning.session_dprime_curve(df, group_cols=["subject"])
        fname = "learning_curve_dprime"
    else:
        result = learning.session_learning_curve(
            df, metric_col="trial_accuracy", group_cols=["subject"],
        )
        fname = "learning_curve_accuracy"

    print(result.to_string(index=False))
    _save_tsv(result, output_dir / "group" / f"{fname}.tsv")

    if args.plot:
        y = "dprime_2afc" if metric == "dprime" else "mean"
        fig_path = str(output_dir / "figures" / f"{fname}.png")
        if metric == "dprime":
            plotting.plot_dprime_curve(result, save_path=fig_path,
                                       backend=args.plot_backend)
        else:
            plotting.plot_learning_curve(result, y=y, save_path=fig_path,
                                          backend=args.plot_backend)
        print(f"  Figure: {fig_path}")


def cmd_rt(args):
    """Compute RT summaries."""
    root = Path(args.bids_root)
    output_dir = Path(args.output_dir)
    _ensure_output_dir(output_dir)

    df = io.load_tb2afc(root, args.subjects)
    df = preprocessing.filter_rt(df)

    result = rt.rt_summary(df, group_cols=["subject"])
    print(result.to_string(index=False))
    _save_tsv(result, output_dir / "group" / "rt_summary.tsv")

    rt_acc = rt.rt_by_accuracy(df, group_cols=["subject"])
    _save_tsv(rt_acc, output_dir / "group" / "rt_by_accuracy.tsv")

    if args.plot:
        fig_path = str(output_dir / "figures" / "rt_distribution.png")
        plotting.plot_rt_distribution(df, group_col="subject",
                                       save_path=fig_path,
                                       backend=args.plot_backend)
        print(f"  Figure: {fig_path}")


def cmd_sme(args):
    """Compute subsequent memory effect."""
    root = Path(args.bids_root)
    output_dir = Path(args.output_dir)
    _ensure_output_dir(output_dir)

    enc_df = io.load_encoding(root, args.subjects)
    enc_df = preprocessing.remap_scanner_resp(enc_df)
    rec_df = io.load_tb2afc(root, args.subjects)

    result = encoding.subsequent_memory_effect(
        enc_df, rec_df, group_cols=["subject"],
    )
    print(result.to_string(index=False))
    _save_tsv(result, output_dir / "group" / "sme_by_encoding_rating.tsv")

    if args.plot:
        fig_path = str(output_dir / "figures" / "sme_bar.png")
        plotting.plot_subsequent_memory(result, save_path=fig_path,
                                         backend=args.plot_backend)
        print(f"  Figure: {fig_path}")


def cmd_final(args):
    """Analyze final session data."""
    root = Path(args.bids_root)
    output_dir = Path(args.output_dir)
    _ensure_output_dir(output_dir)

    fin_df = io.load_fin2afc(root, args.subjects)
    tb_df = io.load_tb2afc(root, args.subjects)
    tl_df = io.load_fintimeline(root, args.subjects)

    # FIN vs TB comparison
    comp = final_session.fin_vs_tb_accuracy(fin_df, tb_df)
    _save_tsv(comp, output_dir / "group" / "fin_comparison.tsv")

    # Timeline by condition
    tl_cond = final_session.timeline_by_condition(tl_df)
    print(tl_cond.to_string(index=False))
    _save_tsv(tl_cond, output_dir / "group" / "timeline_by_condition.tsv")

    if args.plot:
        fig_path = str(output_dir / "figures" / "fin_comparison.png")
        plotting.plot_fin_comparison(comp, save_path=fig_path,
                                      backend=args.plot_backend)
        fig_path2 = str(output_dir / "figures" / "timeline_responses.png")
        plotting.plot_timeline_responses(tl_df, save_path=fig_path2,
                                          backend=args.plot_backend)


def cmd_all(args):
    """Run all analyses."""
    print("=== Running all behavioral analyses ===\n")
    for cmd in [cmd_summary, cmd_accuracy, cmd_dprime, cmd_learning,
                cmd_rt, cmd_sme, cmd_final]:
        name = cmd.__name__.replace("cmd_", "")
        print(f"\n--- {name} ---")
        try:
            cmd(args)
        except Exception as e:
            print(f"  ERROR in {name}: {e}", file=sys.stderr)
    print("\n=== Done ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bids-root", type=str, default=str(DEFAULT_BIDS_ROOT),
        help="BIDS dataset root directory",
    )
    parser.add_argument(
        "--subjects", type=str, nargs="*", default=None,
        help="Subject IDs (e.g., 03 04 05)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(DEFAULT_BIDS_ROOT / "derivatives" / DERIVATIVES_DIR),
        help="Output directory for results",
    )
    parser.add_argument(
        "--plot", action="store_true", default=False,
        help="Generate figures",
    )
    parser.add_argument(
        "--plot-backend", type=str, default="matplotlib",
        choices=["matplotlib", "plotly", "both"],
        help="Plotting backend",
    )

    subparsers = parser.add_subparsers(dest="command", help="Analysis to run")

    # summary
    subparsers.add_parser("summary", help="Print dataset overview")

    # accuracy
    p_acc = subparsers.add_parser("accuracy", help="Accuracy by condition")
    p_acc.add_argument("--by", nargs="*", default=["enCon"],
                       help="Grouping columns (e.g., enCon reCon)")

    # dprime
    p_dp = subparsers.add_parser("dprime", help="d-prime analysis")
    p_dp.add_argument("--by", nargs="*", default=[],
                      help="Additional grouping columns")

    # learning
    p_learn = subparsers.add_parser("learning", help="Learning curves")
    p_learn.add_argument("--metric", type=str, default="accuracy",
                         choices=["accuracy", "dprime"],
                         help="Metric to track")

    # rt
    subparsers.add_parser("rt", help="Reaction time analysis")

    # sme
    subparsers.add_parser("sme", help="Subsequent memory effect")

    # final
    subparsers.add_parser("final", help="Final session analysis")

    # all
    p_all = subparsers.add_parser("all", help="Run all analyses")
    p_all.add_argument("--by", nargs="*", default=["enCon"],
                       help="Default grouping for accuracy")
    p_all.add_argument("--metric", type=str, default="accuracy",
                       choices=["accuracy", "dprime"])

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Ensure --by exists for commands that don't define it
    if not hasattr(args, "by"):
        args.by = []
    if not hasattr(args, "metric"):
        args.metric = "accuracy"

    cmd_map = {
        "summary": cmd_summary,
        "accuracy": cmd_accuracy,
        "dprime": cmd_dprime,
        "learning": cmd_learning,
        "rt": cmd_rt,
        "sme": cmd_sme,
        "final": cmd_final,
        "all": cmd_all,
    }

    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
