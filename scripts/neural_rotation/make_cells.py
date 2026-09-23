#!/usr/bin/env python3
"""Write the cells manifest fit_ladder.sbatch iterates over.

One line per subject x ROI x pair, tab-separated ``subject rung roi pair``,
from ``functional_rois/ladder.tsv`` (rungs i-iii) and each subject's
``<sub>_task-prf_masks.tsv`` (rung iv: pos / negstrict / union at every
threshold present). ROIs flagged ``too_small`` are skipped and listed.

Usage:
    python make_cells.py --subjects sub-## [sub-## ...] --out cells.tsv \\
        [--pairs enc:ret-word enc:ret-image ret-word:ret-image enc:ret-word:ret-image] \\
        [--rungs i ii iii iv] [--rois mPFC ...]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parent.parent


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tb = _load_module("glmsingle_tb", SCRIPTS / "glmsingle_tb.py")
PAIRS = ["enc:ret-word", "enc:ret-image", "ret-word:ret-image", "enc:ret-word:ret-image"]


def prf_rois(roi_root: Path, subject: str) -> list:
    t = roi_root / subject / f"{subject}_task-prf_masks.tsv"
    if not t.exists():
        return []
    df = pd.read_csv(t, sep="\t")
    df = df[df["space"] == tb.SPACE]
    out = []
    for _, r in df.iterrows():
        thr = f"{r['threshold_pct']:g}".replace(".", "p")
        out.append(f"Prf{str(r['mask']).capitalize()}Thr{thr}")
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="+", default=PAIRS)
    ap.add_argument("--rungs", nargs="+", default=["i", "ii", "iii", "iv"])
    ap.add_argument("--rois", nargs="+", default=None,
                    help="only these ROI names (any rung); default every ROI of the chosen rungs")
    ap.add_argument("--roi-root", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = tb.load_config()
    roi_root = Path(args.roi_root) if args.roi_root else Path(cfg["output_dir"]) / "functional_rois"
    ladder = pd.read_csv(roi_root / "ladder.tsv", sep="\t")
    small = ladder[ladder["too_small"].astype(bool)]["roi"].tolist()
    ladder = ladder[~ladder["too_small"].astype(bool)]
    lines = []
    for sub in args.subjects:
        rois = [(r["rung"], r["roi"]) for _, r in ladder.iterrows() if r["rung"] in args.rungs]
        if "iv" in args.rungs:
            rois += [("iv", r) for r in prf_rois(roi_root, sub)]
        if args.rois:
            unknown = sorted(set(args.rois) - {r for _, r in rois})
            if unknown:
                sys.exit(f"ERROR: --rois not on the ladder for {sub}: {unknown}")
            rois = [(g, r) for g, r in rois if r in args.rois]
        for rung, roi in rois:
            for pair in args.pairs:
                lines.append(f"{sub}\t{rung}\t{roi}\t{pair}")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} cells -> {args.out}" + (f"; skipped too-small ROIs: {small}" if small else ""))


if __name__ == "__main__":
    main()
