#!/usr/bin/env python3
"""Write NSD shared1000 single-trial betas as an ROI pattern cache, so
``item_reliability.py`` scores NSD with exactly the code it uses for MMMData.

Inputs (one NSD subject, one space):
  --shared-dir   a directory holding ``shared_betas_z.npy`` (trials x units),
                 ``trials.tsv`` (session, trial = 1-based trial index within the
                 session, mmm_id, rep) and ``rois.npz`` (boolean unit masks)
  --responses    NSD's ``ppdata/subj##/behav/responses.tsv`` — the source of
                 each trial's RUN. Run boundaries are read, never assumed.
  --roi-file     optional npz of boolean unit masks to use instead of the
                 shared dir's ``rois.npz`` (e.g. atlas areas mapped onto the
                 same surface, so two datasets are read with one ROI set)

Output: npz with ``patterns_<ROI>`` (units x trials), ``voxidx_<ROI>``, and the
trial metadata ``item_reliability.py`` reads (session, run, task = "nsd",
mmmId, sharedId = 0, enCon = NaN, onset = the session trial index, which
orders trials within a run).

Usage:
    python nsd_shared_cache.py --shared-dir <.../shared1000/subj##/func1pt8mm> \\
        --responses <.../ppdata/subj##/behav/responses.tsv> \\
        --rois streams_early streams_ventral --out <subj##_nsd_roipatterns.npz>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shared-dir", required=True, type=Path)
    ap.add_argument("--responses", required=True, type=Path)
    ap.add_argument("--rois", nargs="+", required=True)
    ap.add_argument("--roi-file", type=Path, default=None,
                    help="npz of unit masks to use instead of <shared-dir>/rois.npz")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    roi_file = args.roi_file or args.shared_dir / "rois.npz"
    for f in (args.shared_dir / "shared_betas_z.npy", args.shared_dir / "trials.tsv", roi_file):
        if not f.is_file():
            sys.exit(f"missing {f}")
    if not args.responses.is_file():
        sys.exit(f"missing {args.responses}")

    t = pd.read_csv(args.shared_dir / "trials.tsv", sep="\t")
    beh = pd.read_csv(args.responses, sep="\t", usecols=["SESSION", "RUN", "TRIAL", "73KID"])
    beh = beh.sort_values(["SESSION", "RUN", "TRIAL"]).copy()
    beh["strial"] = beh.groupby("SESSION").cumcount() + 1
    m = t.merge(beh, left_on=["session", "trial"], right_on=["SESSION", "strial"],
                how="left", validate="one_to_one")
    if m["RUN"].isna().any():
        sys.exit(f"{int(m['RUN'].isna().sum())} trials have no RUN in responses.tsv")
    # the join is on position; the image shown must agree
    bad = (m["73KID"] != m["nsd_id"]).sum()
    if bad:
        sys.exit(f"{bad} trials: responses.tsv 73KID != trials.tsv nsd_id after the join")

    rois = np.load(roi_file)
    unknown = [r for r in args.rois if r not in rois.files]
    if unknown:
        sys.exit(f"ROIs not in {roi_file.name}: {unknown}; have {rois.files}")
    B = np.load(args.shared_dir / "shared_betas_z.npy", mmap_mode="r")
    if B.shape[0] != len(t):
        sys.exit(f"shared_betas_z rows {B.shape[0]} != trials {len(t)}")
    bad = [r for r in args.rois if rois[r].shape != (B.shape[1],)]
    if bad:
        sys.exit(f"ROI masks {bad} do not have one entry per unit ({B.shape[1]})")

    out = {"subject": args.shared_dir.parent.name, "arm": "nsd",
           "beta_type": "NSD-b3", "space": args.shared_dir.name,
           "roi_names": np.array(args.rois),
           "session": np.array([f"ses-{s:02d}" for s in m["session"]], dtype=object),
           "run": m["RUN"].astype(int).to_numpy(),
           "task": np.array(["nsd"] * len(m), dtype=object),
           "mmmId": m["mmm_id"].astype(str).to_numpy().astype(object),
           "sharedId": np.zeros(len(m)), "enCon": np.full(len(m), np.nan),
           "onset": m["trial"].astype(float).to_numpy()}
    for r in args.rois:
        idx = np.flatnonzero(rois[r].astype(bool))
        out[f"patterns_{r}"] = np.ascontiguousarray(B[:, idx].T, dtype=np.float32)
        out[f"voxidx_{r}"] = idx
        print(f"{r}: {len(idx)} units", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
