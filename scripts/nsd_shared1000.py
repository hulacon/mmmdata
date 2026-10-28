#!/usr/bin/env python3
"""NSD shared1000 responses for comparisons with MMMData (see ``neuroimaging/nsd.py``).

One NSD subject and one space per call; everything goes to ``<out>/subjNN/<space>/``
(``fsaverage`` is written as ``fsaverage<ico>``, e.g. ``fsaverage6``):

  extract       session-z-scored shared-trial betas, image means, NSD noise ceiling
  rois          ROI masks over the space's units (``rois.npz`` + ``rois.tsv``)
  reliability   ROI-mean, pattern and RDM reliability across repeats, never within a session;
                in ``func1pt8mm`` also over hippocampus-sized samples of all, edge and deep
                white matter, the controls that partial volume can and cannot reach

NSD is located by ``paths.nsd_dir``; ``--betas`` names the directory holding
``nsddata_betas/`` when it is not the release tree.

Usage:
    python nsd_shared1000.py extract --subj N --space func1pt8mm --betas <dir> --out <dir>
    python nsd_shared1000.py rois --subj N --space fsaverage --out <dir>
    python nsd_shared1000.py reliability --subj N --space func1pt8mm --out <dir> --rois hippocampus,phc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src" / "python") not in sys.path:  # idempotent: tests import this module repeatedly
    sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

from neuroimaging import nsd  # noqa: E402

#: Reliability's default ROIs per space.
REL_ROIS = {"func1pt8mm": ["hippocampus", "hippocampus_anterior", "hippocampus_posterior",
                           "hippocampus_gray_gap1", "amygdala_aseg", "erc", "prc", "phc",
                           "nsdgeneral", "streams_early", "streams_ventral"],
            "fsaverage": ["nsdgeneral", "streams_early", "streams_ventral"]}
WM_NULLS = ("white_matter", "white_matter_edge", "white_matter_deep")
N_PERM = 1000
SEED = 20261028


def out_dir(root: Path, subj: int, space: str, ico: int) -> Path:
    return Path(root) / f"subj{subj:02d}" / (space if space == "func1pt8mm" else f"fsaverage{ico}")


def shared1000_table() -> pd.DataFrame:
    from core.config import load_config

    bids = Path(load_config()["paths"]["bids_project_dir"])
    return pd.read_csv(bids / "stimuli" / "shared1000" / "shared1000.csv")


def cmd_extract(a, paths: nsd.NSDPaths, out: Path) -> None:
    nsd.extract_shared(paths, a.subj, a.space, out, shared1000_table(), ico=a.ico,
                       log=lambda m: print(m, flush=True))


def cmd_rois(a, paths: nsd.NSDPaths, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if a.space == "func1pt8mm":
        r, mni_y = nsd.volume_rois(paths, a.subj)
        np.savez_compressed(out / "rois.npz", mni_y=mni_y, **r)
    else:
        r = nsd.surface_rois(paths, a.ico)
        np.savez_compressed(out / "rois.npz", **r)
    counts = pd.Series({k: int(v.sum()) for k, v in r.items()}, name="n_units")
    counts.to_csv(out / "rois.tsv", sep="\t")
    print(counts.to_string())


def cmd_reliability(a, paths: nsd.NSDPaths, out: Path) -> None:
    trials = pd.read_csv(out / "trials.tsv", sep="\t")
    rois = dict(np.load(out / "rois.npz"))
    names = a.rois.split(",") if a.rois else REL_ROIS[a.space]
    rng = np.random.default_rng(SEED)
    if a.space == "func1pt8mm":
        n = int(rois[names[0]].sum())
        for k in WM_NULLS:
            rois[f"{k}_null"] = nsd.sample_units(rois[k], n, rng)
        names = names + [f"{k}_null" for k in WM_NULLS]
    missing = [k for k in names if k not in rois]
    if missing:
        raise SystemExit(f"no ROI {missing} in {out / 'rois.npz'}; run `rois` first")
    union = np.flatnonzero(np.any([rois[k] for k in names], axis=0))
    z = np.load(out / "shared_betas_z.npy", mmap_mode="r")
    zu = np.asarray(z[:, union])  # one pass over the file
    col = np.full(z.shape[1], -1)
    col[union] = np.arange(union.size)
    rows = []
    for k in names:
        rows.append({"roi": k, **nsd.pattern_reliability(zu[:, col[np.flatnonzero(rois[k])]], trials, N_PERM, rng)})
        print(k, "done", flush=True)
    tab = pd.DataFrame(rows).set_index("roi")
    tab.to_csv(out / "reliability_roi.tsv", sep="\t")
    print(tab.round(4).to_string())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verb", choices=("extract", "rois", "reliability"))
    ap.add_argument("--subj", type=int, required=True, help="NSD subject number, 1-8")
    ap.add_argument("--space", choices=nsd.SPACES, required=True)
    ap.add_argument("--ico", type=int, choices=sorted(nsd.ICO_VERTICES), default=6,
                    help="fsaverage: icosahedral order to cut to (6 = fsaverage6, MMMData's grid)")
    ap.add_argument("--out", type=Path, required=True, help="root; writes <out>/subjNN/<space>/")
    ap.add_argument("--betas", type=Path, help="directory holding nsddata_betas/ (default: paths.nsd_dir)")
    ap.add_argument("--rois", help="reliability: comma-separated ROI names (default: per space)")
    a = ap.parse_args(argv)
    paths = nsd.NSDPaths.from_config(a.betas)
    out = out_dir(a.out, a.subj, a.space, a.ico)
    {"extract": cmd_extract, "rois": cmd_rois, "reliability": cmd_reliability}[a.verb](a, paths, out)


if __name__ == "__main__":
    main()
