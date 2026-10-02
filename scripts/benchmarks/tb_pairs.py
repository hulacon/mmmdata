#!/usr/bin/env python3
"""One row per studied image-word pair per subject, with the behaviour the pair drew.

The input to ``psytwill bench congruence``: TB pairings are random per
subject, so each subject's pairs are an independent draw of image-word
congruence, and the outcomes below say what people did with each pair.

Columns:

- ``subject, image_id, word_id``: ``image_id`` is the shared1000 stimulus id
  (``shared####_nsd#####``), ``word_id`` the twp1000 word, both as keyed in
  the Contract B store
- ``enCon``: exposure condition (1 single, 2 repeats, 3 triplets)
- ``n_enc``: encoding presentations found
- ``assoc_first`` / ``assoc_mean``: the encoding associability rating
  (1 not at all .. 3 very well; button codes 6/7/8 remapped) on the first
  presentation, and over all presentations with a response
- ``afc_correct``: 2-AFC accuracy (mean over tests of this pair; NaN when
  untested or every test went unanswered)
- ``afc_sure_correct``: 1 when a test was answered correctly with high
  confidence, 0 otherwise, NaN when untested or unanswered

A pair seen with two different images or words within one subject stops
the build: the pairing is the unit, and a remap would silently merge two.

Usage:
    python tb_pairs.py -o <dir>/tb_pairs.csv [--subjects 03 04 ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

from behavioral.io import load_encoding, load_tb2afc  # noqa: E402
from behavioral.preprocessing import remap_scanner_resp  # noqa: E402
from core.config import load_config  # noqa: E402


def image_id(mmm_id, nsd_id) -> str:
    return f"shared{int(mmm_id):04d}_nsd{int(nsd_id):05d}"


def encoding_pairs(bids_root: Path, subjects) -> pd.DataFrame:
    enc = load_encoding(bids_root, subjects=subjects)
    enc = enc[enc["trial_type"] == "image"].copy()
    enc = remap_scanner_resp(enc, output_col="assoc")
    enc["image_id"] = [image_id(m, n) for m, n in zip(enc["mmmId"], enc["nsdId"])]
    enc = enc.rename(columns={"word": "word_id"})
    enc = enc.sort_values(["subject", "session", "run", "onset"])
    bad = enc.groupby(["subject", "word_id"])["image_id"].nunique()
    bad = bad[bad > 1]
    if len(bad):
        raise SystemExit(f"{len(bad)} (subject, word) keys pair with more than one image; first: "
                         f"{bad.index[0]}. Resolve before building pairs.")
    g = enc.groupby(["subject", "image_id", "word_id"], sort=False)
    out = g.agg(enCon=("enCon", "first"), n_enc=("assoc", "size"), assoc_mean=("assoc", "mean")).reset_index()
    first = enc.dropna(subset=["assoc"]).groupby(["subject", "image_id", "word_id"])["assoc"].first()
    return out.merge(first.rename("assoc_first").reset_index(), how="left")


def afc_outcomes(bids_root: Path, subjects) -> pd.DataFrame:
    afc = load_tb2afc(bids_root, subjects=subjects)
    resp = pd.to_numeric(afc["resp"], errors="coerce")
    answered = resp.isin([1, 2, 3, 4])
    acc = pd.to_numeric(afc["trial_accuracy"], errors="coerce").where(answered)
    afc = afc.assign(afc_correct=acc,
                     afc_sure_correct=((acc == 1) & resp.isin([1, 4])).astype(float).where(answered),
                     word_id=afc["word"])
    return afc.groupby(["subject", "word_id"])[["afc_correct", "afc_sure_correct"]].mean().reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", required=True, help="CSV to write (a data table: keep it beside the data, "
                                                          "not in a repo)")
    ap.add_argument("--subjects", nargs="+", help="bare labels; default every sub-* under the BIDS root")
    args = ap.parse_args()
    bids_root = Path(load_config()["paths"]["bids_project_dir"])
    subjects = args.subjects or sorted(p.name[4:] for p in bids_root.glob("sub-*") if p.is_dir())
    pairs = encoding_pairs(bids_root, subjects)
    afc = afc_outcomes(bids_root, subjects)
    out = pairs.merge(afc, on=["subject", "word_id"], how="left")
    orphan = afc.merge(pairs[["subject", "word_id"]], how="left", indicator=True)
    n_orphan = int((orphan["_merge"] == "left_only").sum())
    if n_orphan:
        raise SystemExit(f"{n_orphan} 2-AFC (subject, word) key(s) match no encoding pair; check the word "
                         "column before trusting the join")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    by = out.groupby("subject").agg(pairs=("image_id", "size"), tested=("afc_correct", lambda s: int(s.notna().sum())),
                                    rated=("assoc_first", lambda s: int(s.notna().sum())))
    print(f"wrote {len(out)} pairs -> {args.output}")
    print(by.to_string())
    print(f"afc_correct mean {np.nanmean(out['afc_correct']):.3f}; assoc_first counts "
          f"{out['assoc_first'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
