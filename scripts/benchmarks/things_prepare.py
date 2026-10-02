#!/usr/bin/env python3
"""THINGS odd-one-out inputs for ``psytwill bench oddoneout``.

The THINGS-data triplet splits (Hebart et al. 2023, Figshare+ 20552784,
``full_triplet_dataset.zip``) store each triplet as three 0-based indices
into the THINGS concept order (``variables/unique_id.txt``), reordered so
the chosen pair comes first and the odd one out last. This script makes
both sides speak concept ids:

  triplets  each split -> CSV with ``item1, item2, item3, odd, split``
            (``odd`` = item3, by the source's ordering)
  key       an extractor CSV over one image per concept
            (``<concept>_<nn><b|s|n>.jpg``) -> the same CSV with
            ``stimulus_id`` = concept, so the store loader keys it like any
            other set, plus ``<output stem>_reference.csv`` saying which
            concepts carry their true reference image

Reference images: THINGS marks the images of its object-naming study with
``b`` (``s`` = other web images, ``n`` = ImageNet), and the triplet task
showed one such image per concept. 191 of the 1854 reference images were
excluded from the public database for quality, so for those concepts the
image participants saw is not available; the fetch step substitutes each
one's first public image. Score triplets both ways — all of them, and only
those whose three items are all ``b`` — since a substitute is a different
photograph of the concept.

The images themselves are research-use only and stay where they were
downloaded; nothing here copies them.

Usage:
    python things_prepare.py triplets --root <things dir> -o <things dir>/triplets/
    python things_prepare.py key ebind.csv -o ebind_keyed.csv --root <things dir>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("trainset", "validationset", "testset1", "testset2", "testset2_repeat", "testset3")
STEM = re.compile(r"^(?P<concept>.+)_(?P<num>\d+)(?P<kind>[bsn])$")


def concepts(root: Path) -> list[str]:
    ids = [ln.strip() for ln in (root / "variables" / "unique_id.txt").read_text().splitlines() if ln.strip()]
    if len(ids) != 1854 or len(set(ids)) != len(ids):
        raise SystemExit(f"unique_id.txt holds {len(ids)} ids ({len(set(ids))} distinct); expected 1854 distinct")
    return ids


def cmd_triplets(args) -> None:
    root = Path(args.root)
    ids = np.array(concepts(root))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        src = root / "triplet_dataset" / f"{split}.txt"
        if not src.exists():
            print(f"  {split}: absent, skipped")
            continue
        t = np.loadtxt(src, dtype=int, ndmin=2)
        if t.min() < 0 or t.max() >= len(ids):
            raise SystemExit(f"{src}: index range {t.min()}..{t.max()} outside 0..{len(ids) - 1}")
        df = pd.DataFrame({"item1": ids[t[:, 0]], "item2": ids[t[:, 1]], "item3": ids[t[:, 2]]})
        df["odd"] = df["item3"]
        df["split"] = split
        df.to_csv(out / f"{split}.csv", index=False)
        print(f"  {split}: {len(df)} triplets over {len(np.unique(t))} concepts -> {out / f'{split}.csv'}")


def cmd_key(args) -> None:
    ids = set(concepts(Path(args.root)))
    df = pd.read_csv(args.input)
    stems = df["filename"].map(lambda f: Path(str(f)).stem)
    parsed = stems.str.extract(STEM)
    if parsed["concept"].isna().any():
        raise SystemExit(f"{parsed['concept'].isna().sum()} filename(s) do not parse as <concept>_<nn><b|s|n> "
                         f"(e.g. {stems[parsed['concept'].isna()].iloc[0]})")
    sid = parsed["concept"]
    if sid.duplicated().any():
        raise SystemExit(f"{sid.duplicated().sum()} concept(s) have more than one image; keep one per concept")
    unknown = sorted(set(sid) - ids)
    if unknown:
        raise SystemExit(f"{len(unknown)} image stem(s) match no THINGS concept id (e.g. {unknown[:3]}); "
                         "map them explicitly before keying")
    missing = sorted(ids - set(sid))
    # viz2psy stamps the filename stem (<concept>_01b); the triplets name concepts
    df = df.drop(columns=["stimulus_id"], errors="ignore")
    df.insert(0, "stimulus_id", sid)
    df.to_csv(args.output, index=False)
    ref = pd.DataFrame({"stimulus_id": sid, "image": stems, "is_reference": parsed["kind"] == "b"})
    ref_path = Path(args.output).with_name(Path(args.output).stem + "_reference.csv")
    ref.to_csv(ref_path, index=False)
    print(f"keyed {len(df)} rows -> {args.output}; {len(missing)} concept(s) without an image; "
          f"{int((~ref['is_reference']).sum())} substitute(s) for an unpublished reference image -> {ref_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("triplets")
    t.add_argument("--root", required=True, help="dir holding variables/ and triplet_dataset/")
    t.add_argument("-o", "--output", required=True)
    t.set_defaults(func=cmd_triplets)
    k = sub.add_parser("key")
    k.add_argument("input")
    k.add_argument("-o", "--output", required=True)
    k.add_argument("--root", required=True)
    k.set_defaults(func=cmd_key)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
