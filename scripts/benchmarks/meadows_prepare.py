#!/usr/bin/env python3
"""NSD Meadows judgements as inputs for ``psytwill bench rdm`` and ``psytwill bench rating``.

NSD's public behavioural release (``nsddata/bdata/meadows/``) holds two files
from the Meadows platform:

- ``Meadows_nsd-multiple-arrangements_v_v2_tree.json``: per participant
  (a Meadows pseudonym) -> ``tasks``; the ``multiarrange`` task carries
  ``stimuli`` (each with ``name`` and the image file's ``size`` in bytes) and
  ``rdm``, the condensed upper triangle of that participant's
  dissimilarities in ``stimuli`` order (scipy ``squareform`` layout).
- ``Meadows_nsd-multiple-arrangements_v_v2_3D.mat``: drag-rate ratings,
  ``property`` (tasks x participants x stimuli, NaN where a block did not
  show the image), with ``dragrate_tasks`` naming each task
  (``<dimension> - block <k>``), ``participation_ids`` and ``stimuli``.

Stimulus names are the shared1000 file stems (``shared####_nsd#####``),
which is how the Contract B store keys shared1000. The join is on that name,
never on NSD id arithmetic (the shared1000 tables disagree on nsdId by one).
Each name must be a shared1000 image, and with ``--check-bytes`` each image's
byte size must equal the Meadows record's, so a renamed or re-encoded file
cannot pass as the image people judged.

Outputs (in ``-o``):

  rdm_pairs.csv   participant, image_a, image_b, dissimilarity   (one row per judged pair)
  ratings.csv     participant, stimulus_id, <dimension>...       (one row per participant per image)
  prepare.json    source checksums, counts and the checks that ran

Usage:
    python meadows_prepare.py --root <meadows dir> -o <meadows dir>/prepared [--check-bytes]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

TREE = "Meadows_nsd-multiple-arrangements_v_v2_tree.json"
MAT = "Meadows_nsd-multiple-arrangements_v_v2_3D.mat"
BLOCK = re.compile(r"^(?P<dim>.+?) - block (?P<k>\d+)$")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shared1000_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    from core.config import load_config

    return Path(load_config()["paths"]["bids_project_dir"]) / "stimuli" / "shared1000"


def arrangement_pairs(tree: dict) -> tuple[pd.DataFrame, dict[str, int]]:
    """Long pairs table from every participant's multi-arrangement RDM, plus each stimulus's byte size."""
    from scipy.spatial.distance import squareform

    rows, sizes = [], {}
    for pid, d in tree.items():
        tasks = [t for t in d["tasks"] if t.get("task", {}).get("task_type") == "multiarrange"]
        if len(tasks) != 1:
            raise SystemExit(f"participant {pid}: {len(tasks)} multiarrange tasks; expected 1")
        t = tasks[0]
        names = [s["name"] for s in t["stimuli"]]
        if len(set(names)) != len(names):
            raise SystemExit(f"participant {pid}: a stimulus is listed twice")
        n = len(names)
        rdm = np.asarray(t["rdm"], dtype=float)
        if rdm.size != n * (n - 1) // 2:
            raise SystemExit(f"participant {pid}: rdm has {rdm.size} entries; {n} stimuli need {n * (n - 1) // 2}")
        if np.isnan(rdm).any() or (rdm < 0).any():
            raise SystemExit(f"participant {pid}: rdm holds NaN or negative values")
        D = squareform(rdm)
        ii, jj = np.triu_indices(n, k=1)
        a = np.array(names)[ii]
        b = np.array(names)[jj]
        lo, hi = np.where(a < b, a, b), np.where(a < b, b, a)
        rows.append(pd.DataFrame({"participant": pid, "image_a": lo, "image_b": hi, "dissimilarity": D[ii, jj]}))
        for s in t["stimuli"]:
            if sizes.setdefault(s["name"], s["size"]) != s["size"]:
                raise SystemExit(f"stimulus {s['name']} has two byte sizes across participants")
    return pd.concat(rows, ignore_index=True), sizes


def all_sizes(tree: dict) -> dict[str, int]:
    """Byte size of every stimulus any task showed (arrangement and rating blocks)."""
    sizes: dict[str, int] = {}
    for d in tree.values():
        for t in d["tasks"]:
            for st in t.get("stimuli") or []:
                if sizes.setdefault(st["name"], st["size"]) != st["size"]:
                    raise SystemExit(f"stimulus {st['name']} has two byte sizes in the Meadows record")
    return sizes


def _strings(x) -> list[str]:
    """A MATLAB char matrix or cell array of strings as a list of stripped str (padding removed)."""
    out = []
    for v in np.asarray(x, dtype=object).ravel():
        while isinstance(v, np.ndarray):
            v = v.ravel()[0] if v.size == 1 else "".join(map(str, v.ravel()))
        out.append(str(v).strip())
    return out


def rating_table(mat_path: Path) -> pd.DataFrame:
    """One row per participant per image, one column per rated dimension.

    Each dimension's blocks show disjoint image subsets, so every image must
    be rated in exactly one block per participant; anything else stops the
    build rather than averaging or picking a block.
    """
    import scipy.io as sio

    m = sio.loadmat(mat_path)
    tasks, pids, stims = (_strings(m[k]) for k in ("dragrate_tasks", "participation_ids", "stimuli"))
    prop = np.asarray(m["property"], dtype=float)
    if prop.shape != (len(tasks), len(pids), len(stims)):
        raise SystemExit(f"property has shape {prop.shape}; expected {(len(tasks), len(pids), len(stims))}")
    dims: dict[str, list[int]] = {}
    for k, t in enumerate(tasks):
        mt = BLOCK.match(t)
        if not mt:
            raise SystemExit(f"drag-rate task {t!r} is not '<dimension> - block <k>'")
        dims.setdefault(mt["dim"], []).append(k)
    out = pd.DataFrame([(p, s) for p in pids for s in stims], columns=["participant", "stimulus_id"])
    for dim, ks in dims.items():
        block = prop[ks]  # (blocks, participants, stimuli)
        n_rated = (~np.isnan(block)).sum(0)
        if (n_rated != 1).any():
            bad = int((n_rated != 1).sum())
            raise SystemExit(f"{dim}: {bad} participant-image cells are rated in {sorted(set(n_rated.ravel()))} "
                             "blocks; expected exactly 1")
        out[dim] = np.nansum(block, axis=0).ravel()
    return out


def cmd_prepare(args) -> None:
    root, out = Path(args.root), Path(args.output)
    tree = json.loads((root / TREE).read_text())
    pairs, sizes = arrangement_pairs(tree)
    ratings = rating_table(root / MAT)
    s1000 = shared1000_dir(args.shared1000)
    files = pd.read_csv(s1000 / "shared1000.csv")["fileNames"].astype(str)
    stems = {Path(f).stem: f for f in files}
    judged = sorted(sizes)
    rated = sorted(ratings["stimulus_id"].unique())
    unknown = sorted(set(judged + rated) - set(stems))
    if unknown:
        raise SystemExit(f"{len(unknown)} Meadows stimulus name(s) are not shared1000 images (e.g. {unknown[:3]})")
    checks = {"names_in_shared1000": True, "byte_sizes_checked": False}
    if args.check_bytes:
        img = s1000 / "images"
        every = all_sizes(tree)
        bad = [s for s, b in every.items() if (img / stems[s]).stat().st_size != b]
        if bad:
            raise SystemExit(f"{len(bad)} Meadows image(s) differ in byte size from the shared1000 file "
                             f"(e.g. {bad[:3]}); these are not the files people judged")
        checks |= {"byte_sizes_checked": True, "byte_sizes_match": f"{len(every)}/{len(every)}"}
    if sorted(tree) != sorted(ratings["participant"].unique()):
        raise SystemExit("the rating file's participants are not the arrangement file's")
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "rdm_pairs.csv", index=False)
    ratings.to_csv(out / "ratings.csv", index=False)
    dims = [c for c in ratings.columns if c not in ("participant", "stimulus_id")]
    record = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "mmmdata scripts/benchmarks/meadows_prepare.py",
        "sources": {f: {"sha256": _sha256(root / f)} for f in (TREE, MAT)},
        "rdm_pairs": {"n_participants": int(pairs["participant"].nunique()), "n_images": len(judged),
                      "n_rows": int(len(pairs))},
        "ratings": {"n_participants": int(ratings["participant"].nunique()), "n_images": len(rated),
                    "dimensions": dims, "range": [float(ratings[dims].min().min()), float(ratings[dims].max().max())]},
        "checks": checks,
    }
    (out / "prepare.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"  rdm_pairs.csv: {record['rdm_pairs']}")
    print(f"  ratings.csv: {record['ratings']}")
    print(f"  checks: {checks}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="directory holding the two Meadows files")
    ap.add_argument("-o", "--output", required=True, help="output directory")
    ap.add_argument("--shared1000", help="shared1000 stimulus dir (default: <bids_project_dir>/stimuli/shared1000)")
    ap.add_argument("--check-bytes", action="store_true",
                    help="require each judged image's byte size to match the Meadows record")
    cmd_prepare(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
