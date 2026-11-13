#!/usr/bin/env python3
"""Shared vs unique directions of a psytwill block, by brain variance at matched variance.

A block's directions are each scored alone by the harness (``harness.py directions``: one
``dir-<B>.c<NNN>`` unit per direction and subject, held-out films R² per column). This driver
fits nothing. Per subject it reads every direction's unit, takes the ROI mean of R² over the
columns of each ROI inside the films reliability mask (HCP-MMP1's 22 sections,
``families.familyD``, as ``rescue.py``) and over the whole mask, and runs braintwill's
``universality_test`` on the (directions x ROIs) matrix: eigenvalue strata, the upper-minus-
lower-half contrast of u within each stratum (``T``), its within-stratum label-permutation
null, and the partial Spearman of u with R² given the eigenvalue (and a direction reliability
where the table has one).

The per-direction table (one per block, ``--blocks B=<path>``; CSV or parquet) holds:

  component   the direction index c (``dir-<B>.c<NNN>``), unique
  eigenvalue  the block's eigenvalue of c (the strata are built from it)
  u           the universality score of c (model-readout prereg R7)
  rho         optional: the direction's split-half reliability (prereg R8); a covariate of
              the partial Spearman only

The read covers the table's components only, and reports whether they are every component
of the block's projection (``complete``). A component without a cached unit is an error
naming the ``directions`` job that writes it.

Per ROI it reports T, the null's mean and 95th percentile, the one-sided permutation p,
whether T exceeds the 95th percentile, and the partial Spearman; with several blocks, the p
Holm-adjusted over the blocks (per subject and ROI) beside the raw p. Every direction's ROI
R² is written out. Descriptive: no verdict, no subject-count rule.

Usage:
    python universality.py read --blocks V=<V.parquet>,A=<A.parquet>,L=<L.parquet> --out <dir>
    python universality.py read --blocks L.dev010=<table.csv> --subs 03 --out <dir>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import harness as h  # noqa: E402  (puts functional_space and src/python on sys.path)
import rescue as rs  # noqa: E402

#: The read's fixed settings (model-readout prereg §5.3). A different value is a different
#: read, recorded as such; these are not tuning knobs.
READ = {"n_strata": 4, "n_perm": 10_000, "seed": 0, "family": rs.RULE["family"], "min_cols": rs.RULE["min_cols"],
        "register": "films"}


def load_table(path: Path, block: str) -> pd.DataFrame:
    """A block's per-direction table, checked (module docstring)."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} (the {block} direction table) is missing")
    t = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    need = {"component", "eigenvalue", "u"}
    if need - set(t.columns):
        raise SystemExit(f"{path}: missing columns {sorted(need - set(t.columns))}; it needs component, eigenvalue, u "
                         "and optionally rho")
    if t["component"].duplicated().any():
        raise SystemExit(f"{path}: a component appears twice")
    for c in ("eigenvalue", "u"):
        if not np.isfinite(t[c].to_numpy(float)).all():
            raise SystemExit(f"{path}: {c} is not finite for every component")
    if "rho" in t.columns and not np.isfinite(t["rho"].to_numpy(float)).all():
        raise SystemExit(f"{path}: rho is given but not finite for every component")
    t = t.astype({"component": int}).sort_values("component").reset_index(drop=True)
    if (t["component"] < 0).any():
        raise SystemExit(f"{path}: negative component")
    return t


def parse_blocks(text: str) -> dict[str, Path]:
    out = {}
    for part in text.split(","):
        block, sep, path = part.partition("=")
        if not sep or not path:
            raise SystemExit(f"--blocks {text!r}: expected B=<path>[,B=<path>...]")
        try:
            h.block_stream(block)
        except ValueError as exc:
            raise SystemExit(str(exc))
        if block in out:
            raise SystemExit(f"--blocks: {block} twice")
        out[block] = Path(path)
    return out


def roi_means(r2: np.ndarray, rois: dict[str, np.ndarray]) -> np.ndarray:
    """``(n_directions, n_rois)`` ROI means of per-column R² ``(n_directions, n_columns)``."""
    return np.column_stack([r2[:, ix].mean(axis=1) for ix in rois.values()])


def holm_over_blocks(p: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per ROI, the blocks' p Holm-adjusted over the blocks (``rescue.holm``)."""
    blocks = list(p)
    n_roi = len(next(iter(p.values())))
    out = {b: np.empty(n_roi) for b in blocks}
    for j in range(n_roi):
        adj = rs.holm({b: float(p[b][j]) for b in blocks})
        for b in blocks:
            out[b][j] = adj[b]
    return out


def _sha256(path: Path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def subject_read(paths, cache, block: str, table: pd.DataFrame, sub: str, labels: np.ndarray):
    """One block and subject: the ROI R² matrix, the ROIs, and the test (``universality_test``)."""
    from braintwill.encoding import universality_test

    setup = h.direction_setup(paths, block, sub)
    k = setup["k"]
    if table["component"].max() >= k:
        raise SystemExit(f"{block}: the table names components up to {table['component'].max()}; the projection has {k}")
    r2, valid, keys = [], None, {}
    for c in table["component"]:
        key = h.direction_key(setup, block, int(c), sub)
        got = cache.get(key)
        if got is None:
            raise SystemExit(f"{h.direction_space(block, int(c))} sub-{sub} is not scored: {cache.path(key)}; run "
                             f"harness.py directions --block {block} --sub {sub} --components {c}:{c + 1}")
        r2.append(np.asarray(got.r2, np.float64))
        valid = got.valid if valid is None else valid & got.valid
        keys[int(c)] = str(cache.path(key))
    r2 = np.vstack(r2)
    mask = np.asarray(setup["mask"], bool)
    rois = rs.roi_columns(labels, mask, valid, READ["min_cols"])
    rois[rs.ALL_ROI] = np.flatnonzero(mask & valid)
    m = roi_means(r2, rois)
    rho = table["rho"].to_numpy(float) if "rho" in table.columns else None
    test = universality_test(m, table["u"].to_numpy(float), table["eigenvalue"].to_numpy(float), rho,
                             n_strata=READ["n_strata"], n_perm=READ["n_perm"], seed=READ["seed"])
    return {"rois": rois, "r2": m, "test": test, "k": k, "keys": keys, "mask_path": str(setup["mask_path"])}


def cmd_read(args) -> None:
    import families as fam
    import grayordinates as go
    from braintwill.encoding import ScoreCache

    import cha

    blocks = parse_blocks(args.blocks)
    tables = {b: load_table(p, b) for b, p in blocks.items()}
    paths = h.Paths()
    cache = ScoreCache(paths.out)
    grid = pd.read_csv(go.grayordinates_path(paths.cleaned), sep="\t")
    labels = np.asarray(fam.family_labels(grid, cha.Paths().atlases)[READ["family"]])
    subs = args.subs.split(",") if args.subs else list(h.fm.SUBJECTS)
    got = {b: {} for b in blocks}
    for sub in subs:
        for b in blocks:
            got[b][sub] = subject_read(paths, cache, b, tables[b], sub, labels)
            print(f"sub-{sub} {b}: {len(tables[b])} directions, {len(got[b][sub]['rois'])} ROIs", flush=True)
    roi_rows, dir_rows, entry = [], [], {}
    for sub in subs:
        p = {b: got[b][sub]["test"]["p"] for b in blocks}
        adj = holm_over_blocks(p) if len(blocks) > 1 else None
        for b in blocks:
            g, t = got[b][sub], got[b][sub]["test"]
            names = list(g["rois"])
            for j, roi in enumerate(names):
                roi_rows.append({"block": b, "sub": sub, "roi": roi, "n_cols": int(g["rois"][roi].size),
                                 "T": t["T"][j], "null_mean": t["null_mean"][j], "null_q95": t["null_q95"][j],
                                 "p": t["p"][j], "p_holm_blocks": None if adj is None else adj[b][j],
                                 "exceeds_q95": bool(t["T"][j] > t["null_q95"][j]),
                                 "partial_spearman": t["partial_spearman"][j]})
            tb = tables[b]
            for i, c in enumerate(tb["component"]):
                for j, roi in enumerate(names):
                    dir_rows.append({"block": b, "sub": sub, "component": int(c), "eigenvalue": tb["eigenvalue"][i],
                                     "u": tb["u"][i], "rho": tb["rho"][i] if "rho" in tb.columns else None,
                                     "stratum": int(t["strata"][i]), "half": int(t["halves"][i]),
                                     "roi": roi, "r2": g["r2"][i, j]})
            entry.setdefault(b, {})[f"sub-{sub}"] = {"k": g["k"], "n_directions": len(tb),
                                                     "complete": len(tb) == g["k"], "mask": g["mask_path"],
                                                     "units": g["keys"]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"universality_{'+'.join(blocks)}_{READ['register']}"
    pd.DataFrame(roi_rows).to_csv(f"{stem}_rois.csv", index=False)
    pd.DataFrame(dir_rows).to_csv(f"{stem}_directions.csv", index=False)
    side = {"read": READ, "statistic": "T: mean over eigenvalue strata of mean R2 (upper half of u) minus mean R2 "
                                       "(lower half); null: u permuted within strata (braintwill universality_test)",
            "pooled_roi": rs.ALL_ROI, "multiplicity": "p_holm_blocks: Holm over the blocks read, per subject and ROI; "
                                                      "raw p beside it; no verdict",
            "tables": {b: {"path": str(p), "sha256": _sha256(p), "rho": "rho" in tables[b].columns}
                       for b, p in blocks.items()},
            "blocks": entry, "subjects": subs, "provenance": h.provenance("numpy", {}),
            "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    Path(f"{stem}.json").write_text(json.dumps(side, indent=2, default=str) + "\n")
    print(f"wrote {stem}.json, {stem}_rois.csv, {stem}_directions.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("read")
    r.add_argument("--blocks", required=True, help="B=<direction table>[,B=<table>...] (module docstring)")
    r.add_argument("--subs", help="comma-separated subjects (default: the harness's)")
    r.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()
    {"read": cmd_read}[args.verb](args)


if __name__ == "__main__":
    main()
