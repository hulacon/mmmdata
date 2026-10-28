#!/usr/bin/env python3
"""NSD shared1000 register for braintwill's encoding harness (an outside comparison dataset).

The same harness as ``harness.py`` (nested grouped CV, banded ridge, per-column held-out R²,
scores cached by feature-set content, register spec and subject), on NSD's responses to the
shared1000 images, so an MMMData score has an NSD counterpart fitted the same way.

Register ``nsd-shared1000``:

  rows       one per shared image with at least ``--min-reps`` trials (default: every repeat
             the subject has, NSD's design gives 3): the image mean of session-z-scored betas
             (``scripts/nsd_shared1000.py extract``), each column then z-scored over images.
  groups     the image; with one row per image, folds are over images.
  columns    the space's units (``func1pt8mm`` voxels or ``fsaverage<ico>`` vertices),
             restricted to the union of ``--rois`` from that space's ``rois.npz`` when given.
  features   a ``TB_SPACES`` name (the shared1000 tables tb-enc reads), or a per-image TSV
             (``stimulus_id`` + feature columns) with ``--bands name=regex,...`` assigning
             columns to bands in order.

The ridge settings (α grid, inner folds, band-weight search) are ``harness.film_spec``'s;
outer folds default to ``harness.OUTER_FOLDS``. ``--cache`` is required: NSD scores never go
into MMMData's derivatives.

Usage:
    python nsd.py plan --subj 1 --space fsaverage6 --root <nsd_shared1000 --out> --features ebind --cache <dir>
    python nsd.py score --subj 1 --space func1pt8mm --root <dir> --features feats.tsv \\
        --bands clip='^clip_',vao='^vao_' --rois phc,prc --outer-folds 10 --cache <dir>
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as h  # noqa: E402  (also puts braintwill and src/python on the path)
from braintwill.encoding import (EncodingSpec, FrameFeatures, ScoreCache, cache_key,  # noqa: E402
                                 feature_set_hash, item_design, score)

REGISTER = "nsd-shared1000"


def unit_dir(root: Path, subj: int, space: str) -> Path:
    d = Path(root) / f"subj{subj:02d}" / space
    if not (d / "image_mean_z.npy").exists():
        raise FileNotFoundError(f"no image_mean_z.npy in {d}; run scripts/nsd_shared1000.py extract "
                                f"(fsaverage is written as fsaverage<ico>)")
    return d


def table_features(path: Path, bands: dict[str, str]) -> dict[str, FrameFeatures]:
    """One-frame features per image from a TSV; ``bands`` maps band name -> column regex."""
    df = pd.read_csv(path, sep="\t").set_index("stimulus_id")
    if df.index.duplicated().any():
        raise RuntimeError(f"{path}: an image appears twice")
    cols, layout = [], {}
    for band, pattern in bands.items():
        got = [c for c in df.columns if re.search(pattern, c) and c not in cols]
        if not got:
            raise KeyError(f"{path}: no column matches band {band!r} ({pattern})")
        layout[band] = slice(len(cols), len(cols) + len(got))
        cols += got
    x = df[cols].to_numpy(np.float32)
    return {sid: FrameFeatures(sid, [0.0], x[i:i + 1], 1.0, layout) for i, sid in enumerate(df.index)}


def parse_bands(text: str | None) -> dict[str, str]:
    if not text:
        raise SystemExit("--bands is required with a TSV feature set (name=regex,...)")
    return dict(part.split("=", 1) for part in text.split(","))


class Request:
    def __init__(self, a) -> None:
        self.dir = unit_dir(a.root, a.subj, a.space)
        images = pd.read_csv(self.dir / "images.tsv", sep="\t")
        min_reps = a.min_reps or int(images.n_reps.max())
        self.rows = images[images.n_reps >= min_reps].reset_index(drop=True)
        self.rois = a.rois.split(",") if a.rois else []
        if self.rois:
            r = np.load(self.dir / "rois.npz")
            missing = [k for k in self.rois if k not in r.files]
            if missing:
                raise SystemExit(f"no ROI {missing} in {self.dir / 'rois.npz'}")
            self.units = np.flatnonzero(np.any([r[k] for k in self.rois], axis=0))
        else:
            self.units = None
        if Path(a.features).suffix in (".tsv", ".txt"):
            feats = table_features(Path(a.features), parse_bands(a.bands))
            self.label = {"features": str(Path(a.features).resolve()), "bands": parse_bands(a.bands)}
        else:
            feats = h.image_features(h.Paths().stimfeat, a.features)
            self.label = {"space": a.features}
        missing = sorted(set(self.rows.stimulus_id) - set(feats))
        if missing:
            raise RuntimeError(f"{len(missing)} images have no features, e.g. {missing[0]}")
        self.feats = {sid: feats[sid] for sid in self.rows.stimulus_id}
        base = h.film_spec(solver=a.solver, n_iter=a.n_iter)
        self.spec = EncodingSpec(ridge=base.ridge, outer_folds=a.outer_folds)
        sha = hashlib.sha256()
        for f in ("trials.tsv", "images.tsv"):
            sha.update((self.dir / f).read_bytes())
        self.register = {"name": REGISTER, "space": a.space, "design": "item",
                         "rows": f"image means of session-z-scored betas, images with >= {min_reps} trials",
                         "groups": "image", "standardize": "z-score per column over images",
                         "columns": {"rois": self.rois or "all units"},
                         "responses": {"betas": "NSD betas_fithrf_GLMdenoise_RR",
                                       "metadata_sha256": sha.hexdigest()}}
        self.sub = f"subj{a.subj:02d}"
        self.key = cache_key(feature_set_hash(self.feats.values()), self.register, self.spec, self.sub)

    def design(self):
        m = np.load(self.dir / "image_mean_z.npy", mmap_mode="r")
        idx = self.rows.shared_idx.to_numpy() - 1
        y = np.asarray(m[idx] if self.units is None else m[np.ix_(idx, self.units)], dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            y = (y - y.mean(0)) / y.std(0)
        x, bands = item_design(self.feats, self.rows.stimulus_id)
        return x, y.astype(np.float32), self.rows.stimulus_id.to_numpy(), bands

    def summary(self) -> dict:
        first = next(iter(self.feats.values()))
        return {"register": REGISTER, "sub": self.sub, "images": len(self.rows),
                "units": None if self.units is None else int(self.units.size),
                "bands": {b: sl.stop - sl.start for b, sl in first.bands.items()},
                "outer_folds": self.spec.outer_folds}


def cmd_plan(a) -> None:
    req = Request(a)
    cache = ScoreCache(a.cache)
    print(json.dumps({**req.summary(), "spec_hash": req.key["spec_hash"], "feature_hash": req.key["feature_hash"],
                      "path": str(cache.path(req.key)), "cached": cache.get(req.key) is not None}, indent=2))


def cmd_score(a) -> None:
    req = Request(a)
    cache = ScoreCache(a.cache)
    hit = cache.get(req.key)
    if hit is not None:
        print(f"cached: {cache.path(req.key)}\n{json.dumps(hit.summary)}")
        return
    t0 = time.time()
    x, y, groups, bands = req.design()
    t_load = time.time() - t0
    result = score(x, y, groups, bands, req.spec, backend=a.backend)
    seconds = {"load": round(t_load, 1), "score": round(time.time() - t0 - t_load, 1)}
    h._ensure_root(Path(a.cache))
    label = {**req.label, "units": None if req.units is None else req.units.tolist()}
    dest = cache.put(req.key, result, h.provenance(a.backend, seconds), label)
    print(f"wrote {dest}\n{json.dumps(result.summary)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    for name in ("plan", "score"):
        p = sub.add_parser(name)
        p.add_argument("--subj", type=int, required=True, help="NSD subject number, 1-8")
        p.add_argument("--space", required=True, help="func1pt8mm or fsaverage<ico> (as nsd_shared1000.py wrote it)")
        p.add_argument("--root", type=Path, required=True, help="nsd_shared1000.py's --out")
        p.add_argument("--features", required=True, help=f"one of {sorted(h.TB_SPACES)}, or a per-image .tsv")
        p.add_argument("--bands", help="TSV features: name=regex,... in band order")
        p.add_argument("--rois", help="comma-separated ROI names; columns = their union")
        p.add_argument("--min-reps", type=int, help="images with at least this many trials (default: the most any has)")
        p.add_argument("--outer-folds", type=int, default=h.OUTER_FOLDS)
        p.add_argument("--solver", choices=("random_search", "hyper_gradient"), default="random_search")
        p.add_argument("--n-iter", type=int, default=None)
        p.add_argument("--backend", default="torch_cuda")
        p.add_argument("--cache", type=Path, required=True, help="score cache root (outside MMMData's derivatives)")
    a = ap.parse_args(argv)
    {"plan": cmd_plan, "score": cmd_score}[a.verb](a)


if __name__ == "__main__":
    main()
