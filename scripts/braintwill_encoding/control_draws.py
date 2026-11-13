#!/usr/bin/env python3
"""Matched-subspace control draws for the encoding harness's stacked arms (stimfeat env).

A named set of battery members enters a stacked partition as **one** member, and its share is
read against random subspaces of the whole battery with the same dimensionality and the same
captured share of the battery's variance (psytwill ``subspace.matched_subspaces``; the
control psytwill's ``readout`` uses for stimulus geometry). The harness runs in an env without
psytwill, so the draws are made here, once, and written beside the harness's scores:

    <derivatives>/encoding_harness/controls/<pool>/
        pool/<stimulus_id>.npz   the pool's columns per film (feature-cache format), one band
                                 per pool source
        <set>.npz                the set's pool columns, its k and share, and the n bases
        pool.json                provenance

The harness reads them as arm sources (``harness.py``: ``ctrl:<pool>.<set>`` = the set's
columns, ``ctrl:<pool>.<set>.d<NNN>`` = the pool's columns through basis NNN), centred over
each film window as every source is.

**The pool.** A psytwill release block's members, each in its fit-time whitened coordinates
(``SpaceWhitener.transform``), plus extra sources that are not members of that block (word-on,
cuts, store models from other tables), each PR-whitened on the pool's rows
(``fit_whitener``, ``scaling="pr"``). The rows are every 0.5 s frame of the films-register
pool films, which are the same for every subject (checked), so one set of draws serves them
all. Nothing here reads brain data.

**Absent members (nested gate, DECIDED 2026-11-12).** A member is absent from a row where
any of its columns is undefined (psytwill's rule). matched_subspaces needs a complete matrix,
so a member absent from some rows enters as the harness's nested gate does for a gated
feature: its coordinates centred over the rows where it is present and 0 where it is absent,
plus one ``<source>:present`` column (1 where present). The pool, the set and every draw use
that one representation, so the set is a coordinate subset of the pool.

Verbs:

  build   one pool: load, whiten, gate, write the pool cache and every set's draws
  show    a written pool's provenance and each set's k, share and draw share range

Usage (stimfeat env, on SLURM; ``control_draws.sbatch``):
    python control_draws.py build --pool dev-v010
    python control_draws.py show --pool dev-v010
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT / "src" / "python", HERE.parent / "functional_space", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import films as fm  # noqa: E402
from core.config import load_config  # noqa: E402
from film_events import event_bins, film_word_onsets, load_cuts  # noqa: E402

FRAME_S = 0.5
KEY = ("stimulus_id", "time")
TREE = "encoding_harness"
CONTROLS = "controls"
#: The release sets a pool's block members; ``skip`` leaves members out with a reason.
#: ``extra`` sources are not members of the block: ``word-on``, ``cuts``, or
#: ``store:<table>:<model>[.<feature>]`` (a films store model, or one feature of it).
#: ``sets`` are member lists (block members and extra sources by name). ``n``, ``n_pool``,
#: ``tol`` and ``seed`` fix the draws; the harness names draw NNN of a set, so they are part
#: of what a draw is.
#:
#: ``dev-`` pools exist to build and check the route before psytwill-space 0.2.0 fixes the
#: battery; their sets are placeholders, not decisions, and nothing read from them is a result.
POOLS: dict[str, dict] = {
    "dev-v010": {
        "release": "0.1.0",
        "block": "V",
        # 0.1.0 V was fitted on the pre-0.13.0 places columns; the store holds the fixed ones.
        "skip": {"places": "0.1.0 V was fitted on pre-viz2psy-0.13.0 places columns"},
        "extra": ("word-on", "cuts", "store:movies_audio_frames:speech_rate.speech_rate_words",
                  "store:movies_audio_frames:onsets", "store:movies_audio_frames:loudness",
                  "store:movies_frames:motion"),
        "sets": {
            "nameable": ("saliency", "gist", "yolo", "emonet", "llstat", "depth", "faces_extent",
                         "faces_layout", "aesthetics", "resmem", "word-on", "cuts",
                         "store:movies_audio_frames:speech_rate.speech_rate_words",
                         "store:movies_audio_frames:onsets", "store:movies_audio_frames:loudness",
                         "store:movies_frames:motion"),
            "strict": ("gist", "llstat", "store:movies_audio_frames:onsets",
                       "store:movies_audio_frames:loudness", "store:movies_frames:motion"),
            "no-e": ("saliency", "gist", "yolo", "emonet", "depth", "faces_extent", "faces_layout",
                     "aesthetics", "resmem"),
        },
        "n": 100, "n_pool": 2000, "tol": 0.01, "seed": 0,
    },
}


class Paths:
    def __init__(self) -> None:
        cfg = load_config()["paths"]
        self.derivatives = Path(cfg["output_dir"])
        self.fit_corpora = Path(cfg["fit_corpora_dir"])
        self.stimfeat = self.derivatives / "stimuli_features"
        self.releases = self.fit_corpora / "space" / "releases"
        self.controls = self.derivatives / TREE / CONTROLS


def pool_dir(controls: Path, pool: str) -> Path:
    return Path(controls) / pool


def pool_films(windows: pd.DataFrame) -> list[str]:
    """The films-register pool films (alignment rows, no held-out session), which must be the
    same films for every subject."""
    per = {}
    for sub in fm.SUBJECTS:
        rows = windows[(windows["sub"] == sub) & (windows["role"] == "alignment")]
        if rows["ses"].isin(fm.HELDOUT_SESSIONS).any():
            raise RuntimeError(f"sub-{sub}: alignment rows from held-out sessions")
        per[sub] = frozenset(rows["stimulus_id"])
    if len(set(per.values())) != 1:
        raise RuntimeError(f"the pool films differ across subjects ({ {s: len(v) for s, v in per.items()} }); "
                           "one set of draws cannot serve them all")
    films = sorted(next(iter(per.values())))
    if len(films) != fm.N_POOL:
        raise RuntimeError(f"{len(films)} pool films, expected {fm.N_POOL}")
    return films


def parse_extra(src: str) -> tuple[str, str | None, str | None, str | None]:
    """``(kind, table, model, feature)`` of an extra pool source."""
    if src in ("word-on", "cuts"):
        return src, None, None, None
    kind, _, rest = src.partition(":")
    table, _, mf = rest.partition(":")
    model, _, feature = mf.partition(".")
    if kind != "store" or not table or not model:
        raise ValueError(f"extra source {src!r}: expected word-on, cuts or store:<table>:<model>[.<feature>]")
    return "store", table, model, feature or None


def check_pool(pool: str) -> dict:
    if pool not in POOLS:
        raise ValueError(f"unknown pool {pool!r}: one of {sorted(POOLS)}")
    decl = POOLS[pool]
    for s in decl["extra"]:
        parse_extra(s)
    return decl


def nested_gate(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """A source's coordinates with the nested gate: present rows centred over present rows, 0
    where absent, and the presence column (None when present on every row)."""
    present = np.isfinite(Z).all(axis=1)
    if not present.any():
        raise RuntimeError("a source absent from every pool row")
    if present.all():
        return Z, None
    out = np.zeros_like(Z)
    out[present] = Z[present] - Z[present].mean(axis=0)
    return out, present.astype(np.float64)


def set_columns(bands: dict[str, slice], members: tuple[str, ...]) -> np.ndarray:
    """A set's pool columns (each member's band, its presence column included), in set order."""
    missing = [m for m in members if m not in bands]
    if missing:
        raise ValueError(f"set members {missing} are not pool sources ({list(bands)})")
    return np.concatenate([np.arange(bands[m].start, bands[m].stop) for m in members])


def _load_block(paths: Paths, decl: dict, films: list[str]):
    """Block members' fit-time whitened coordinates on every frame of the films, keyed rows."""
    from psytwill.release import load_release
    from psytwill.space import member_source, select_features
    from psytwill.store import align_spaces, load_spaces

    rel_path = paths.releases / f"psytwill-space_{decl['release']}.json"
    fit = load_release(rel_path).block(decl["block"])
    members = [m for m in fit.members if m not in decl["skip"]]
    stray = [m for m in decl["skip"] if m not in fit.members]
    if stray:
        raise ValueError(f"skip {stray}: not members of block {decl['block']}")
    table = paths.stimfeat / "psytwill" / "movies_frames_features.parquet"
    # whole table: a slice of films can lack a sparse model's columns
    got = load_spaces(table, key=KEY, models=sorted({member_source(m)[0] for m in members}), window=FRAME_S)
    spaces = {}
    for m in members:
        model, cols = member_source(m)
        if model not in got:
            raise RuntimeError(f"{table}: no model {model!r} for member {m!r}")
        sm = got[model] if cols is None else select_features(got[model], cols)
        spaces[m] = select_features(sm, fit.map.whiteners[m].features)
    aligned, labels = align_spaces(spaces)
    keep = np.array([lab.split("|")[0] in set(films) for lab in labels])
    coords = {m: fit.map.whiteners[m].transform(aligned[m].X[keep]) for m in members}
    labels = [lab for lab, k in zip(labels, keep) if k]
    prov = {"release": str(rel_path), "release_sha256": _sha256(rel_path), "block": decl["block"],
            "members": {m: int(fit.map.whiteners[m].rank) for m in members}, "skip": decl["skip"],
            "table": str(table)}
    return coords, labels, prov


def _row_index(labels: list[str]) -> tuple[list[str], np.ndarray, dict[tuple[str, int], int]]:
    sids = [lab.split("|")[0] for lab in labels]
    bins = np.array([int(round(float(lab.split("|")[1]) / FRAME_S)) for lab in labels])
    return sids, bins, {(s, int(b)): i for i, (s, b) in enumerate(zip(sids, bins))}


def _load_extra(paths: Paths, src: str, labels: list[str]) -> np.ndarray:
    """An extra source's raw columns on the pool rows (NaN where it has no value)."""
    from psytwill.space import select_features
    from psytwill.store import load_spaces

    kind, table, model, feature = parse_extra(src)
    sids, bins, at = _row_index(labels)
    films = sorted(set(sids))
    n_bins = {s: int(bins[[i for i, x in enumerate(sids) if x == s]].max()) + 1 for s in films}
    if kind in ("word-on", "cuts"):
        out = np.zeros((len(labels), 1))
        if kind == "word-on":
            on = film_word_onsets(paths.stimfeat)
            ev = {s: on.loc[on["stimulus_id"] == s, "onset"].to_numpy(float) for s in films}
        else:
            cuts = load_cuts(paths.fit_corpora)
            missing = sorted(set(films) - set(cuts["stimulus_id"]))
            if missing:
                raise RuntimeError(f"cut table holds no cuts for films {missing}")
            ev = {s: cuts.loc[cuts["stimulus_id"] == s, "bin"].to_numpy(float) * FRAME_S for s in films}
        for s in films:
            hit = event_bins(ev[s], n_bins[s], FRAME_S, f"{s} {kind}")
            for b in np.flatnonzero(hit):
                if (s, int(b)) in at:
                    out[at[(s, int(b))], 0] = 1.0
        return out
    path = paths.stimfeat / "psytwill" / f"{table}_features.parquet"
    got = load_spaces(path, key=KEY, models=[model], window=FRAME_S, stimulus_ids=films)
    if model not in got:
        raise RuntimeError(f"{path}: no model {model!r}")
    sm = got[model] if feature is None else select_features(got[model], [feature])
    out = np.full((len(labels), sm.X.shape[1]), np.nan)
    hits = 0
    for j, lab in enumerate(sm.labels):
        s, t = lab.split("|")[0], float(lab.split("|")[1])
        i = at.get((s, int(round(t / FRAME_S))))
        if i is not None:
            out[i] = sm.X[j]
            hits += 1
    if hits < len(labels):
        raise RuntimeError(f"{src}: {len(labels) - hits} pool rows have no row in {path.name}")
    return out


def _pr_whiten(name: str, X: np.ndarray) -> np.ndarray:
    """An extra source PR-whitened on the pool rows where it is complete (NaN rows stay NaN)."""
    from psytwill.space import fit_whitener
    from psytwill.store import SpaceMatrix

    ok = np.isfinite(X).all(axis=1)
    keep = X.std(axis=0, where=ok[:, None]) > 0 if ok.any() else np.zeros(X.shape[1], bool)
    if not keep.any():
        raise RuntimeError(f"{name}: constant on the pool rows")
    X = X[:, keep]
    feats = [f"{name}_{j}" for j in range(X.shape[1])]
    sm = SpaceMatrix(name=name, labels=[str(i) for i in range(int(ok.sum()))], X=X[ok], features=feats)
    w = fit_whitener(sm, X[ok], scaling="pr")
    out = np.full((X.shape[0], w.rank), np.nan)
    out[ok] = w.transform(X[ok])
    return out


def build_pool(paths: Paths, pool: str) -> dict:
    from psytwill.subspace import captured_share, matched_subspaces
    import psytwill

    decl = check_pool(pool)
    films = pool_films(fm.load_windows(paths.derivatives))
    coords, labels, prov = _load_block(paths, decl, films)
    for src in decl["extra"]:
        coords[src] = _pr_whiten(src, _load_extra(paths, src, labels))
    parts, bands, names, gated, at = [], {}, [], {}, 0
    for src, Z in coords.items():
        Zg, present = nested_gate(Z)
        cols = [Zg] + ([present[:, None]] if present is not None else [])
        width = sum(c.shape[1] for c in cols)
        bands[src] = slice(at, at + width)
        names += [f"{src}_{j:03d}" for j in range(Zg.shape[1])] + ([f"{src}:present"] if present is not None else [])
        gated[src] = None if present is None else int((present == 0).sum())
        parts += cols
        at += width
    X = np.concatenate(parts, axis=1)
    if not np.isfinite(X).all():
        raise RuntimeError("pool has undefined values after gating")
    out = pool_dir(paths.controls, pool)
    sids, bins, _ = _row_index(labels)
    sids = np.array(sids)
    for s in films:
        idx = np.flatnonzero(sids == s)
        idx = idx[np.argsort(bins[idx])]
        if not np.array_equal(bins[idx], np.arange(bins[idx].max() + 1)):
            raise RuntimeError(f"{s}: pool rows do not cover every grid bin from 0")
        dest = out / "pool" / f"{s}.npz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dest, stimulus_id=s, times=bins[idx] * FRAME_S, x=X[idx].astype(np.float32),
                 band_names=np.array(list(bands)), band_bounds=np.array([[b.start, b.stop] for b in bands.values()]))
    rng = np.random.RandomState(decl["seed"])
    sets = {}
    for name, members in decl["sets"].items():  # one RandomState, consumed in set order
        idx = set_columns(bands, members)
        k = len(idx)
        Q = np.zeros((X.shape[1], k))
        Q[idx, np.arange(k)] = 1.0
        share = captured_share(X, Q)
        ms = matched_subspaces(X, k, share, n=decl["n"], n_pool=decl["n_pool"], tol=decl["tol"], random_state=rng)
        np.savez(out / f"{name}.npz", members=np.array(members), columns=idx, k=k, share=share,
                 bases=np.stack(ms.bases).astype(np.float32), shares=ms.shares,
                 gamma_bracket=np.array(ms.gamma_bracket))
        sets[name] = {"members": list(members), "k": int(k), "share": float(share),
                      "draw_share_range": list(ms.share_range), "gamma_bracket": list(ms.gamma_bracket)}
        print(f"{pool}.{name}: k {k}, share {share:.4f}, draws {ms.share_range[0]:.4f}-{ms.share_range[1]:.4f}",
              flush=True)
    meta = {"pool": pool, "created": _dt.datetime.now(_dt.timezone.utc).isoformat(), **prov,
            "extra": list(decl["extra"]), "films": films, "rows": len(labels), "columns": int(X.shape[1]),
            "bands": {b: [s.start, s.stop] for b, s in bands.items()}, "absent_rows": gated,
            "column_names": names, "pool_sha256": hashlib.sha256(np.ascontiguousarray(X, np.float32)).hexdigest(),
            "draws": {k: decl[k] for k in ("n", "n_pool", "tol", "seed")}, "sets": sets,
            "gate": "nested: present rows centred over present rows, 0 where absent, plus <source>:present",
            "extra_scaling": "PR-whitened on the pool rows (psytwill fit_whitener, scaling pr)",
            "psytwill_version": psytwill.__version__, "mmmdata_commit": _git_head(),
            "development_pool": pool.startswith("dev")}
    (out / "pool.json").write_text(json.dumps(meta, indent=1))
    return meta


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def cmd_build(args) -> None:
    meta = build_pool(Paths(), args.pool)
    print(f"{args.pool}: {meta['rows']} rows x {meta['columns']} columns on {len(meta['films'])} films")


def cmd_show(args) -> None:
    path = pool_dir(Paths().controls, args.pool) / "pool.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing; run `control_draws.py build --pool {args.pool}`")
    meta = json.loads(path.read_text())
    print(f"{args.pool}: {meta['rows']} rows x {meta['columns']} columns, release {Path(meta['release']).name}, "
          f"psytwill {meta['psytwill_version']}, mmmdata {meta['mmmdata_commit']}")
    for name, s in meta["sets"].items():
        print(f"  {name}: k {s['k']}, share {s['share']:.4f}, draws {s['draw_share_range'][0]:.4f}-"
              f"{s['draw_share_range'][1]:.4f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("build", cmd_build), ("show", cmd_show)):
        p = sub.add_parser(name)
        p.add_argument("--pool", required=True)
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
