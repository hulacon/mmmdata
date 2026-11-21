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
        draws/<set>/<stimulus_id>.npy
                                 every draw's features per film, (n, frames, k) float32: the
                                 pool's columns through each basis, computed here once
        pool.json                provenance

The harness reads them as arm sources (``harness.py``: ``ctrl:<pool>.<set>`` = the set's
columns, ``ctrl:<pool>.<set>.d<NNN>`` = draw NNN's written features), centred over each film
window as every source is. **The harness never projects.** A cache key hashes a unit's
feature values, and a matrix product's last bits depend on the CPU and the thread count it
ran on: a draw projected in the stack job (GPU node) and again in the reader (CPU node) got
two keys (model-readout log 2026-11-12). Written once, a draw is loaded, never recomputed, so
every job keys it alike.

**The pool.** One or more psytwill release blocks' members, each in its fit-time whitened
coordinates (``SpaceWhitener.transform``; a later block's rows placed on the first block's frames),
plus extra sources that are not members of that block (word-on,
cuts, store models from other tables), each PR-whitened on the pool's rows
(``fit_whitener``, ``scaling="pr"``). The rows are every 0.5 s frame of the films-register
pool films, which are the same for every subject (checked), so one set of draws serves them
all. A set may declare the sources its draws come from (``within``): its draws are then matched
within those columns only and are 0 on the rest. Nothing here reads brain data.

**Absent members (nested gate, DECIDED 2026-11-12).** A member is absent from a row where
any of its columns is undefined (psytwill's rule). matched_subspaces needs a complete matrix,
so a member absent from some rows enters as the harness's nested gate does for a gated
feature: its coordinates centred over the rows where it is present and 0 where it is absent,
plus one ``<source>:present`` column (1 where present). The pool, the set and every draw use
that one representation, so the set is a coordinate subset of the pool.

Verbs:

  build   one pool: load, whiten, gate, write the pool cache, every set's bases and draws
  draws   (re)write every set's draw features from a written pool and its bases (``build``
          ends with it): the pool and the bases are not recomputed, so the set's own units
          keep their keys
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
#: The films store table each release block's members are read from.
BLOCK_TABLES = {"V": "movies_frames", "A": "movies_audio_frames"}

#: The release sets a pool's block members: ``block`` names one block, ``blocks`` several (their
#: members are pool sources by member name, so member names must not repeat across blocks);
#: ``skip`` leaves members out with a reason.
#: ``extra`` sources are not members of a block: ``word-on``, ``cuts``, or
#: ``store:<table>:<model>[.<feature>]`` (a films store model, or one feature of it).
#: ``sets``: each a member list (block members and extra sources by name), or a dict with
#: ``members``, optional ``within`` (the sources its draws are taken from; default every pool
#: source) and optional ``draws`` (False: the set is a source only, with no control draws).
#: ``n``, ``n_pool``, ``tol`` and ``seed`` fix the draws; the harness names draw NNN of a set,
#: so they are part of what a draw is.
#:
#: ``dev-`` pools exist to build and check the route before psytwill-space 0.2.0 fixes the
#: battery; their sets are placeholders, not decisions, and nothing read from them is a result.
#:
#: ``r021`` is release 0.2.1's one shared pool (one arm build per release; mmmdata-agents
#: model-readout 2026-11-19, psytwill-space-next 2026-11-20). Its set lists are copied from
#: those efforts' decisions, which own them; the comments name where each comes from.
_V021 = ("clip", "depth", "dinov2", "ebind", "emonet", "faces_extent", "faces_layout", "gist",
         "llstat", "resmem", "saliency")
_A021 = ("clap", "conversation", "ebind_audio", "egemaps", "loudness", "music_emotion", "onsets",
         "pitch", "psychoacoustic", "rhythm", "speech", "speech_emotion", "speech_rate", "timbre",
         "tonal")
_E_WORD_AUDIO = ("word-on", "cuts", "store:movies_audio_frames:speech_rate.speech_rate_words",
                 "store:movies_audio_frames:onsets", "store:movies_audio_frames:loudness",
                 "store:movies_frames:motion")
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
    "r021": {
        "release": "0.2.1",
        "blocks": ("V", "A"),
        "skip": {},
        # model-readout's E sources (readout-prereg §2.2): raw Contract B columns, PR-whitened
        # here as in dev-v010. onsets / loudness / speech_rate are also A members; as extras they
        # are separate sources (store:...), never the A block's coordinates.
        "extra": _E_WORD_AUDIO,
        "sets": {
            # model-readout 2(b), readout-prereg §2.2: N_full, N_strict, N_-E.
            # R14: draws from the V battery's whitened columns ∪ E's word and audio sources.
            "mr-full": {"members": ("saliency", "emonet", "gist", "llstat", "depth", "faces_extent",
                                    "faces_layout", "resmem") + _E_WORD_AUDIO,
                        "within": _V021 + _E_WORD_AUDIO},
            "mr-strict": {"members": ("gist", "llstat", "store:movies_frames:motion",
                                      "store:movies_audio_frames:onsets",
                                      "store:movies_audio_frames:loudness"),
                          "within": _V021 + _E_WORD_AUDIO},
            "mr-no-e": {"members": ("saliency", "emonet", "gist", "depth", "faces_extent", "faces_layout",
                                    "resmem"),
                        "within": _V021 + _E_WORD_AUDIO},
            # psytwill-space-next step-5 spec §3 (S4): P-interp = the V + A block members whose
            # columns are labelled nameable (0.2.1 labels: every member is all-nameable or
            # all-embedding); -noprobe drops probe members (no zero-shot member is in V or A);
            # -strict keeps formula / formula-on-model members plus those with a passing
            # check-register row on the films domain (speech). Draws from the V + A battery.
            "fr-nameable": {"members": ("depth", "emonet", "faces_extent", "faces_layout", "gist", "llstat",
                                        "resmem", "saliency", "conversation", "egemaps", "loudness",
                                        "music_emotion", "onsets", "pitch", "psychoacoustic", "rhythm",
                                        "speech", "speech_emotion", "speech_rate", "timbre", "tonal"),
                            "within": _V021 + _A021},
            "fr-noprobe": {"members": ("depth", "faces_extent", "faces_layout", "gist", "llstat", "resmem",
                                       "conversation", "egemaps", "loudness", "onsets", "pitch",
                                       "psychoacoustic", "rhythm", "speech", "speech_emotion", "speech_rate",
                                       "timbre", "tonal"),
                           "within": _V021 + _A021},
            "fr-strict": {"members": ("gist", "llstat", "conversation", "egemaps", "loudness", "onsets", "pitch",
                                      "psychoacoustic", "rhythm", "speech", "speech_rate", "timbre", "tonal"),
                          "within": _V021 + _A021},
            # P-embed: the embedding members, one source; no draws (§3 reads draws for P-interp only).
            "fr-embed": {"members": ("clip", "dinov2", "ebind", "clap", "ebind_audio"), "draws": False},
            # Block-vs-member banding (§6): each V + A member alone, in its fit-time coordinates.
            **{f"m-{m}": {"members": (m,), "draws": False} for m in _V021 + _A021},
        },
        "n": 100, "n_pool": 2000, "tol": 0.01, "seed": 0,
    },
}


def pool_blocks(decl: dict) -> tuple[str, ...]:
    """The release blocks a pool declares (``block`` or ``blocks``, exactly one of them)."""
    if ("block" in decl) == ("blocks" in decl):
        raise ValueError("a pool declares exactly one of `block` or `blocks`")
    blocks = (decl["block"],) if "block" in decl else tuple(decl["blocks"])
    unknown = [b for b in blocks if b not in BLOCK_TABLES]
    if not blocks or unknown or len(set(blocks)) != len(blocks):
        raise ValueError(f"pool blocks {blocks}: each once, from {sorted(BLOCK_TABLES)}")
    return blocks


def set_spec(entry) -> dict:
    """A set's declaration as ``{members, within, draws}`` (a bare member list draws from every
    pool source)."""
    if isinstance(entry, dict):
        stray = set(entry) - {"members", "within", "draws"}
        if stray or "members" not in entry:
            raise ValueError(f"set entry keys {sorted(entry)}: expected members [, within, draws]")
        within = entry.get("within")
        return {"members": tuple(entry["members"]), "within": None if within is None else tuple(within),
                "draws": bool(entry.get("draws", True))}
    return {"members": tuple(entry), "within": None, "draws": True}


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


#: Where a set's draw features live (the harness reads them through this, never projects).
DRAWS = "draws"


def draw_features_path(controls: Path, pool: str, name: str, stimulus_id: str) -> Path:
    """One film's draw features of one set: an ``(n, frames, k)`` float32 ``.npy``."""
    return pool_dir(controls, pool) / DRAWS / name / f"{stimulus_id}.npy"


def project_draws(x: np.ndarray, bases: np.ndarray) -> np.ndarray:
    """``(frames, columns)`` pool features through ``(n, columns, k)`` bases ->
    ``(n, frames, k)`` float32 (the product in float64)."""
    x, bases = np.asarray(x, np.float64), np.asarray(bases, np.float64)
    if x.ndim != 2 or bases.ndim != 3 or bases.shape[1] != x.shape[1]:
        raise ValueError(f"pool features {x.shape} and bases {bases.shape} do not compose")
    return np.matmul(x[None], bases).astype(np.float32)


def write_draws(controls: Path, pool: str, sets: list[str], films: list[str]) -> dict:
    """Every set's draw features per film, from the written pool and bases. Returns, per set,
    the draws written and the bytes on disk."""
    out = {}
    for name in sets:
        with np.load(pool_dir(controls, pool) / f"{name}.npz") as z:
            bases = z["bases"]
        if bases.shape[0] == 0:  # a source-only set (``draws: False``)
            continue
        nbytes = 0
        for s in films:
            with np.load(pool_dir(controls, pool) / "pool" / f"{s}.npz") as f:
                x = f["x"]
            dest = draw_features_path(controls, pool, name, s)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.stem + ".tmp.npy")
            np.save(tmp, project_draws(x, bases))
            tmp.replace(dest)
            nbytes += dest.stat().st_size
        out[name] = {"n": int(bases.shape[0]), "k": int(bases.shape[2]), "bytes": int(nbytes)}
        print(f"{pool}.{name}: {bases.shape[0]} draws x {len(films)} films written ({nbytes / 1e9:.2f} GB)", flush=True)
    return out


def _draws_meta(written: dict) -> dict:
    return {"layout": f"{DRAWS}/<set>/<stimulus_id>.npy, (n, frames, k) float32",
            "computed": "pool float32 columns @ float32 bases, in float64, stored float32",
            "written": _dt.datetime.now(_dt.timezone.utc).isoformat(), "mmmdata_commit": _git_head(),
            "sets": written}


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
    pool_blocks(decl)
    for s in decl["extra"]:
        parse_extra(s)
    for name, entry in decl["sets"].items():
        spec = set_spec(entry)
        if spec["within"] is not None:
            outside = [m for m in spec["members"] if m not in spec["within"]]
            if outside:
                raise ValueError(f"set {name!r}: members {outside} are not in its draws' `within` sources")
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


def _block_coords(paths: Paths, rel_path: Path, block: str, skip: dict, films: list[str]):
    """One block's members' fit-time whitened coordinates on every frame of the films."""
    from psytwill.release import load_release
    from psytwill.space import member_source, select_features
    from psytwill.store import align_spaces, load_spaces

    fit = load_release(rel_path).block(block)
    members = [m for m in fit.members if m not in skip]
    table = paths.stimfeat / "psytwill" / f"{BLOCK_TABLES[block]}_features.parquet"
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
    return coords, labels, {m: int(fit.map.whiteners[m].rank) for m in members}, str(table), fit.members


def _onto_rows(name: str, X: np.ndarray, labels: list[str], at: dict[tuple[str, int], int]) -> np.ndarray:
    """A later block's rows placed on the pool rows by (film, frame bin); every pool row must
    have one (its values may be undefined, which the nested gate handles)."""
    out = np.full((len(at), X.shape[1]), np.nan)
    hit = np.zeros(len(at), bool)
    for j, lab in enumerate(labels):
        s, t = lab.split("|")[0], float(lab.split("|")[1])
        i = at.get((s, int(round(t / FRAME_S))))
        if i is not None:
            if hit[i]:
                raise RuntimeError(f"{name}: two rows on pool row {s} t={t}")
            out[i], hit[i] = X[j], True
    if not hit.all():
        raise RuntimeError(f"{name}: {(~hit).sum()} pool rows have no row")
    return out


def _load_block(paths: Paths, decl: dict, films: list[str]):
    """Every declared block's members' fit-time whitened coordinates on every frame of the films,
    on the first block's rows."""
    rel_path = paths.releases / f"psytwill-space_{decl['release']}.json"
    blocks = pool_blocks(decl)
    coords, labels, ranks, tables, seen = {}, None, {}, {}, []
    for b in blocks:
        c, lab, r, table, members = _block_coords(paths, rel_path, b, decl["skip"], films)
        seen += members
        dup = sorted(set(c) & set(coords))
        if dup:
            raise ValueError(f"members {dup} are in more than one pool block")
        if labels is None:
            labels, at = lab, _row_index(lab)[2]
        else:
            c = {m: _onto_rows(f"{b}.{m}", Z, lab, at) for m, Z in c.items()}
        coords.update(c)
        ranks.update(r)
        tables[b] = table
    stray = [m for m in decl["skip"] if m not in seen]
    if stray:
        raise ValueError(f"skip {stray}: not members of blocks {blocks}")
    prov = {"release": str(rel_path), "release_sha256": _sha256(rel_path), "blocks": list(blocks),
            "members": ranks, "skip": decl["skip"], "tables": tables}
    if "block" in decl:  # one-block pools keep their provenance keys
        prov.update(block=decl["block"], table=tables[decl["block"]])
        del prov["blocks"], prov["tables"]
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


def build_sets(X: np.ndarray, bands: dict[str, slice], decl: dict, out: Path, pool: str,
               captured_share, matched_subspaces) -> dict:
    """Each declared set's pool columns, k, captured share and draw bases -> ``<set>.npz``.
    A set with ``within`` takes its draws from those sources' columns only: its share is captured
    within them and its bases are 0 on every other pool column. A ``draws: False`` set gets no
    bases (``(0, columns, k)``) and consumes no randomness. Returns each set's provenance."""
    rng = np.random.RandomState(decl["seed"])
    sets = {}
    for name, entry in decl["sets"].items():  # one RandomState, consumed in set order by sets with draws
        spec = set_spec(entry)
        members = spec["members"]
        idx = set_columns(bands, members)
        k = len(idx)
        within = (np.arange(X.shape[1]) if spec["within"] is None
                  else np.sort(set_columns(bands, spec["within"])))
        Xw = X[:, within]
        pos = np.searchsorted(within, idx)
        Q = np.zeros((len(within), k))
        Q[pos, np.arange(k)] = 1.0
        share = captured_share(Xw, Q)
        bases = np.zeros((0, X.shape[1], k), np.float32)
        info = {"members": list(members), "k": int(k), "share": float(share),
                "within": None if spec["within"] is None else list(spec["within"]), "draws": spec["draws"]}
        extra = {}
        if spec["draws"]:
            ms = matched_subspaces(Xw, k, share, n=decl["n"], n_pool=decl["n_pool"], tol=decl["tol"],
                                   random_state=rng)
            bases = np.zeros((decl["n"], X.shape[1], k), np.float32)
            bases[:, within, :] = np.stack(ms.bases).astype(np.float32)  # 0 outside `within`
            extra = {"shares": ms.shares, "gamma_bracket": np.array(ms.gamma_bracket)}
            info.update(draw_share_range=list(ms.share_range), gamma_bracket=list(ms.gamma_bracket))
        np.savez(out / f"{name}.npz", members=np.array(members), columns=idx, k=k, share=share,
                 within=within, bases=bases, **extra)
        sets[name] = info
        rng_txt = (f", draws {info['draw_share_range'][0]:.4f}-{info['draw_share_range'][1]:.4f}"
                   if spec["draws"] else ", no draws")
        print(f"{pool}.{name}: k {k}, share {share:.4f}{rng_txt}", flush=True)
    return sets


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
    sets = build_sets(X, bands, decl, out, pool, captured_share, matched_subspaces)
    meta = {"pool": pool, "created": _dt.datetime.now(_dt.timezone.utc).isoformat(), **prov,
            "extra": list(decl["extra"]), "films": films, "rows": len(labels), "columns": int(X.shape[1]),
            "bands": {b: [s.start, s.stop] for b, s in bands.items()}, "absent_rows": gated,
            "column_names": names, "pool_sha256": hashlib.sha256(np.ascontiguousarray(X, np.float32)).hexdigest(),
            "draws": {k: decl[k] for k in ("n", "n_pool", "tol", "seed")}, "sets": sets,
            "gate": "nested: present rows centred over present rows, 0 where absent, plus <source>:present",
            "extra_scaling": "PR-whitened on the pool rows (psytwill fit_whitener, scaling pr)",
            "psytwill_version": psytwill.__version__, "mmmdata_commit": _git_head(),
            "development_pool": pool.startswith("dev")}
    meta["draw_features"] = _draws_meta(write_draws(paths.controls, pool, list(sets), films))
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


def cmd_draws(args) -> None:
    path = pool_dir(Paths().controls, args.pool) / "pool.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing; run `control_draws.py build --pool {args.pool}`")
    meta = json.loads(path.read_text())
    meta["draw_features"] = _draws_meta(write_draws(Paths().controls, args.pool, list(meta["sets"]), meta["films"]))
    path.write_text(json.dumps(meta, indent=1))


def cmd_show(args) -> None:
    path = pool_dir(Paths().controls, args.pool) / "pool.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing; run `control_draws.py build --pool {args.pool}`")
    meta = json.loads(path.read_text())
    print(f"{args.pool}: {meta['rows']} rows x {meta['columns']} columns, release {Path(meta['release']).name}, "
          f"psytwill {meta['psytwill_version']}, mmmdata {meta['mmmdata_commit']}")
    for name, s in meta["sets"].items():
        rng = (f"draws {s['draw_share_range'][0]:.4f}-{s['draw_share_range'][1]:.4f}"
               if s.get("draws", True) else "no draws")
        print(f"  {name}: k {s['k']}, share {s['share']:.4f}, {rng}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("build", cmd_build), ("draws", cmd_draws), ("show", cmd_show)):
        p = sub.add_parser(name)
        p.add_argument("--pool", required=True)
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
