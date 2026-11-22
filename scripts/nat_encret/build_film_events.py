#!/usr/bin/env python3
"""Build the film event table: SEG-C events and SEG-B scenes with semantic centrality.

    build_film_events.py build [--out DIR] [--device cpu]
        Write <out>/film_events_segc.csv, film_events_segb.csv and film_events.meta.json.
    build_film_events.py check [--out DIR] [--device cpu]
        Rebuild in memory and compare with the tables on disk; check every NaN is declared and
        that no column prefix collides with a *2psy model's (needs the stimfeat env).

--out defaults to <config output_dir>/stimuli_features/movies. Run in the stimfeat env, which
carries sentence-transformers and openpyxl. Set HF_HOME to the shared cache, as the env's
activate script does, and HF_HUB_OFFLINE=1.

What is in the tables (stimulus-only: nothing here depends on a participant):
  * film_events_segc: one row per SEG-C event (parse_movie_annotations.py, level C). It holds
    times, description text, its SEG-B scene, a ``story`` flag, duration and description length.
    It also holds semantic centrality under all-mpnet-base-v2 (embedded here) and under the
    word2psy SEG-C chunk embeddings already in the store (annot_<model>_chunks.csv).
  * film_events_segb: one row per SEG-B scene. It holds the label, times, and the text of its
    story SEG-C events joined in time order, with mpnet centrality of that text. The stored
    word2psy embeddings are of SEG-B labels, not of the joined text, so only mpnet is computed
    at this level.

Semantic centrality follows Lee & Chen (2022, Nat Commun 13:4235). Embed each node's text and
L2-normalise it. A node's degree is the sum of its cosine similarities to every other story node
of the same film. Centrality is that degree z-scored within the film (population SD). Non-story
nodes are left out of the graph, and their centrality is NaN (declared ``undefined``).

Contract B section 4.1 (schema 1.4): CSV tables keyed by stimulus_id, one .meta.json sidecar,
feature columns carry their model's prefix, and NaN appears only in columns with a ``nulls`` entry.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import encret_io as io
from build_nat_encret import event_grid

STEM = "film_events"
SCHEMA_VERSION = "1.4"
MPNET = "sentence-transformers/all-mpnet-base-v2"
STORED = ("minilm", "clip_text", "clap_text", "ebind_text")  # word2psy SEG-C chunk embeddings

# SEG-C events with no story content: pure black screens, a school logo and a producer tag.
# The real (film, seg_number) keys are load-bearing: the flag records a judgement on these
# particular annotations, and no generic rule reproduces it. A rule on "black screen" text would
# miss the two logos. Title reveals inside a shot that carry story action stay story events.
# Every key must exist in the parsed annotations; the build fails if one does not.
NONSTORY = {
    ("chicken-or-egg", 29),  # "The screen is black."
    ("fool", 28),            # "A black screen"
    ("identity", 22),        # "There is a black screen."
    ("identity", 27),        # "A black screen."
    ("wormwood", 20),        # "A black screen."
    ("wormwood", 1),         # school logo
    ("present", 1),          # producer tag
}

SEGC_COLS = ["stimulus_id", "onset", "offset", "n_words", "filmevent_seg_number", "filmevent_segb",
             "filmevent_story", "filmevent_duration", "filmevent_text"]
SEGB_COLS = ["stimulus_id", "onset", "offset", "n_words", "filmevent_seg_number", "filmevent_label",
             "filmevent_n_events", "filmevent_n_story_events", "filmevent_story", "filmevent_story_duration",
             "filmevent_story_text"]
STRING_COLS = {"filmevent_text", "filmevent_label", "filmevent_story_text"}


def _words(s: pd.Series) -> pd.Series:
    return s.fillna("").str.split().str.len().astype(int)


# ---------------------------------------------------------------- tables
def segc_table(ann: pd.DataFrame, nonstory=NONSTORY) -> pd.DataFrame:
    """One row per SEG-C event, in film then time order."""
    ev = io.segc_events(ann)
    ev["seg_number"] = ev.seg_number.astype(int)
    keys = set(zip(ev.film, ev.seg_number))
    absent = sorted(set(nonstory) - keys)
    if absent:
        raise ValueError(f"NONSTORY keys not in the parsed SEG-C events: {absent}; the annotations changed, "
                         "so re-check the story flag by hand before rebuilding")
    grid = event_grid(ev, io.segb_spans(ann))[["film", "seg_number", "segb"]]
    ev = ev.merge(grid, on=["film", "seg_number"], how="left", validate="1:1")
    story = [int((f, s) not in nonstory) for f, s in zip(ev.film, ev.seg_number)]
    T = pd.DataFrame(dict(stimulus_id=ev.film, onset=ev.onset, offset=ev.offset, n_words=_words(ev.text),
                          filmevent_seg_number=ev.seg_number, filmevent_segb=ev.segb.astype(int),
                          filmevent_story=story, filmevent_duration=ev.offset - ev.onset,
                          filmevent_text=ev.text))
    return T[SEGC_COLS].reset_index(drop=True)


def segb_table(ann: pd.DataFrame, segc: pd.DataFrame) -> pd.DataFrame:
    """One row per SEG-B scene, with the joined text of its story SEG-C events."""
    b = ann[ann.level == "B"]
    B = pd.DataFrame(dict(stimulus_id=b.stimulus_id.values, onset=b.onset.values, offset=b.offset.values,
                          filmevent_seg_number=b.seg_number.astype(int).values,
                          filmevent_label=b.description.fillna("").astype(str).str.strip().values))
    c = segc.sort_values(["stimulus_id", "onset"])
    key = ["stimulus_id", "filmevent_segb"]
    n_all = c.groupby(key).size().rename("filmevent_n_events")
    s = c[c.filmevent_story == 1]
    agg = s.groupby(key).agg(filmevent_n_story_events=("filmevent_text", "size"),
                             filmevent_story_duration=("filmevent_duration", "sum"),
                             filmevent_story_text=("filmevent_text", lambda t: " ".join(x.strip() for x in t)))
    agg = agg.join(n_all, how="outer").reset_index().rename(columns={"filmevent_segb": "filmevent_seg_number"})
    B = B.merge(agg, on=["stimulus_id", "filmevent_seg_number"], how="left", validate="1:1")
    orphan = agg.merge(B[["stimulus_id", "filmevent_seg_number"]], how="left", indicator=True)
    if (orphan._merge == "left_only").any():
        raise ValueError("SEG-C events point at SEG-B scenes the annotations lack")
    for col in ("filmevent_n_events", "filmevent_n_story_events"):
        B[col] = B[col].fillna(0).astype(int)
    B["filmevent_story_duration"] = B.filmevent_story_duration.fillna(0.0)
    B["filmevent_story_text"] = B.filmevent_story_text.fillna("")
    B["filmevent_story"] = (B.filmevent_n_story_events > 0).astype(int)
    B["n_words"] = _words(B.filmevent_story_text)
    return B[SEGB_COLS].sort_values(["stimulus_id", "onset"], kind="stable").reset_index(drop=True)


def centrality(vectors: np.ndarray, film, story) -> tuple[np.ndarray, np.ndarray]:
    """(degree, z) per row. NaN on non-story rows, which are left out of every film's graph.

    degree needs 2 story nodes in the film and z needs 3; below that they are NaN.
    """
    v = np.asarray(vectors, dtype=np.float64)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    film, story = np.asarray(film), np.asarray(story).astype(bool)
    deg = np.full(len(v), np.nan)
    z = np.full(len(v), np.nan)
    for f in pd.unique(film):
        idx = np.flatnonzero((film == f) & story)
        if len(idx) < 2:
            continue
        s = v[idx] @ v[idx].T
        np.fill_diagonal(s, 0.0)
        d = s.sum(axis=1)
        deg[idx] = d
        if len(idx) >= 3:  # two nodes always share one degree, so z would be 0/0
            z[idx] = (d - d.mean()) / d.std(ddof=0)
    return deg, z


# ---------------------------------------------------------------- embeddings
def stored_vectors(sf: Path, model: str, segc: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """word2psy SEG-C chunk embeddings of ``model``, aligned to ``segc`` rows.

    Raises when the stored rows were made from other annotations (times or text differ).
    """
    ch_path = sf / "movies" / f"annot_{model}_chunks.csv"
    meta_path = sf / "movies" / f"annot_{model}.meta.json"
    lab_path, inp_path = sf / "_labels" / "movies_annot_segc.csv", sf / "_inputs" / "movies_annot_segc.csv"
    for p in (ch_path, meta_path, lab_path, inp_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing: the word2psy SEG-C campaign cell for {model} must exist")
    ch = pd.read_csv(ch_path, dtype={"stimulus_id": str}, keep_default_na=False)
    lab = pd.read_csv(lab_path, dtype={"stimulus_id": str}, keep_default_na=False)
    inp = pd.read_csv(inp_path, dtype={"stimulus_id": str}, keep_default_na=False)
    if not (len(ch) == len(lab) == len(inp)) or not (ch.chunk_idx.values == lab.chunk_idx.values).all() \
            or not (ch.stimulus_id.values == lab.stimulus_id.values).all():
        raise ValueError(f"{ch_path.name}: rows do not line up with {lab_path.name}")
    cols = [c for c in ch.columns if re.fullmatch(rf"{re.escape(model)}_\d+", c)]
    key = pd.DataFrame(dict(stimulus_id=lab.stimulus_id, filmevent_seg_number=lab.seg_number.astype(int),
                            onset_s=ch.onset.astype(float), offset_s=ch.offset.astype(float),
                            text_s=inp.description.astype(str).str.strip(), row=np.arange(len(ch))))
    j = segc.merge(key, on=["stimulus_id", "filmevent_seg_number"], how="left", validate="1:1")
    bad = j.row.isna() | ~np.isclose(j.onset, j.onset_s) | ~np.isclose(j.offset, j.offset_s) \
        | (j.filmevent_text != j.text_s)
    if bad.any():
        raise ValueError(f"{ch_path.name}: {int(bad.sum())} SEG-C events differ from the stored chunks (times or "
                         "text); re-run the word2psy SEG-C cell on the current annotations first")
    V = ch[cols].to_numpy(dtype=np.float64)[j.row.astype(int).to_numpy()]
    meta = json.loads(meta_path.read_text())
    prov = dict(path=str(ch_path), sha256=hashlib.sha256(ch_path.read_bytes()).hexdigest(),
                checkpoint=meta["models"][model]["checkpoint"], extractor=meta.get("extractor"),
                extractor_version=meta.get("extractor_version"), created_at=meta.get("created_at"), dim=len(cols))
    return V, prov


def embed_mpnet(texts: list[str], device: str | None = None) -> tuple[np.ndarray, dict]:
    """all-mpnet-base-v2 embeddings, L2-normalised, with provenance and truncation counts."""
    import sentence_transformers
    from huggingface_hub import constants
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MPNET, device=device)
    ntok = np.array([len(model.tokenizer(t)["input_ids"]) for t in texts])
    E = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    # The cache holds only the files sentence-transformers loads, so snapshot_download(local_files_only)
    # calls the snapshot incomplete; the ref file names the revision that was loaded.
    ref = Path(constants.HF_HUB_CACHE) / f"models--{MPNET.replace('/', '--')}" / "refs" / "main"
    revision = ref.read_text().strip() if ref.exists() else f"unresolved (no {ref})"
    prov = dict(checkpoint=MPNET, revision=revision, package="sentence-transformers",
                package_version=sentence_transformers.__version__, device=str(model.device),
                max_seq_length=int(model.max_seq_length), n_texts=len(texts),
                n_truncated=int((ntok > model.max_seq_length).sum()), tokens_max=int(ntok.max()),
                tokens_median=float(np.median(ntok)))
    return np.asarray(E, dtype=np.float64), prov


# ---------------------------------------------------------------- sidecar
def _centrality_entry(model: str, parent: str, prov: dict, tables: list[str]) -> dict:
    p = f"centrality_{model}"
    node = "SEG-C event description" if tables == ["segc"] else \
        "SEG-C event description (segc) or SEG-B scene's joined story text (segb)"
    return {
        "package": "mmmdata", "checkpoint": parent, "embedding": prov,
        "features": {"columns": [f"{p}_degree", f"{p}_z"], "tables": {t: [f"{p}_degree", f"{p}_z"] for t in tables}},
        "nulls": {f"{p}_{c}": {"means": "undefined",
                               "when": "the row is not a story node (filmevent_story = 0), or its film has fewer "
                                       f"than {k} story nodes at this level"} for c, k in (("degree", 2), ("z", 3))},
        "labels": {
            "default": {"interpretability": "nameable", "derivation": "formula-on-model", "derived_from": parent},
            "columns": {
                f"{p}_degree": {"LongName": f"semantic degree ({model})",
                                "Description": f"Sum of cosine similarities between this {node} embedding and "
                                               "those of every other story node of the same film, at the same "
                                               "level (Lee & Chen 2022)."},
                f"{p}_z": {"LongName": f"semantic centrality ({model})",
                           "Description": f"{p}_degree z-scored within film over story nodes (population SD)."},
            },
        },
    }


FILMEVENT_LABELS = {
    "default": {"interpretability": "nameable", "derivation": "bookkeeping"},
    "columns": {
        "filmevent_seg_number": {"Description": "Annotation number of the event (segc: SEG-C) or scene (segb: "
                                                "SEG-B), as parse_movie_annotations.py emits it."},
        "filmevent_segb": {"Description": "SEG-B scene holding the event's midpoint, else the scene whose midpoint "
                                          "is nearest (segb filmevent_seg_number)."},
        "filmevent_story": {"Description": "1 if the node has story content. segc: 0 for black screens and "
                                           "studio/producer tags. segb: 1 if the scene holds a story event.",
                            "Levels": {"0": "non-story", "1": "story"}},
        "filmevent_duration": {"derivation": "formula", "Description": "offset - onset.", "Units": "s"},
        "filmevent_text": {"Description": "Human SEG-C description (annotations, whitespace-trimmed)."},
        "filmevent_label": {"Description": "Human SEG-B label."},
        "filmevent_n_events": {"Description": "SEG-C events assigned to the scene (filmevent_segb)."},
        "filmevent_n_story_events": {"Description": "Of those, story events."},
        "filmevent_story_duration": {"derivation": "formula", "Units": "s",
                                     "Description": "Summed duration of the scene's story SEG-C events. The "
                                                    "annotated SEG-B spans overlap their neighbours in a few films, "
                                                    "so this is not offset - onset."},
        "filmevent_story_text": {"Description": "The scene's story SEG-C descriptions joined in time order with "
                                                "single spaces (the text mpnet embeds at this level)."},
    },
}


def sidecar(segc, segb, out: Path, prov_stored: dict, prov_mpnet: dict, prov_mpnet_b: dict, runtime: float):
    # Read as text: importing mmmget pulls in duckdb, which the stimfeat env does not carry.
    vfile = io.REPO / "src" / "python" / "mmmget" / "_version.py"
    version = re.search(r'__version__\s*=\s*"([^"]+)"', vfile.read_text()).group(1)

    models = {
        "filmevent": {
            "package": "mmmdata", "features": {"columns": sorted(set(SEGC_COLS[4:] + SEGB_COLS[4:])),
                                               "tables": {"segc": SEGC_COLS[4:], "segb": SEGB_COLS[4:]}},
            "dtype": {c: "string" for c in sorted(STRING_COLS)},
            "nulls": {},
            "labels": FILMEVENT_LABELS,
        },
        "centrality_mpnet": _centrality_entry("mpnet", MPNET, dict(segc=prov_mpnet, segb=prov_mpnet_b),
                                              ["segc", "segb"]),
    }
    for m, p in prov_stored.items():
        models[f"centrality_{m}"] = _centrality_entry(m, p["checkpoint"], p, ["segc"])
    return {
        "schema_version": SCHEMA_VERSION,
        "extractor": "mmmdata", "extractor_version": version,
        "code_version": io.code_version(),
        "generator": "mmmdata scripts/nat_encret/build_film_events.py",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "input": {
            "type": "annotations",
            "annotations": "stimuli/movies/movie_annotations (parse_movie_annotations.py, with its corrections)",
            "n_segc": len(segc), "n_segb": len(segb), "n_story_segc": int(segc.filmevent_story.sum()),
            "nonstory": sorted([list(k) for k in NONSTORY]),
            "method": "Lee & Chen 2022 semantic centrality: cosine weighted degree among a film's story nodes, "
                      "z-scored within film (population SD). Embeddings L2-normalised.",
        },
        "output": {t: {"path": str(out / f"{STEM}_{t}.csv"), "rows": len(df), "columns": len(df.columns)}
                   for t, df in (("segc", segc), ("segb", segb))},
        "models": models,
        "total_runtime_sec": round(runtime, 3),
    }


# ---------------------------------------------------------------- build / check
def build(sf: Path, device: str | None = None, embed=embed_mpnet, ann: pd.DataFrame | None = None,
          nonstory=NONSTORY):
    """(segc, segb, provenance) built from the annotations and the stored embeddings."""
    ann = io.annotations() if ann is None else ann
    segc = segc_table(ann, nonstory)
    segb = segb_table(ann, segc)
    prov_stored = {}
    for m in STORED:
        V, prov_stored[m] = stored_vectors(sf, m, segc)
        segc[f"centrality_{m}_degree"], segc[f"centrality_{m}_z"] = centrality(V, segc.stimulus_id, segc.filmevent_story)
    E, prov_c = embed(segc.filmevent_text.tolist(), device)
    segc["centrality_mpnet_degree"], segc["centrality_mpnet_z"] = centrality(E, segc.stimulus_id, segc.filmevent_story)
    has = segb.filmevent_story == 1
    Eb_s, prov_b = embed(segb.filmevent_story_text[has].tolist(), device)
    Eb = np.ones((len(segb), Eb_s.shape[1]))  # non-story rows stay outside the graph; never read
    Eb[has.to_numpy()] = Eb_s
    segb["centrality_mpnet_degree"], segb["centrality_mpnet_z"] = centrality(Eb, segb.stimulus_id, has)
    return segc, segb, dict(stored=prov_stored, mpnet=prov_c, mpnet_b=prov_b)


def undeclared_nans(df: pd.DataFrame, meta: dict) -> list[str]:
    declared = {c for m in meta["models"].values() for c in (m.get("nulls") or {})}
    return [c for c in df.columns if df[c].isna().any() and c not in declared]


def cmd_build(a):
    sf = io.paths()["derivatives"] / "stimuli_features"
    out = (a.out or sf / "movies").resolve()
    t0 = time.time()
    segc, segb, prov = build(sf, a.device)
    meta = sidecar(segc, segb, out, prov["stored"], prov["mpnet"], prov["mpnet_b"], time.time() - t0)
    bad = undeclared_nans(segc, meta) + undeclared_nans(segb, meta)
    if bad:
        raise RuntimeError(f"NaN in undeclared columns: {bad}")
    out.mkdir(parents=True, exist_ok=True)
    for t, df in (("segc", segc), ("segb", segb)):
        df.to_csv(out / f"{STEM}_{t}.csv", index=False, float_format="%.6g")
    (out / f"{STEM}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(dict(out=str(out), segc=len(segc), segb=len(segb),
                          story_segc=int(segc.filmevent_story.sum()), story_segb=int(segb.filmevent_story.sum()),
                          mpnet=prov["mpnet"])))


def family_collisions(columns) -> list[str]:
    """Prefixes here that equal or nest with a *2psy model prefix, or use a reserved name."""
    sys.path.insert(0, str(io.REPO / "scripts"))
    import stimfeat_preflight as pf
    from psytwill.spaces import INDEX_COLUMNS as reserved

    owners, errors = pf.collect_prefixes()
    if errors:
        raise RuntimeError(f"family prefixes unreadable (run in the stimfeat env): {errors[:3]}")
    ours = {"filmevent"} | {f"centrality_{m}" for m in ("mpnet", *STORED)}
    out = [f"{p} vs {q} ({owners[q]})" for p in ours for q in owners
           if p == q or p.startswith(q + "_") or q.startswith(p + "_")]
    out += [f"reserved column used as a feature: {c}" for c in columns
            if c in reserved and c not in {"stimulus_id", "onset", "offset", "n_words"}]
    return out


def table_diffs(fresh: pd.DataFrame, disk: pd.DataFrame) -> list[str]:
    """Columns where ``disk`` differs from ``fresh``: numbers to the CSV's 6 significant digits."""
    if list(disk.columns) != list(fresh.columns) or len(disk) != len(fresh):
        return ["shape or columns differ from a fresh build"]
    out = []
    for c in fresh.columns:
        x, y = fresh[c], disk[c]
        if pd.api.types.is_numeric_dtype(x) and c not in STRING_COLS:
            same = np.allclose(x.astype(float), y.astype(float), rtol=1e-5, atol=1e-5, equal_nan=True)
        else:
            same = (x.astype(str).values == y.astype(str).values).all()
        if not same:
            out.append(f"{c} differs from a fresh build")
    return out


def cmd_check(a):
    sf = io.paths()["derivatives"] / "stimuli_features"
    out = (a.out or sf / "movies").resolve()
    meta = json.loads((out / f"{STEM}.meta.json").read_text())
    fresh_c, fresh_b, _ = build(sf, a.device)
    problems = []
    for t, fresh in (("segc", fresh_c), ("segb", fresh_b)):
        disk = pd.read_csv(out / f"{STEM}_{t}.csv", dtype={"stimulus_id": str}, keep_default_na=False,
                           na_values={c: ["", "NaN", "nan"] for c in fresh.columns if c not in STRING_COLS})
        problems += [f"{t}: {c}" for c in undeclared_nans(disk, meta)]
        problems += [f"{t}: {d}" for d in table_diffs(fresh, disk)]
    try:
        problems += family_collisions(list(fresh_c.columns) + list(fresh_b.columns))
    except (ImportError, RuntimeError) as exc:
        problems.append(f"family prefix check could not run ({exc}); run in the stimfeat env")
    print(json.dumps(dict(out=str(out), problems=problems)))
    return 1 if problems else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="verb", required=True)
    for name in ("build", "check"):
        p = sp.add_parser(name)
        p.add_argument("--out", type=Path, default=None, help="default: <output_dir>/stimuli_features/movies")
        p.add_argument("--device", default=None, help="torch device for mpnet (default: sentence-transformers' choice)")
    a = ap.parse_args(argv)
    return dict(build=cmd_build, check=cmd_check)[a.verb](a) or 0


if __name__ == "__main__":
    sys.exit(main())
