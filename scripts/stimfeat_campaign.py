#!/usr/bin/env python
"""Contract B §4.2 extraction campaign driver: the whole *2psy suite over
all three MMMData stimulus sets, once, resumably.

Replaces the four hand-written `stimfeat_*.sbatch` files and the pre-0.6.0
`run_viz2psy.py`. The campaign is a matrix of **cells**:

    cell = (set, source, model, unit)

`source` is what a model actually consumes -- an image, a video's frames, a
video's soundtrack, a word string, a machine caption, a human caption, a human
scene annotation. One stimulus set feeds several sources, and a source feeds
every model of exactly one package. `unit` is one CLI invocation's worth of
input (all 1,000 images at once; one movie at a time).

Every cell writes one §4.1 family -- `<stem>.csv` (or `<stem>_frames.csv` /
`<stem>_chunks.csv`) plus exactly one `<stem>.meta.json`. The sidecar is the
done-marker: a cell whose `.meta.json` exists is skipped, which makes the whole
campaign idempotent and resumable at model granularity with no state file.

Usage
-----
    stimfeat_campaign.py plan                      # the manifest (start here)
    stimfeat_campaign.py plan --json               # machine-readable
    stimfeat_campaign.py plan --set movies --todo  # filtered
    stimfeat_campaign.py inputs                    # build derived input CSVs
    stimfeat_campaign.py run --set shared1000 --source image --model clip
    stimfeat_campaign.py run --set movies --source frames --unit adventure-time
    stimfeat_campaign.py run --set movies --source audio --dry-run
    stimfeat_campaign.py verify                    # §4.1-check what is written
    stimfeat_campaign.py aggregate --dry-run       # psytwill groups
    stimfeat_campaign.py aggregate --set movies    # build them

Filters (`--set/--source/--model/--unit`) compose and apply to every verb.
`--dry-run` on `run` prints the commands it would execute and exits 0.

Env: /gpfs/projects/hulacon/shared/envs/stimfeat (needs all three extractors).
Pre-flight `stimfeat_preflight.py` must be CLEAN before any run -- `run`
refuses to start otherwise (§4.1).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup -- use config if importable, else fall back to well-known path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))
try:
    from core.config import load_config
    _config = load_config(config_dir=_REPO_ROOT / "config")
    BIDS_ROOT = Path(_config["paths"]["bids_project_dir"])
except Exception:
    BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")

STIM_DIR = BIDS_ROOT / "stimuli"
REGISTRY_DIR = STIM_DIR / "stimulus_registry"
OUT_ROOT = BIDS_ROOT / "derivatives" / "stimuli_features"
# Derived word2psy inputs (word lists, joined captions, parsed annotations).
# Under the store rather than in `stimuli/`: they are generated, and writes
# into the BIDS stimuli tree are blocked anyway.
INPUT_DIR = OUT_ROOT / "_inputs"
# Experiment variables that used to ride along inside the extractor inputs:
# annotator, seg_number, ASR confidence, caption ordinal. They are not
# features and word2psy no longer carries them (0.6.0 made passthrough
# opt-in), so they live here instead -- one row per input row, keyed by
# `chunk_idx`, which is the extractor's own row ordinal. Joining a labels
# table back onto the feature table is (stimulus_id, chunk_idx).
LABEL_DIR = OUT_ROOT / "_labels"

STIMFEAT_ENV = Path("/gpfs/projects/hulacon/shared/envs/stimfeat")
PY = str(STIMFEAT_ENV / "bin" / "python")

GRID_HOP = 0.5  # §4.2.5 shared movie grid, seconds
VOICES = ("echo", "nova", "onyx", "shimmer")


# ---------------------------------------------------------------------------
# Model lists, read from the live registries (never from module filenames --
# word2psy's registry names differ, e.g. `fasttext` <- fasttext_embed.py)
# ---------------------------------------------------------------------------
def registry(package: str) -> list[str]:
    import importlib
    return list(importlib.import_module(f"{package}.cli").MODEL_REGISTRY)


# Cells that cannot run in the current environment. Kept in the matrix and
# reported, never silently dropped: a missing cell must be visible as blocked.
#
# MEASURED, not declared. Each entry names the import an optional extra
# provides; availability is probed against the live interpreter. A
# hand-maintained blocked-list is only correct until someone installs
# something, and its staleness is invisible -- it reads as a real gap.
OPTIONAL_IMPORTS = {
    "beats": ("beat_this", "pip install 'beat_this @ git+https://github.com/CPJKU/beat_this'"),
    "diarize": ("pyannote.audio", "pip install 'pyannote.audio>=4.0' (also needs an HF token and gate acceptance)"),
    "egemaps": ("opensmile", "pip install 'opensmile>=2.5' (audEERING research licence, non-commercial)"),
    "ebind": ("ebind", "pip install 'ebind @ git+https://github.com/encord-team/ebind'"),
    "ebind_audio": ("ebind", "pip install 'ebind @ git+https://github.com/encord-team/ebind'"),
    "ebind_text": ("ebind", "pip install 'ebind @ git+https://github.com/encord-team/ebind'"),
}


def _unavailable() -> dict[str, str]:
    """Probe the live env for each optional extra's import."""
    import importlib.util
    blocked = {}
    for model, (module, fix) in OPTIONAL_IMPORTS.items():
        try:
            present = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            present = False
        if not present:
            blocked[model] = f"{module} not importable -- {fix}"
    return blocked


UNAVAILABLE = _unavailable()


# ---------------------------------------------------------------------------
# Registry readers
# ---------------------------------------------------------------------------
def _tsv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def movies() -> list[dict]:
    return _tsv(REGISTRY_DIR / "movies.tsv")


def shared1000() -> list[dict]:
    return _tsv(REGISTRY_DIR / "shared1000.tsv")


def twp1000() -> list[dict]:
    return _tsv(REGISTRY_DIR / "twp1000.tsv")


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------
@dataclass
class Unit:
    """One CLI invocation's worth of input."""
    id: str                     # "all", or a movie stimulus_id
    out_dir: Path
    inputs: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)   # per-unit CLI args


@dataclass
class Source:
    """A (set, source) pair: one input modality feeding one package."""
    set_: str
    source: str
    package: str                # viz2psy | aud2psy | word2psy
    prefix: str                 # output filename prefix ("" | "caption_" | ...)
    gpu: bool
    what: str                   # one-line description for the manifest
    units_fn: object            # () -> list[Unit]
    depends: str | None = None  # "set/source/model" that must exist first
    note: str | None = None
    deferred: str | None = None  # why this source is out of the current wave
    # Run every unit of one model through a single CLI invocation, so the
    # model's weights load once instead of once per unit. Only worth it when
    # units are SHORT -- for movie-length inputs the load is already amortised
    # and the per-model re-decode would cost more than it saves. aud2psy only
    # (its 0.14.0 --inputs-from); viz2psy/word2psy units are whole sets already.
    batch: bool = False

    def units(self) -> list[Unit]:
        return self.units_fn()

    @property
    def models(self) -> list[str]:
        return registry(self.package)

    @property
    def key(self) -> str:
        return f"{self.set_}/{self.source}"


def _movie_units(subdir_per_movie: bool = True) -> list[Unit]:
    out = []
    for row in movies():
        mid = row["stimulus_id"]
        out.append(Unit(
            id=mid,
            out_dir=OUT_ROOT / "movies" / mid,
            inputs=[str(STIM_DIR / "movies" / row["video_file"])],
            extra=["--stimulus-id", mid],
        ))
    return out


def _image_units() -> list[Unit]:
    paths = [str(STIM_DIR / "shared1000" / r["image_file"]) for r in shared1000()]
    return [Unit(id="all", out_dir=OUT_ROOT / "shared1000", inputs=paths)]


def _cue_units() -> list[Unit]:
    paths = [str(STIM_DIR / "movies" / r["cue_file"]) for r in movies() if r.get("cue_file")]
    return [Unit(id="all", out_dir=OUT_ROOT / "movie_cues", inputs=paths)]


def _text_unit(name: str, out_dir: Path, csv_path: Path,
               text_col: str, id_col: str) -> list[Unit]:
    return [Unit(id=name, out_dir=out_dir, inputs=[str(csv_path)],
                 extra=["--text-column", text_col, "--id-column", id_col])]


def _movie_caption_units() -> list[Unit]:
    out = []
    for row in movies():
        mid = row["stimulus_id"]
        d = OUT_ROOT / "movies" / mid
        out.append(Unit(id=mid, out_dir=d, inputs=[str(d / "caption.csv")],
                        extra=["--text-column", "caption_text",
                               "--id-column", "stimulus_id"]))
    return out


def build_sources() -> list[Source]:
    S = []

    # -- shared1000 -------------------------------------------------------
    S.append(Source(
        "shared1000", "image", "viz2psy", "", True,
        "1,000 NSD images, one invocation per model",
        _image_units,
    ))
    S.append(Source(
        "shared1000", "caption", "word2psy", "caption_", True,
        "BLIP machine captions of the same images, scored as text",
        lambda: _text_unit("all", OUT_ROOT / "shared1000",
                           OUT_ROOT / "shared1000" / "caption.csv",
                           "caption_text", "stimulus_id"),
        depends="shared1000/image/caption",
        note="SECONDARY (Ben, 2026-08-22): humancap covers the semantic-text "
             "role with human language. Keep this arm for the "
             "human-vs-machine caption contrast, not as a semantic space in "
             "its own right.",
    ))
    S.append(Source(
        "shared1000", "humancap", "word2psy", "humancap_", True,
        "5 human COCO captions per image (5,004 rows), scored as text",
        lambda: _text_unit("all", OUT_ROOT / "shared1000",
                           INPUT_DIR / "shared1000_humancap.csv",
                           "caption", "stimulus_id"),
        note="5 rows per stimulus_id; chunk_idx disambiguates. Aggregation to "
             "one row per image is a psytwill-side decision, not an extractor one.",
    ))

    # -- twp1000 ----------------------------------------------------------
    S.append(Source(
        "twp1000", "word", "word2psy", "", True,
        "1,000 spoken-word strings, scored as text",
        lambda: _text_unit("all", OUT_ROOT / "twp1000",
                           INPUT_DIR / "twp1000_words.csv",
                           "word", "stimulus_id"),
    ))
    S.append(Source(
        "twp1000", "word_audio", "aud2psy", "", True,
        "spoken-word audio (~0.54 s each), in the voice actually presented",
        _word_audio_units,
        batch=True,
        note="REOPENED 2026-08-22 (Ben), scoped to the presented voice only: the "
             "word->voice assignment is frozen across subjects, so 1,000 of the "
             "4,000 recordings carry the whole design and the arm is 18,000 cells, "
             "not 72,000. Rationale is aud2psy coverage of the TB components -- a "
             "matched acoustic comparison against NAT, and audio regressors -- "
             "which no other source provides. Cost is compute, not disk: ~25 s per "
             "cell is almost entirely model load, for ~0.54 s of audio. "
             "--all-voices restores the full 4,000.",
    ))

    # -- movies -----------------------------------------------------------
    S.append(Source(
        "movies", "frames", "viz2psy", "", True,
        f"60 movies, frames on the {GRID_HOP} s grid",
        _movie_units,
    ))
    S.append(Source(
        "movies", "audio", "aud2psy", "", True,
        f"60 movie soundtracks on the {GRID_HOP} s grid",
        _movie_units,
    ))
    S.append(Source(
        "movies", "caption", "word2psy", "caption_", True,
        "BLIP captions of each movie's frames, scored as text",
        _movie_caption_units,
        depends="movies/frames/caption",
        note="SECONDARY (Ben, 2026-08-22): the human SEG-B/SEG-C annotations "
             "carry the semantic-text role for movies. BLIP also hallucinates "
             "on stylised frames (the Adventure Time cue reads 'the simpsons "
             "family'). Keep for the human-vs-machine contrast only.",
    ))
    S.append(Source(
        "movies", "annot", "word2psy", "annot_", True,
        "1,726 human SEG-C segment descriptions across 59 movies",
        lambda: _text_unit("segc", OUT_ROOT / "movies",
                           INPUT_DIR / "movies_annot_segc.csv",
                           "description", "stimulus_id"),
        note="annotator is a covariate (one annotator per movie, no double-coding); "
             "body-double is 36% unannotated. Both carried in the input CSV.",
    ))
    S.append(Source(
        "movies", "annot_segb", "word2psy", "annotb_", True,
        "376 human SEG-B event labels across 59 movies",
        lambda: _text_unit("segb", OUT_ROOT / "movies",
                           INPUT_DIR / "movies_annot_segb.csv",
                           "description", "stimulus_id"),
    ))
    S.append(Source(
        "movies", "transcript", "word2psy", "transcript_", True,
        "2,052 ASR speech segments across the 54 movies that contain dialogue",
        lambda: _text_unit("all", OUT_ROOT / "movies",
                           INPUT_DIR / "movies_transcript.csv",
                           "text", "stimulus_id"),
        depends="movies/audio/transcribe",
        note="what characters SAY, the auditory-linguistic counterpart of "
             "movies/caption. 6 movies are silent and contribute no rows. "
             "asr_confidence and no_speech_prob ride along as covariates -- "
             "these are machine transcripts, not a script.",
    ))
    S.append(Source(
        "movies", "cue", "viz2psy", "", True,
        "60 movie cue images",
        _cue_units,
    ))
    S.append(Source(
        "movies", "cue_caption", "word2psy", "caption_", True,
        "BLIP captions of the 60 cue images, scored as text",
        lambda: _text_unit("all", OUT_ROOT / "movie_cues",
                           OUT_ROOT / "movie_cues" / "caption.csv",
                           "caption_text", "stimulus_id"),
        depends="movies/cue/caption",
        note="SECONDARY (Ben, 2026-08-22): the cue images are not shown on "
             "every trial, so this is completeness rather than a space any "
             "analysis is waiting on. Already extracted; cheap to keep.",
    ))

    return S


# Set by --all-voices. Default False: only the voice each word was actually
# presented in. See _word_audio_units for why that is the right default.
ALL_VOICES = False
# Set by --shard I/N; (i, n) 1-based, or None for every unit.
SHARD = None


def _shard(units: list) -> list:
    """Round-robin slice of units for --shard I/N.

    Round-robin rather than contiguous blocks so every shard draws from the
    whole alphabet: units are cost-homogeneous here, but a contiguous split
    would put any future ordering correlation (length, source, voice) entirely
    inside one shard and skew its wall clock.
    """
    if SHARD is None:
        return units
    i, n = SHARD
    return [u for k, u in enumerate(units) if k % n == (i - 1)]


def _word_audio_units() -> list[Unit]:
    """One unit per word, in the voice that word was actually presented in.

    All four recordings of every word exist, but the word -> voice assignment
    is frozen across subjects (registry column `presented_voice`), so 3,000 of
    the 4,000 files are never heard by anyone. Scoring them would quadruple a
    compute-bound arm to describe audio no participant received.

    It would also break Contract B. Every voice of a word carries the same
    `--stimulus-id`, separated only by output directory, so extracting all four
    yields four rows sharing one `stimulus_id` with no voice column -- a
    duplicate-key refusal in `psytwill features`. Restricted to the presented
    voice, each word has exactly one row and the key is unique by construction.

    `--all-voices` restores the full 4,000 for a question that genuinely needs
    the unheard recordings (e.g. voice-identity controls). Its duplicate-key
    consequence is real and is the caller's to resolve.
    """
    out = []
    missing = 0
    for row in twp1000():
        wid = row["stimulus_id"]
        if ALL_VOICES:
            voices = [v for v in VOICES if row.get(f"audio_file_{v}")]
        else:
            pv = (row.get("presented_voice") or "").strip()
            if not pv:
                missing += 1
                continue
            voices = [pv]
        for voice in voices:
            col = f"audio_file_{voice}"
            if not row.get(col):
                continue
            out.append(Unit(
                id=f"{wid}_{voice}",
                out_dir=OUT_ROOT / "twp1000" / "word_audio" / voice,
                inputs=[str(STIM_DIR / "twp1000" / row[col])],
                extra=["--stimulus-id", wid],
            ))
    if missing and not ALL_VOICES:
        print(f"  NOTE  {missing} words have no presented_voice in the registry "
              f"(never presented to any subject yet); rebuild the registry after "
              f"new subjects are BIDSified to pick them up.")
    return out


# ---------------------------------------------------------------------------
# Cell status
# ---------------------------------------------------------------------------
def stem_for(src: Source, unit: Unit, model: str) -> Path:
    """The `-o` stem for one cell. Its `.meta.json` is the done-marker."""
    name = f"{src.prefix}{model}"
    if src.source == "word_audio":
        name = f"{unit.id}_{model}"
    return unit.out_dir / f"{name}.csv"


def is_done(stem: Path) -> bool:
    return stem.with_suffix(".meta.json").exists()


def command_for(src: Source, unit: Unit, model: str) -> list[str]:
    stem = stem_for(src, unit, model)
    if src.package == "viz2psy":
        cmd = [PY, "-m", "viz2psy.cli", model, *unit.inputs,
               "-o", str(stem), "--batch-size", "64", "--no-viz", "--quiet"]
        if src.source in ("frames",):
            cmd += ["--frame-interval", str(GRID_HOP), "--no-save-frames"]
        cmd += unit.extra
        return cmd
    if src.package == "aud2psy":
        return [PY, "-m", "aud2psy.cli", model, *unit.inputs,
                "-o", str(stem), "--hop", str(GRID_HOP), *unit.extra]
    if src.package == "word2psy":
        return [PY, "-m", "word2psy.cli", model, *unit.inputs,
                "-o", str(stem), *unit.extra]
    raise ValueError(f"unknown package {src.package}")


# ---------------------------------------------------------------------------
# Post-write §4.1 compliance
# ---------------------------------------------------------------------------
# The prefix pre-flight checks that the *declared* prefix namespace is
# collision-free. It cannot see what a model actually writes -- word2psy's
# `wordform` and `lexical_norms` emitted bare `length` / `valence` /
# `zipf_frequency` through 0.3.1 while the pre-flight reported CLEAN, because
# every extraction before 2026-08-20 used only embedding models, whose columns
# are prefixed by construction. So every cell is checked against the consumer
# that actually has to attribute the columns: psytwill's own resolver.
#
# A declaration-based gate is a filter with two sides, and this one only ever
# looked at one of them. Through 2026-08-23 it narrowed to `is_declared(c)` --
# flagging only columns the sidecar *claims as features* that psytwill cannot
# attribute -- and so missed three real defects in a row:
#
#   1. `is_downbeat` (aud2psy `beats`): never declared at all, so invisible.
#      `is_declared()` gates on declaration; an undeclared emitted column
#      cannot fail a check that only inspects declarations.
#   2. 335 word-level tables carrying features but no `stimulus_id`: the gate
#      checked column *attributability* and never that the key column exists.
#   3. aud2psy sidecars declare columns as `models.<m>.columns`, while
#      viz2psy/word2psy nest them under `models.<m>.features.columns`. Reading
#      only the nested form left `declared` empty for every aud2psy cell, so
#      even the forward check was a no-op there.
#
# The gate now mirrors the consumer rather than the declarations, and checks
# emission against the sidecar in BOTH directions. psytwill melts every column
# not in its INDEX_COLUMNS and attributes it by model-name prefix, so that --
# not the declared column list -- is the contract a cell has to satisfy.

#: Kinds of §4.1 violation, in report order. Each maps to one directional
#: question about the cell's emitted tables vs. its sidecar.
VIOLATION_KINDS = {
    "unattributable": "emitted feature column matches no model prefix",
    "no_stimulus_id": "table has features but no stimulus_id",
    "undeclared": "emitted, attributable, but absent from the sidecar",
    "not_emitted": "declared as a feature but written to no table",
}

#: Aggregate tables rename `x` to `x_mean` / `x_sd` / ... (§4.1).
_AGG_SUFFIXES = ("_mean", "_sd", "_min", "_max")


def _reserved_columns() -> set[str]:
    """psytwill's INDEX_COLUMNS -- the columns it will not melt as features."""
    try:
        from psytwill.spaces import RESERVED_COLUMNS
        return set(RESERVED_COLUMNS)
    except ImportError:
        pass
    try:  # psytwill >= 0.5 spells it INDEX_COLUMNS
        from psytwill.spaces import INDEX_COLUMNS
        return set(INDEX_COLUMNS)
    except ImportError:  # older psytwill; §4.1 carries the canonical list
        return {
            "stimulus_id", "filename", "filepath", "image_idx", "time",
            "onset", "offset", "chunk_idx", "chunk_label", "n_words", "word",
            "word_idx", "sentence_idx", "voice", "speaker", "turn_idx",
        }


def _declared_columns(sidecar: dict) -> tuple[set[str], set[str]]:
    """(explicitly declared feature columns, models that declared a list).

    §4.1 permits two declaration shapes and the store uses both: viz2psy and
    word2psy nest the list under ``models.<m>.features.columns``; aud2psy puts
    it directly on the entry as ``models.<m>.columns``. Embedding models
    declare a ``pattern`` + ``count`` instead of a list -- they contribute no
    columns here and their model name is left out of the second set, which is
    what keeps `undeclared` / `not_emitted` from firing on 1,024 EBind dims.
    """
    declared: set[str] = set()
    listed: set[str] = set()
    for name, entry in (sidecar.get("models") or {}).items():
        if not isinstance(entry, dict):
            continue
        feats = entry.get("features")
        cols = None
        if isinstance(feats, dict) and isinstance(feats.get("columns"), list):
            cols = feats["columns"]
        elif isinstance(entry.get("columns"), list):
            cols = entry["columns"]
        if cols is None:
            continue
        declared.update(cols)
        listed.add(name)
    return declared, listed


def _family_tables(stem: Path) -> list[Path]:
    """The CSVs of one cell's family: `<stem>.csv` and `<stem>_<table>.csv`.

    Anchored on a `_` boundary, and attributed to the **longest** cell stem
    that claims it. One directory holds many cells whose stems prefix each
    other: `shared1000/image` writes `caption.csv`, and the word2psy models
    that consume those captions write `caption_emotion_chunks.csv`,
    `caption_wordform_words.csv`, and nine more. A bare prefix glob hands all
    of them to the `caption` cell, whose sidecar declares one BLIP model and
    therefore cannot attribute a single `emotion_*` column -- 11 violations
    against the wrong sidecar. The pre-2026-08-23 gate globbed the same way
    and never showed it, because `is_declared()` filtered every one of those
    columns back out.
    """
    base = stem.stem
    # `.stem` on "caption_emotion.meta.json" is "caption_emotion.meta", not
    # the cell stem -- Path.stem strips one suffix, and a sidecar has two.
    siblings = {
        name for name in (
            p.name.removesuffix(".meta.json")
            for p in stem.parent.glob("*.meta.json")
        )
        if name != base and name.startswith(base + "_")
    }

    def claimed_by_sibling(name: str) -> bool:
        return any(name == sib or name.startswith(sib + "_") for sib in siblings)

    return sorted(
        p for p in stem.parent.glob(f"{base}*.csv")
        if (p.stem == base or p.stem.startswith(base + "_"))
        and not claimed_by_sibling(p.stem)
    )


def family_violations(stem: Path) -> list[tuple[str, str, list[str]]]:
    """§4.1 violations for one cell's family as (kind, table, columns).

    Empty for a compliant family. `kind` is a key of VIOLATION_KINDS; `table`
    is the CSV's filename, or the family stem for the family-wide checks.
    """
    from psytwill.features import _resolve_models

    reserved = _reserved_columns()
    meta = stem.with_suffix(".meta.json")
    if not meta.exists():
        return []
    sidecar = json.loads(meta.read_text())
    declared, listed_models = _declared_columns(sidecar)

    def base_name(col: str) -> str:
        for suffix in _AGG_SUFFIXES:
            if col.endswith(suffix):
                return col[: -len(suffix)]
        return col

    out: list[tuple[str, str, list[str]]] = []
    emitted_bases: set[str] = set()

    for table in _family_tables(stem):
        with open(table) as f:
            cols = next(csv.reader(f), [])
        if not cols:
            continue
        features = [c for c in cols if c not in reserved]
        emitted_bases.update(base_name(c) for c in features)

        # Direction 1 -- what psytwill cannot attribute. No declaration
        # filter: an undeclared unprefixed column is exactly the defect the
        # old `is_declared()` narrowing made invisible.
        mapping, un = _resolve_models(features, sidecar)
        if un:
            out.append(("unattributable", table.name, sorted(un)))

        # The key column, which attributability says nothing about. psytwill
        # falls back to `chunk_label` with a warning -- which for
        # `movies/caption` made the caption *text* the stimulus id.
        if features and "stimulus_id" not in cols:
            out.append(("no_stimulus_id", table.name, []))

        # Direction 2 -- emitted and attributable, but the sidecar never says
        # so. Only meaningful for models that declared an explicit list.
        if declared:
            stray = sorted(
                c for c in features
                if base_name(c) not in declared and c not in declared
                and mapping.get(c) in listed_models
            )
            if stray:
                out.append(("undeclared", table.name, stray))

    # Direction 2, the other way -- declared but written nowhere. A sidecar
    # that promises a column the family does not contain is as wrong as one
    # that omits a column it does.
    if declared:
        missing = sorted(
            c for c in declared
            if c not in reserved and c not in emitted_bases
            and base_name(c) not in emitted_bases
        )
        if missing:
            out.append(("not_emitted", stem.name, missing))

    return out


def report_violations(
    label: str,
    violations: list[tuple[str, str, list[str]]],
    indent: str = "      ",
) -> None:
    """Print one cell's violations, one line per (kind, table)."""
    for kind, table, cols in violations:
        detail = f": {cols[:6]}{' ...' if len(cols) > 6 else ''}" if cols else ""
        print(f"{indent}§4.1 {label} {table} [{kind}] "
              f"{VIOLATION_KINDS[kind]}{detail}")


# ---------------------------------------------------------------------------
# Derived input CSVs
# ---------------------------------------------------------------------------
def _write_labels(name: str, rows: list[dict], cols: list[str]) -> Path:
    """Write the experiment-side labels for one derived input.

    `chunk_idx` is written explicitly rather than left implicit in row
    order. word2psy numbers chunks 0..N-1 over the input file, so the two
    agree by construction -- but materialising it means a later reordering
    of the input breaks the join loudly instead of silently re-pairing
    every label with the wrong row.
    """
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    path = LABEL_DIR / name
    # `stimulus_id` and `chunk_idx` are this table's own key and are written
    # below; a caller asking for either would duplicate the column.
    present = [c for c in cols
               if rows and c in rows[0]
               and c not in ("stimulus_id", "chunk_idx")]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stimulus_id", "chunk_idx", *present],
                           extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"stimulus_id": r["stimulus_id"], "chunk_idx": i,
                        **{c: r.get(c) for c in present}})
    return path


def build_inputs(force: bool = False) -> list[str]:
    """Generate the derived word2psy input CSVs. Idempotent."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    made = []

    # twp1000 word strings: stimulus_id IS the word
    p = INPUT_DIR / "twp1000_words.csv"
    if force or not p.exists():
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stimulus_id", "word"])
            for row in twp1000():
                w.writerow([row["stimulus_id"], row["stimulus_id"]])
        made.append(str(p))

    # Human COCO captions joined to registry stimulus_ids on **cocoId**.
    # Not on nsdId: coco_captions.csv numbers nsdId 0-based (2950..72948) while
    # the registry -- and the `nsd02951` inside every stimulus_id -- is 1-based.
    # Joining on nsdId silently matches 12/1000 images to the WRONG captions.
    # cocoId agrees exactly (1000/1000) and has no such convention split.
    p = INPUT_DIR / "shared1000_humancap.csv"
    if force or not p.exists():
        by_coco = {str(r["cocoId"]): r["stimulus_id"] for r in shared1000()}
        src = STIM_DIR / "shared1000" / "coco_captions.csv"
        unmatched = set()
        matched = set()
        kept = []
        with open(src) as f, open(p, "w", newline="") as out:
            w = csv.writer(out)
            w.writerow(["stimulus_id", "caption"])
            for row in csv.DictReader(f):
                sid = by_coco.get(str(row["cocoId"]))
                if sid is None:
                    unmatched.add(row["cocoId"])
                    continue
                matched.add(sid)
                w.writerow([sid, row["caption"]])
                kept.append({"stimulus_id": sid,
                             "caption_index": row["caption_index"]})
        # Which of an image's five captions this is: an ordinal of the
        # annotation set, not a feature of the image.
        _write_labels("shared1000_humancap.csv", kept, ["caption_index"])
        n_reg = len(by_coco)
        if unmatched or len(matched) != n_reg:
            raise SystemExit(
                f"ERROR: COCO caption join is incomplete -- {len(matched)}/{n_reg} "
                f"registry images matched, {len(unmatched)} caption cocoIds "
                f"unknown to the registry. Fix: check that "
                f"{src} and shared1000.tsv describe the same 1,000 images; do "
                f"NOT fall back to nsdId, whose base differs between them.")
        print(f"  humancap: {len(matched)}/{n_reg} images matched on cocoId")
        made.append(str(p))

    # ASR speech segments, concatenated across movies. Built from the
    # aud2psy `transcribe` outputs already in the store rather than from
    # the audio, so this input is a join, not a second inference pass.
    p = INPUT_DIR / "movies_transcript.csv"
    if force or not p.exists():
        # onset/offset stay in the input: they are the stimulus's own
        # coordinates and word2psy cannot derive them. ASR confidences and
        # the segment ordinal are provenance about the transcription, not
        # features of the speech, so they go to the labels table.
        cols = ["stimulus_id", "text", "onset", "offset"]
        # aud2psy 0.15.0 prefixed its feature columns and renamed
        # segment_idx -> chunk_idx (§4.1). The transcript is read here, so
        # the rename lands here too.
        rename = {"transcribe_text": "text"}
        label_rows = []
        n_rows = 0
        silent = []
        with open(p, "w", newline="") as out:
            w = csv.DictWriter(out, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for row in movies():
                sid = row["stimulus_id"]
                src = OUT_ROOT / "movies" / sid / "transcribe_transcript.csv"
                if not src.exists():
                    raise SystemExit(
                        f"ERROR: no transcript for {sid} at {src}. Fix: run "
                        f"`stimfeat_campaign.py run --set movies --source "
                        f"audio --model transcribe` first.")
                with open(src) as f:
                    seg = [{**r, **{v: r[k] for k, v in rename.items() if k in r}}
                           for r in csv.DictReader(f)]
                seg = [r for r in seg if (r.get("text") or "").strip()]
                if not seg:
                    silent.append(sid)
                    continue
                w.writerows(seg)
                # aud2psy's own chunk_idx is the segment's index *within its
                # movie*; this table's chunk_idx is the row ordinal over the
                # concatenation of all movies, which is what word2psy will
                # number against. Two different things, so the upstream one
                # is kept under its own name instead of being clobbered.
                label_rows.extend(
                    {**r, "transcribe_chunk_idx": r.get("chunk_idx")}
                    for r in seg
                )
                n_rows += len(seg)
        _write_labels("movies_transcript.csv", label_rows,
                      ["transcribe_chunk_idx", "transcribe_asr_confidence",
                       "transcribe_no_speech_prob"])
        print(f"  transcript: {n_rows} speech segments from "
              f"{len(movies()) - len(silent)} movies "
              f"({len(silent)} with no speech: {', '.join(silent) or 'none'})")
        made.append(str(p))

    # movie annotations, both levels, via the parser (which applies the
    # declared corrections table and refuses stale ones)
    segc = INPUT_DIR / "movies_annot_segc.csv"
    segb = INPUT_DIR / "movies_annot_segb.csv"
    if force or not (segc.exists() and segb.exists()):
        tsv = INPUT_DIR / "movies_annot_all.tsv"
        # The parser exits 1 whenever it has anything to report, and it always
        # does: body-double's annotation pass is 36% short of the film and no
        # correction can close that. So the TSV existing is the success test,
        # not the return code -- but the report is still echoed, because a
        # *new* problem must not be silently absorbed by a known one.
        r = subprocess.run(
            [sys.executable, str(_SCRIPT_DIR / "parse_movie_annotations.py"),
             "-o", str(tsv)],
            capture_output=True, text=True,
        )
        if not tsv.exists():
            sys.stderr.write(r.stdout + r.stderr)
            raise SystemExit(
                "ERROR: parse_movie_annotations.py wrote no TSV. "
                "Fix: run it with --check and resolve what it reports.")
        # Print the detail lines under a PROBLEMS/REFUSED header too. Printing
        # only the header meant "annotations: PROBLEMS (1):" appeared on every
        # run with nothing saying what the problem was.
        echo = False
        for line in r.stdout.splitlines():
            head = line.strip().startswith(("PROBLEMS", "REFUSED")) or ": REFUSED" in line
            if head:
                echo = True
            elif echo and not line.startswith((" ", "\t")):
                echo = False
            if head or echo:
                print(f"  annotations: {line.strip()}")
        _split_annotations(tsv, segb, segc)
        made += [str(segb), str(segc)]

    return made


def _split_annotations(tsv: Path, segb: Path, segc: Path) -> None:
    """Split the parser's long-form TSV into one CSV per annotation level."""
    rows = _tsv(tsv)
    if not rows:
        raise SystemExit(f"ERROR: {tsv} is empty. Fix: re-run "
                         f"parse_movie_annotations.py --check and read its report.")
    level_col = "level" if "level" in rows[0] else None
    if level_col is None:
        raise SystemExit(
            f"ERROR: {tsv} has no `level` column (found {list(rows[0])}). "
            f"Fix: parse_movie_annotations.py must emit one row per segment "
            f"with a level column naming SEG-B vs SEG-C.")
    # The extractor gets identity, the segment's own bounds, and the text.
    # Who annotated it, which segment number it is, and whether it was
    # corrected are variables of this study, not features of the stimulus.
    keep = ["stimulus_id", "onset", "offset", "description"]
    labels = ["seg_number", "annotator", "duration", "corrected"]
    for out_path, want in ((segb, "B"), (segc, "C")):
        sel = [r for r in rows if r[level_col].upper().endswith(want)]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[c for c in keep if c in rows[0]],
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(sel)
        _write_labels(out_path.name, sel, labels)


# ---------------------------------------------------------------------------
# Verbs
# ---------------------------------------------------------------------------
def _filtered(args) -> list[tuple[Source, Unit, str]]:
    cells = []
    for src in build_sources():
        if args.set and src.set_ != args.set:
            continue
        if args.source and src.source != args.source:
            continue
        if src.deferred and not args.include_deferred:
            continue
        try:
            units = _shard(src.units())
        except FileNotFoundError as e:
            print(f"  warning: {src.key} units unavailable: {e}", file=sys.stderr)
            continue
        models = [m for m in src.models if not args.model or m == args.model]
        for unit in units:
            if args.unit and unit.id != args.unit:
                continue
            for model in models:
                cells.append((src, unit, model))
    return cells


def cmd_plan(args) -> int:
    sources = build_sources()
    rows = []
    for src in sources:
        if args.set and src.set_ != args.set:
            continue
        if args.source and src.source != args.source:
            continue
        if src.deferred and not args.include_deferred:
            rows.append(dict(key=src.key, package=src.package,
                             deferred=src.deferred))
            continue
        try:
            units = _shard(src.units())
        except FileNotFoundError as e:
            rows.append(dict(key=src.key, package=src.package, error=str(e)))
            continue
        models = [m for m in src.models if not args.model or m == args.model]
        runnable = [m for m in models if m not in UNAVAILABLE]
        blocked = [m for m in models if m in UNAVAILABLE]
        done = todo = 0
        for unit in units:
            for model in runnable:
                if is_done(stem_for(src, unit, model)):
                    done += 1
                else:
                    todo += 1
        rows.append(dict(
            key=src.key, package=src.package, what=src.what,
            n_units=len(units), n_models=len(runnable), n_blocked=len(blocked),
            blocked=blocked, cells=len(units) * len(runnable),
            done=done, todo=todo, depends=src.depends, note=src.note,
        ))

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    print(f"Contract B §4.2 campaign manifest   store: {OUT_ROOT}")
    print()
    hdr = f"{'set/source':<22} {'pkg':<9} {'units':>6} {'models':>7} {'cells':>7} {'done':>6} {'todo':>7}"
    print(hdr)
    print("-" * len(hdr))
    tot_cells = tot_done = tot_todo = 0
    for r in rows:
        if "error" in r:
            print(f"{r['key']:<22} {r['package']:<9}  !! {r['error']}")
            continue
        if "deferred" in r:
            print(f"{r['key']:<22} {r['package']:<9}  -- deferred "
                  f"(--include-deferred to plan it)")
            continue
        if args.todo and r["todo"] == 0:
            continue
        print(f"{r['key']:<22} {r['package']:<9} {r['n_units']:>6} "
              f"{r['n_models']:>7} {r['cells']:>7} {r['done']:>6} {r['todo']:>7}")
        tot_cells += r["cells"]; tot_done += r["done"]; tot_todo += r["todo"]
    print("-" * len(hdr))
    print(f"{'TOTAL':<22} {'':<9} {'':>6} {'':>7} {tot_cells:>7} {tot_done:>6} {tot_todo:>7}")

    print()
    for r in rows:
        if r.get("blocked"):
            for m in r["blocked"]:
                print(f"  BLOCKED  {r['key']}/{m}: {UNAVAILABLE[m]}")
    for r in rows:
        if r.get("depends"):
            print(f"  DEPENDS  {r['key']} needs {r['depends']} first")
    for r in rows:
        if r.get("note"):
            print(f"  NOTE     {r['key']}: {r['note']}")
    for r in rows:
        if r.get("deferred"):
            print(f"  DEFERRED {r['key']}: {r['deferred']}")
    return 0


def cmd_inputs(args) -> int:
    made = build_inputs(force=args.force)
    if made:
        for p in made:
            print(f"  wrote {p}")
    else:
        print(f"  all derived inputs already present under {INPUT_DIR} "
              f"(--force to rebuild)")
    return 0


def _preflight_clean() -> bool:
    r = subprocess.run([sys.executable, str(_SCRIPT_DIR / "stimfeat_preflight.py"),
                        "--json"], capture_output=True, text=True)
    return r.returncode == 0



def _split_batchable(todo: list) -> tuple[list, list]:
    """Partition pending cells into batch groups and per-cell leftovers.

    A group is one (source, model): every pending unit of that model, run
    through a single invocation. Grouping by model rather than by output
    directory is what makes the saving worth having -- one load for all 1,000
    words, not one per voice subdirectory.
    """
    groups: dict[tuple, list] = {}
    rest = []
    for src, unit, model in todo:
        if src.batch and src.package == "aud2psy" and len(unit.inputs) == 1:
            groups.setdefault((src.key, model), []).append((src, unit, model))
        else:
            rest.append((src, unit, model))
    return list(groups.values()), rest


def _run_batch_group(group: list, args) -> tuple[int, int, list]:
    """Run one (source, model) group through a single aud2psy invocation."""
    import csv as _csv

    src, _, model = group[0]
    label = f"{src.key}/{model}"
    root = OUT_ROOT / src.set_ / src.source
    manifest_dir = INPUT_DIR / "_batch"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"{src.set_}_{src.source}_{model}.csv"

    rows = []
    missing = []
    for s, unit, m in group:
        if not Path(unit.inputs[0]).exists():
            missing.append(unit.id)
            continue
        stem = stem_for(s, unit, m)
        try:
            rel = stem.relative_to(root)
        except ValueError:
            # A unit that writes outside the source root cannot ride the shared
            # -o; fall back rather than silently relocating its output.
            return _run_group_individually(group, args)
        sid = unit.extra[unit.extra.index("--stimulus-id") + 1] \
            if "--stimulus-id" in unit.extra else unit.id
        rows.append({"path": unit.inputs[0], "stimulus_id": sid, "output": str(rel)})

    if not rows:
        print(f"  {label} SKIP: no inputs present ({len(missing)} missing)")
        return 0, len(group), [f"{label} (missing inputs)"]

    with open(manifest, "w", newline="") as f:
        w = _csv.DictWriter(f, ["path", "stimulus_id", "output"])
        w.writeheader()
        w.writerows(rows)

    cmd = [PY, "-m", "aud2psy.cli", model, "--inputs-from", str(manifest),
           "-o", str(root), "--hop", str(GRID_HOP)]
    if args.dry_run:
        print(f"  [batch] {label}: {len(rows)} units in ONE invocation "
              f"(model loads once)\n      {' '.join(cmd)}")
        return 0, 0, []

    print(f"  [batch] {label}: {len(rows)} units, one model load ...", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd)
    dt = time.time() - t0

    n_ok = n_fail = 0
    failures = []
    for s, unit, m in group:
        stem = stem_for(s, unit, m)
        if not is_done(stem):
            n_fail += 1
            failures.append(f"{s.key}/{m}[{unit.id}]")
            continue
        bad = [] if args.no_verify else family_violations(stem)
        if bad:
            report_violations(f"{s.key}/{m}[{unit.id}]", bad)
            n_fail += 1
            kinds = sorted({kind for kind, _, _ in bad})
            failures.append(f"{s.key}/{m}[{unit.id}] ({', '.join(kinds)})")
        else:
            n_ok += 1
    per = dt / max(len(rows), 1)
    print(f"      {'ok' if not n_fail else 'PARTIAL'}  {dt:.1f}s total, "
          f"{per:.2f}s/unit  ({n_ok} ok, {n_fail} failed, rc={r.returncode})")
    return n_ok, n_fail, failures


def _run_group_individually(group: list, args) -> tuple[int, int, list]:
    """Per-cell fallback when a group cannot share one output root."""
    n_ok = n_fail = 0
    failures = []
    for s, unit, m in group:
        stem = stem_for(s, unit, m)
        stem.parent.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            print(f"  {s.key}/{m}[{unit.id}]\n      {' '.join(command_for(s, unit, m))}")
            continue
        r = subprocess.run(command_for(s, unit, m))
        if r.returncode == 0 and is_done(stem):
            n_ok += 1
        else:
            n_fail += 1
            failures.append(f"{s.key}/{m}[{unit.id}]")
    return n_ok, n_fail, failures


def cmd_run(args) -> int:
    cells = _filtered(args)
    redo = getattr(args, "redo", False)
    todo = [(s, u, m) for s, u, m in cells
            if m not in UNAVAILABLE
            and (redo or not is_done(stem_for(s, u, m)))]
    skipped = len(cells) - len(todo)
    blocked = sorted({m for s, u, m in cells if m in UNAVAILABLE})

    if not args.dry_run and todo and not _preflight_clean():
        sys.exit("ERROR: stimfeat_preflight.py is not CLEAN. Contract B §4.1 "
                 "requires a clean prefix pre-flight before any extraction. "
                 "Fix: run `python scripts/stimfeat_preflight.py` and resolve "
                 "what it reports.")

    print(f"{len(cells)} cells matched: {len(todo)} to run, {skipped} already done"
          + (" (--redo: done cells are being re-extracted)" if redo else "")
          + (f", blocked models skipped: {', '.join(blocked)}" if blocked else ""))

    # Batchable sources run one CLI invocation per model over all their pending
    # units, so the weights load once rather than once per unit. Everything
    # else keeps the per-cell path. The done-marker is unchanged either way:
    # one §4.1 family per cell, its .meta.json written by the same save_result.
    batched, todo = _split_batchable(todo)
    n_ok = n_fail = 0
    failures = []
    for group in batched:
        ok, fail, names = _run_batch_group(group, args)
        n_ok += ok
        n_fail += fail
        failures.extend(names)

    for i, (src, unit, model) in enumerate(todo, 1):
        stem = stem_for(src, unit, model)
        cmd = command_for(src, unit, model)
        label = f"{src.key}/{model}[{unit.id}]"
        if args.dry_run:
            shown = list(cmd)
            if len(unit.inputs) > 3:
                k = cmd.index(unit.inputs[0])
                shown = cmd[:k + 1] + [f"...(+{len(unit.inputs) - 1} inputs)"] + \
                    cmd[k + len(unit.inputs):]
            print(f"  [{i}/{len(todo)}] {label}\n      {' '.join(shown)}")
            continue
        missing = [p for p in unit.inputs if not Path(p).exists()]
        if missing:
            print(f"  [{i}/{len(todo)}] {label} SKIP: input missing: {missing[0]}")
            n_fail += 1
            failures.append(f"{label} (missing input)")
            continue
        stem.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"  [{i}/{len(todo)}] {label} ...", flush=True)
        r = subprocess.run(cmd)
        dt = time.time() - t0
        if r.returncode == 0 and is_done(stem):
            bad = [] if args.no_verify else family_violations(stem)
            if bad:
                report_violations(label, bad)
                print(f"      Fix: in {src.package}, prefix each column with "
                      f"the model's registry name and declare it in the "
                      f"sidecar, then re-run this cell.")
                n_fail += 1
                kinds = sorted({kind for kind, _, _ in bad})
                failures.append(f"{label} ({', '.join(kinds)})")
                if args.fail_fast:
                    break
            else:
                print(f"      ok  {dt:.1f}s")
                n_ok += 1
        else:
            print(f"      FAIL rc={r.returncode} after {dt:.1f}s")
            n_fail += 1
            failures.append(label)
            if args.fail_fast:
                break

    if args.dry_run:
        return 0
    print(f"\nDone: {n_ok} extracted, {skipped} skipped, {n_fail} failed")
    if failures:
        print("Failed cells:")
        for f in failures:
            print(f"  {f}")
        return 1
    return 0


def cmd_verify(args) -> int:
    """Re-check every already-written cell, without re-extracting anything."""
    cells = [(s_, u, m) for s_, u, m in _filtered(args)
             if is_done(stem_for(s_, u, m))]
    n_bad = 0
    by_kind: dict[str, int] = {k: 0 for k in VIOLATION_KINDS}
    for src, unit, model in cells:
        bad = family_violations(stem_for(src, unit, model))
        if bad:
            n_bad += 1
            report_violations(f"{src.key}/{model}[{unit.id}]", bad, indent="  ")
            for kind, _, _ in bad:
                by_kind[kind] += 1
    print(f"\n{len(cells)} families checked, {n_bad} with §4.1 violations")
    for kind, why in VIOLATION_KINDS.items():
        print(f"  {by_kind[kind]:>6}  {kind:<16} {why}")
    return 1 if n_bad else 0


# ---------------------------------------------------------------------------
# Aggregation -- the psytwill `features` surface (settles-when #5)
# ---------------------------------------------------------------------------
# A cell writes one family; a family is 1-3 tables of *different granularity*
# (`main`/`frames` per stimulus or window, `chunks` per text chunk, `words`
# per word). Those granularities cannot share one long table: they key on the
# same (stimulus_id, onset, offset) and a one-word chunk collides with its own
# word row. So the aggregate groups by (set, source, table), not by set.
#
# Two kinds of table are skipped rather than passed to psytwill, which refuses
# both: ones with no feature columns at all (index-only, e.g. a `speakers`
# table), and ones that carry features but no `stimulus_id` (§4.1). The second
# kind should be empty -- word2psy 0.5.1 fixed the guard that caused it -- and
# is reported loudly if it is not.

AGG_DIR = OUT_ROOT / "psytwill"


def _dir_stems(directory: Path) -> list[str]:
    """Every cell stem written into one directory, longest first."""
    stems = _dir_stems.cache.get(directory)
    if stems is None:
        stems = sorted(
            {stem_for(s_, u, m).stem
             for s_ in build_sources()
             for u in s_.units()
             if u.out_dir == directory
             for m in s_.models},
            key=len, reverse=True,
        )
        _dir_stems.cache[directory] = stems
    return stems


_dir_stems.cache = {}


def family_tables(src: Source, unit: Unit, model: str) -> dict[str, Path]:
    """{table name: csv} for one cell.

    A bare `<stem>*.csv` glob over-matches: `caption.csv`'s stem also globs
    `caption_clip_text_chunks.csv`, which belongs to the `caption` *source*
    (prefix `caption_`), not to the `caption` *model*. Longest stem wins.
    """
    stem = stem_for(src, unit, model)
    mine = stem.stem
    others = [n for n in _dir_stems(stem.parent) if len(n) > len(mine)]
    out = {}
    for path in sorted(stem.parent.glob(mine + "*.csv")):
        if any(path.stem == n or path.stem.startswith(n + "_") for n in others):
            continue
        name = path.stem[len(mine):].lstrip("_") or "main"
        out[name] = path
    return out


def _table_usable(path: Path) -> tuple[bool, str]:
    """Can psytwill aggregate this table? (usable, reason-if-not)."""
    from psytwill.spaces import INDEX_COLUMNS

    with open(path, newline="") as f:
        header = next(csv.reader(f), [])
    if not [c for c in header if c not in INDEX_COLUMNS]:
        return False, "no feature columns"
    if "stimulus_id" not in header:
        return False, "no stimulus_id (§4.1)"
    return True, ""


def aggregate_groups(args) -> tuple[dict[tuple[str, str, str], list[Path]],
                                    list[tuple[Path, str]]]:
    """(set, source, table) -> input CSVs, plus the tables that were skipped."""
    groups: dict[tuple[str, str, str], list[Path]] = {}
    skipped: list[tuple[Path, str]] = []
    for src, unit, model in _filtered(args):
        if not is_done(stem_for(src, unit, model)):
            continue
        for table, path in family_tables(src, unit, model).items():
            usable, why = _table_usable(path)
            if not usable:
                skipped.append((path, why))
                continue
            groups.setdefault((src.set_, src.source, table), []).append(path)
    return {k: sorted(v) for k, v in sorted(groups.items())}, skipped


def agg_output(key: tuple[str, str, str]) -> Path:
    set_, source, table = key
    name = f"{set_}_{source}" + ("" if table == "main" else f"_{table}")
    return AGG_DIR / f"{name}_features.parquet"


def cmd_aggregate(args) -> int:
    """Build the psytwill long-form feature table for each group."""
    if getattr(args, "model", None):
        sys.exit(
            "ERROR: `aggregate --model` would write a partial group.\n"
            "  A group parquet is keyed (set, source, table) and holds EVERY "
            "model of that\n"
            "  table; rebuilding it from one model's CSVs silently drops the "
            "others, and\n"
            "  --redo overwrites the good file with the partial one.\n"
            "  Fix: narrow with --set/--source instead. To pick up a single "
            "re-extracted\n"
            "  model, re-run `run --model NAME --redo` and then `aggregate "
            "--redo` over\n"
            "  the (set, source) pairs that contain it."
        )
    groups, skipped = aggregate_groups(args)
    if not groups:
        print("no aggregatable tables matched the filters")
        return 0

    by_reason: dict[str, int] = {}
    for _, why in skipped:
        by_reason[why] = by_reason.get(why, 0) + 1
    if by_reason:
        print("skipped tables psytwill cannot consume:")
        for why, n in sorted(by_reason.items()):
            print(f"  {n:>6}  {why}")
        bad = [p for p, why in skipped if "stimulus_id" in why]
        if bad:
            print(f"  WARNING: {len(bad)} table(s) have features but no "
                  f"stimulus_id -- re-extract with word2psy >= 0.5.1, "
                  f"e.g. {bad[0]}")
        print()

    print(f"{len(groups)} group(s):")
    todo = []
    for key, paths in groups.items():
        out = agg_output(key)
        state = "done" if out.exists() and not args.redo else "todo"
        print(f"  {'/'.join(key):<34} {len(paths):>6} inputs  -> "
              f"{out.name}  [{state}]")
        if state == "todo":
            todo.append((key, paths, out))
    if args.dry_run:
        print(f"\ndry run: {len(todo)} group(s) would be built")
        return 0
    if not todo:
        print("\nnothing to do (use --redo to rebuild)")
        return 0

    from psytwill.features import build_features

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    n_fail = 0
    for key, paths, out in todo:
        label = "/".join(key)
        t0 = time.time()
        try:
            summary = build_features([str(p) for p in paths], output=out)
        except Exception as exc:                       # noqa: BLE001
            n_fail += 1
            print(f"  FAIL {label}: {type(exc).__name__}: {exc}")
            if args.fail_fast:
                return 1
            continue
        print(f"  ok   {label}: {summary['rows']:,} rows, "
              f"{summary['n_stimuli']:,} stimuli, {len(summary['models'])} "
              f"models, {out.stat().st_size/1e6:.0f} MB, "
              f"{time.time() - t0:.0f}s")
    print(f"\n{len(todo) - n_fail}/{len(todo)} group(s) built")
    return 1 if n_fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)

    def add_filters(p):
        p.add_argument("--set", dest="set", metavar="NAME",
                       help="shared1000 | twp1000 | movies")
        p.add_argument("--source", metavar="NAME",
                       help="image | frames | audio | word | caption | humancap | annot | cue")
        p.add_argument("--model", metavar="NAME", help="one registry model name")
        p.add_argument("--unit", metavar="ID", help="one unit id (e.g. a movie stimulus_id)")
        p.add_argument("--shard", metavar="I/N",
                       help="process only shard I of N (1-based), splitting the "
                            "matched UNITS round-robin. For arm-sized sources "
                            "that exceed one job's wall clock; each shard is "
                            "independently resumable like any other slice")
        p.add_argument("--all-voices", action="store_true",
                       help="twp1000/word_audio: score all 4 voices of every "
                            "word, not just the one presented. Quadruples the "
                            "arm and makes stimulus_id non-unique across voices")
        p.add_argument("--include-deferred", action="store_true",
                       help="include sources deferred out of the current wave")

    p = sub.add_parser("plan", help="print the campaign manifest")
    add_filters(p)
    p.add_argument("--json", action="store_true")
    p.add_argument("--todo", action="store_true", help="hide fully-complete sources")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("inputs", help="build the derived word2psy input CSVs")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_inputs)

    p = sub.add_parser("run", help="extract every matched cell that is not done")
    add_filters(p)
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands instead of running them")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--redo", action="store_true",
                   help="re-extract matched cells even if their sidecar says "
                        "done -- for a model whose output shape changed. "
                        "Always narrow with --model/--source first.")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the post-write §4.1 attribution check")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser(
        "aggregate",
        help="build the psytwill long-form feature table per (set, source, table)")
    add_filters(p)
    p.add_argument("--dry-run", action="store_true",
                   help="list the groups and exit")
    p.add_argument("--redo", action="store_true",
                   help="rebuild groups whose parquet already exists")
    p.add_argument("--fail-fast", action="store_true")
    p.set_defaults(fn=cmd_aggregate)

    p = sub.add_parser("verify", help="§4.1-check every family already in the store")
    add_filters(p)
    p.set_defaults(fn=cmd_verify)

    args = ap.parse_args()
    # Module-level because the Source table holds unit builders as zero-arg
    # thunks; threading a flag through every builder to reach one of them
    # would be worse than a single explicit global set once, here.
    global ALL_VOICES, SHARD
    ALL_VOICES = getattr(args, "all_voices", False)
    s = getattr(args, "shard", None)
    if s:
        try:
            i, n = (int(x) for x in s.split("/"))
        except ValueError:
            sys.exit(f"ERROR: --shard wants I/N (1-based), got {s!r}")
        if not (1 <= i <= n):
            sys.exit(f"ERROR: --shard {s} out of range; need 1 <= I <= N")
        SHARD = (i, n)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
