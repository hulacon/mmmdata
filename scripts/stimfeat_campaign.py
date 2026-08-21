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
UNAVAILABLE = {
    "beats": "aud2psy [beats] extra not installed (beat_this missing)",
    "diarize": "aud2psy [diarize] extra not installed (pyannote missing); HF token is in place",
}


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
        "4,000 word x voice audio files (~1 s each)",
        _word_audio_units,
        deferred="DEFERRED 2026-08-20. 64,000 of the manifest's 66,484 cells -- "
                 "96% of the campaign -- and aud2psy has no per-stimulus aggregate "
                 "table, so each ~1 s word would yield ~2 grid rows rather than the "
                 "one row per (word, voice) the analysis wants. Needs an aggregate "
                 "design in aud2psy before it is worth the GPU time. Include with "
                 "--include-deferred.",
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
        "movies", "cue", "viz2psy", "", True,
        "60 movie cue images",
        _cue_units,
    ))

    return S


def _word_audio_units() -> list[Unit]:
    out = []
    for row in twp1000():
        wid = row["stimulus_id"]
        for voice in VOICES:
            col = f"audio_file_{voice}"
            if not row.get(col):
                continue
            out.append(Unit(
                id=f"{wid}_{voice}",
                out_dir=OUT_ROOT / "twp1000" / "word_audio" / voice,
                inputs=[str(STIM_DIR / "twp1000" / row[col])],
                extra=["--stimulus-id", wid],
            ))
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

def unattributed_features(stem: Path) -> dict[str, list[str]]:
    """{table: [columns psytwill cannot attribute to any model]} for one cell.

    Reserved (non-feature) columns are expected to be unattributed and are
    not reported. Returns {} for a fully compliant family.
    """
    from psytwill.features import _resolve_models
    try:
        from psytwill.spaces import RESERVED_COLUMNS
        reserved = set(RESERVED_COLUMNS)
    except ImportError:  # older psytwill; §4.1 carries the canonical list
        reserved = {
            "stimulus_id", "filename", "filepath", "image_idx", "time",
            "onset", "offset", "chunk_idx", "chunk_label", "n_words", "word",
            "word_idx", "sentence_idx", "voice", "speaker", "turn_idx",
        }
    meta = stem.with_suffix(".meta.json")
    if not meta.exists():
        return {}
    sidecar = json.loads(meta.read_text())
    bad: dict[str, list[str]] = {}
    for table in sorted(stem.parent.glob(f"{stem.stem}*.csv")):
        with open(table) as f:
            cols = next(csv.reader(f), [])
        if not cols:
            continue
        _, un = _resolve_models(cols, sidecar)
        # Columns the campaign itself adds as passthrough context (annotator,
        # seg_number, ...) are not features either; they are whatever the
        # input CSV carried. Only flag columns the sidecar claims as features.
        declared = set()
        for entry in sidecar.get("models", {}).values():
            feats = entry.get("features", {})
            declared.update(feats.get("columns", []) or [])
        # Aggregate tables rename `x` to `x_mean` / `x_sd` / ... (§4.1), so a
        # chunks-level column counts as declared if its base name is.
        def is_declared(col: str) -> bool:
            if col in declared:
                return True
            for suffix in ("_mean", "_sd", "_min", "_max"):
                if col.endswith(suffix) and col[: -len(suffix)] in declared:
                    return True
            return False

        offenders = [c for c in un if c not in reserved and is_declared(c)]
        if offenders:
            bad[table.name] = offenders
    return bad


# ---------------------------------------------------------------------------
# Derived input CSVs
# ---------------------------------------------------------------------------
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
        with open(src) as f, open(p, "w", newline="") as out:
            w = csv.writer(out)
            w.writerow(["stimulus_id", "caption_index", "caption"])
            for row in csv.DictReader(f):
                sid = by_coco.get(str(row["cocoId"]))
                if sid is None:
                    unmatched.add(row["cocoId"])
                    continue
                matched.add(sid)
                w.writerow([sid, row["caption_index"], row["caption"]])
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
        for line in r.stdout.splitlines():
            if line.strip().startswith(("PROBLEMS", "REFUSED")) or ": REFUSED" in line:
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
    keep = ["stimulus_id", "annotator", "seg_number", "onset", "offset",
            "duration", "corrected", "description"]
    for out_path, want in ((segb, "B"), (segc, "C")):
        sel = [r for r in rows if r[level_col].upper().endswith(want)]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[c for c in keep if c in rows[0]],
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(sel)


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
            units = src.units()
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
            units = src.units()
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


def cmd_run(args) -> int:
    cells = _filtered(args)
    todo = [(s, u, m) for s, u, m in cells
            if m not in UNAVAILABLE and not is_done(stem_for(s, u, m))]
    skipped = len(cells) - len(todo)
    blocked = sorted({m for s, u, m in cells if m in UNAVAILABLE})

    if not args.dry_run and todo and not _preflight_clean():
        sys.exit("ERROR: stimfeat_preflight.py is not CLEAN. Contract B §4.1 "
                 "requires a clean prefix pre-flight before any extraction. "
                 "Fix: run `python scripts/stimfeat_preflight.py` and resolve "
                 "what it reports.")

    print(f"{len(cells)} cells matched: {len(todo)} to run, {skipped} already done"
          + (f", blocked models skipped: {', '.join(blocked)}" if blocked else ""))

    n_ok = n_fail = 0
    failures = []
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
            bad = {} if args.no_verify else unattributed_features(stem)
            if bad:
                for table, cols in bad.items():
                    print(f"      §4.1 VIOLATION {table}: {len(cols)} feature "
                          f"column(s) psytwill cannot attribute: {cols[:6]}")
                print(f"      Fix: prefix them with the model's registry name "
                      f"in {src.package}, then re-run this cell.")
                n_fail += 1
                failures.append(f"{label} (unprefixed columns)")
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
    for src, unit, model in cells:
        bad = unattributed_features(stem_for(src, unit, model))
        if bad:
            n_bad += 1
            for table, cols in bad.items():
                print(f"  §4.1 {src.key}/{model}[{unit.id}] {table}: "
                      f"{len(cols)} unattributable: {cols[:6]}")
    print(f"\n{len(cells)} families checked, {n_bad} with unattributable "
          f"feature columns")
    return 1 if n_bad else 0


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
    p.add_argument("--no-verify", action="store_true",
                   help="skip the post-write §4.1 attribution check")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("verify", help="§4.1-check every family already in the store")
    add_filters(p)
    p.set_defaults(fn=cmd_verify)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
