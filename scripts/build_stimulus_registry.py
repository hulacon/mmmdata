#!/usr/bin/env python3
"""
Build the dataset-owned stimulus_id registry (Contract A/B, contracts §4.2).

Writes one TSV per stimulus set into <bids_root>/stimuli/stimulus_registry/,
giving every stimulus one canonical `stimulus_id` and resolving every naming
convention in use (video/cue/annotation stems for movies, image filenames for
shared1000, word×voice audio for twp1000). Consumers (psytwill, the catalog
stimulus tier, agent tools) validate against these tables and never invent ids.

stimulus_id conventions (ratified 2026-08-17, workbench/stimulus-registry):
- movies:     kebab-case normalized title, leading article dropped
              ("adventure-time", "queen-of-basketball", "table-7")
- shared1000: image filename stem ("shared0001_nsd02951") — matches the
              §4.1 extractor default
- twp1000:    the word itself; voice is a reserved column on events/features

events→id rules (verified against all sub-03/04/05 events, 2026-08-17):
- TB/FIN image trials:  events.mmmId  → shared1000.tsv mmmId
- TB/FIN word trials:   (events.word, events.voice) → twp1000.tsv + voice
- NAT trials:           case-insensitive events.movie_name → movies.tsv
                        movie_name / movie_name_variants (closed set)

Usage:
    python build_stimulus_registry.py                # write all three TSVs
    python build_stimulus_registry.py --check        # exit 1 if on-disk TSVs differ
    python build_stimulus_registry.py --validate-events  # assert every events
                                                     # reference resolves
"""

import argparse
import csv
import io
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — use config if importable, else fall back to well-known path
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
VOICES = ("echo", "nova", "onyx", "shimmer")

VIDEO_AFFIX = "_trimmed_normalized_filtered"
ANNOT_RE = re.compile(r"_annotation_master_[A-Z]+$")

# Movies present in the annotation tree but never shown in the experiment.
UNUSED_ANNOTATIONS = {"fargo", "migration", "paddington"}
# Known gaps that should warn, not fail.
KNOWN_MISSING_ANNOTATIONS = {"finders-fee"}


def norm_tokens(title: str) -> list[str]:
    """Normalize a movie title or file stem to comparable tokens."""
    s = unicodedata.normalize("NFKD", title).lower().replace("&", " and ")
    s = re.sub(r"[’']", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    toks = []
    for t in s.split():
        if t == "s" and toks:  # possessive split by punctuation ("miner s")
            toks[-1] += "s"
        elif t not in {"the", "a", "an", "of"}:
            toks.append(t)
    return toks


def slug(title: str) -> str:
    return "-".join(norm_tokens(title))


def read_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def match_file(name_toks: list[str], pool: dict[str, list[str]], kind: str,
               title: str) -> str | None:
    """Resolve a title against a {filename: tokens} pool.

    Exact token match wins; else one token list being a prefix of the other;
    else subset. Ambiguity is an error — the registry must be deterministic.
    """
    hits = []
    for fname, toks in pool.items():
        if toks == name_toks:
            hits.append((3, fname))
        elif toks[: len(name_toks)] == name_toks or name_toks[: len(toks)] == toks:
            hits.append((2, fname))
        elif set(name_toks) <= set(toks) or set(toks) <= set(name_toks):
            hits.append((1, fname))
    if not hits:
        return None
    best = max(score for score, _ in hits)
    top = sorted(fname for score, fname in hits if score == best)
    if len(top) > 1:
        sys.exit(f"ERROR: {kind} match for '{title}' is ambiguous: {top}. "
                 f"Fix: disambiguate the filenames or extend norm_tokens().")
    return top[0]


# ---------------------------------------------------------------------------
# Per-set builders — each returns (header, rows) with rows fully validated
# ---------------------------------------------------------------------------

def scan_nat_events() -> tuple[Counter, dict]:
    """Collect movie_name spellings (with counts) and movie_length values."""
    counts = Counter()
    durations = defaultdict(set)
    events = sorted(BIDS_ROOT.glob("sub-*/ses-*/func/*task-NAT*_events.tsv"))
    if not events:
        sys.exit(f"ERROR: no NAT events files under {BIDS_ROOT}. "
                 f"Fix: check bids_project_dir in config/base.toml.")
    for path in events:
        with open(path, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                name = (row.get("movie_name") or "").strip()
                if not name:
                    continue
                counts[name] += 1
                length = (row.get("movie_length") or "").strip()
                if length:
                    durations[name.lower()].add(float(length))
    return counts, durations


# The movie schedule was authored in Google Sheets, so its exported name
# carries spaces. Accept the tool-safe spelling too, so de-spacing the file
# (staleness-audit 2026-08 section 4) does not have to be atomic with this code.
MOVIE_SCHEDULE_NAMES = ("MMM_movies_Sheet1.csv", "MMM movies - Sheet1.csv")


def movie_schedule_path(mov: Path) -> Path:
    """First existing movie-schedule CSV, tool-safe spelling preferred."""
    for name in MOVIE_SCHEDULE_NAMES:
        candidate = mov / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No movie schedule in {mov}. Expected one of "
        f"{', '.join(MOVIE_SCHEDULE_NAMES)}; export the Google Sheet there.")


def build_movies() -> tuple[list[str], list[list[str]]]:
    mov = STIM_DIR / "movies"
    videos = {f.name: norm_tokens(f.stem.replace(VIDEO_AFFIX, ""))
              for f in (mov / "movie_files").iterdir()}
    cues = {f.name: norm_tokens(f.stem.replace("_cue", ""))
            for f in (mov / "movie_cues").iterdir()}
    annots = {f.name: norm_tokens(ANNOT_RE.sub("", f.stem))
              for f in (mov / "movie_annotations").glob("*.xlsx")}

    styles = {}
    for row in read_csv_rows(movie_schedule_path(mov)):
        name, style = (row.get("Movie name") or "").strip(), (row.get("Movie style") or "").strip()
        if name and style:
            styles.setdefault(name.lower(), style)

    counts, durations = scan_nat_events()
    by_key = defaultdict(list)
    for name, n in counts.items():
        by_key[name.lower()].append((n, name))

    rows = []
    for key in sorted(by_key):
        variants = sorted(by_key[key], reverse=True)
        canonical = variants[0][1]
        toks = norm_tokens(canonical)
        video = match_file(toks, videos, "video", canonical)
        cue = match_file(toks, cues, "cue", canonical)
        annot = match_file(toks, annots, "annotation", canonical)
        for kind, val in (("video", video), ("cue", cue)):
            if val is None:
                sys.exit(f"ERROR: no {kind} file matches '{canonical}'. "
                         f"Fix: add the file or record it as a known gap.")
        sid = slug(canonical)
        if annot is None:
            if sid not in KNOWN_MISSING_ANNOTATIONS:
                sys.exit(f"ERROR: no annotation matches '{canonical}' and it is "
                         f"not in KNOWN_MISSING_ANNOTATIONS.")
            print(f"  warning: '{canonical}' has no annotation file (known gap)")
        dur = sorted(durations[key])
        if len(dur) != 1:
            sys.exit(f"ERROR: conflicting movie_length for '{canonical}': {dur}.")
        if key not in styles:
            sys.exit(f"ERROR: '{canonical}' missing from the movies sheet.")
        rows.append([sid, canonical, "|".join(v for _, v in variants[1:]),
                     f"movie_files/{video}", f"movie_cues/{cue}",
                     f"movie_annotations/{annot}" if annot else "",
                     styles[key], f"{dur[0]:g}"])

    unmatched = set(annots) - {r[5].split("/", 1)[1] for r in rows if r[5]}
    unexpected = {f for f in unmatched
                  if not set(norm_tokens(ANNOT_RE.sub("", Path(f).stem))) & UNUSED_ANNOTATIONS}
    if unexpected:
        sys.exit(f"ERROR: annotation files match no movie and are not known-unused: "
                 f"{sorted(unexpected)}")

    header = ["stimulus_id", "movie_name", "movie_name_variants", "video_file",
              "cue_file", "annotation_file", "style", "duration_s"]
    return header, rows


def build_shared1000() -> tuple[list[str], list[list[str]]]:
    root = STIM_DIR / "shared1000"
    rows = []
    for row in read_csv_rows(root / "shared1000.csv"):
        fname = row["fileNames"]
        if not (root / "images" / fname).exists():
            sys.exit(f"ERROR: shared1000 image missing on disk: images/{fname}")
        rows.append([Path(fname).stem, f"images/{fname}", row["mmmId"],
                     row["nsdId"], row["cocoId"], row["cocoSplit"]])
    if len(rows) != 1000:
        sys.exit(f"ERROR: shared1000.csv has {len(rows)} rows, expected 1000.")
    for col, idx in (("stimulus_id", 0), ("mmmId", 2)):
        vals = [r[idx] for r in rows]
        if len(set(vals)) != len(vals):
            sys.exit(f"ERROR: duplicate {col} values in shared1000.")
    rows.sort(key=lambda r: int(r[2]))
    header = ["stimulus_id", "image_file", "mmmId", "nsdId", "cocoId", "cocoSplit"]
    return header, rows


def build_twp1000() -> tuple[list[str], list[list[str]]]:
    root = STIM_DIR / "twp1000"
    rows = []
    for row in read_csv_rows(root / "twp1000.csv"):
        word, itmno = row["word"], row["itmno"]
        files = []
        for voice in VOICES:
            rel = f"{voice}/{word}_{voice}.mp3"
            if not (root / rel).exists():
                sys.exit(f"ERROR: twp1000 audio missing on disk: {rel}")
            files.append(rel)
        rows.append([word, itmno] + files)
    if len(rows) != 1000:
        sys.exit(f"ERROR: twp1000.csv has {len(rows)} rows, expected 1000.")
    words = [r[0] for r in rows]
    if len(set(words)) != len(words):
        sys.exit("ERROR: duplicate words in twp1000.")
    rows.sort(key=lambda r: int(r[1]))
    header = ["stimulus_id", "itmno"] + [f"audio_file_{v}" for v in VOICES]
    return header, rows


BUILDERS = {"movies": build_movies, "shared1000": build_shared1000,
            "twp1000": build_twp1000}


def render(header: list[str], rows: list[list[str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Events validation — the charter's settle condition, runnable on demand
# ---------------------------------------------------------------------------

def validate_events(tables: dict[str, str]) -> None:
    """Assert every stimulus-bearing events row resolves to a registry row."""
    def col(table, name):
        lines = tables[table].splitlines()
        idx = lines[0].split("\t").index(name)
        return [ln.split("\t")[idx] for ln in lines[1:]]

    movie_names = {n.lower() for n in col("movies", "movie_name")}
    for variants in col("movies", "movie_name_variants"):
        movie_names |= {v.lower() for v in variants.split("|") if v}
    mmm_ids = set(col("shared1000", "mmmId"))
    words = set(col("twp1000", "stimulus_id"))

    def cell(row, key):
        v = (row.get(key) or "").strip()
        return "" if v.lower() in {"n/a", "nan"} else v

    bad = defaultdict(int)
    n_files = n_rows = 0
    for path in sorted(BIDS_ROOT.glob("sub-*/ses-*/func/*_events.tsv")):
        n_files += 1
        with open(path, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                name = cell(row, "movie_name")
                if name:
                    n_rows += 1
                    if name.lower() not in movie_names:
                        bad[f"movie_name '{name}'"] += 1
                mmm = cell(row, "mmmId")
                if mmm:
                    n_rows += 1
                    if str(int(float(mmm))) not in mmm_ids:
                        bad[f"mmmId {mmm}"] += 1
                word = cell(row, "word")
                if word:
                    n_rows += 1
                    if word not in words:
                        bad[f"word '{word}'"] += 1
                    voice = cell(row, "voice")
                    if voice and voice not in VOICES:
                        bad[f"voice '{voice}'"] += 1
    if bad:
        for ref, n in sorted(bad.items()):
            print(f"UNRESOLVED: {ref} ({n} rows)")
        sys.exit(f"ERROR: {sum(bad.values())} events references do not resolve.")
    print(f"events validation OK: {n_rows} stimulus references across "
          f"{n_files} events files all resolve")


README = """\
# Stimulus registry

Dataset-owned canonical `stimulus_id` tables for the three stimulus sets
(constellation contracts §4.2). One TSV per set; consumers (psytwill, the
catalog stimulus tier, agent tools) validate against these tables and never
invent ids. Generated by `code/mmmdata/scripts/build_stimulus_registry.py` —
edit that script, not these files (`--check` compares, `--validate-events`
asserts every events reference resolves).

| Table | stimulus_id | Rows |
|---|---|---|
| `movies.tsv` | kebab-case normalized title, leading article dropped (`adventure-time`, `table-7`) | {movies} |
| `shared1000.tsv` | image filename stem (`shared0001_nsd02951`) | {shared1000} |
| `twp1000.tsv` | the word itself; voice is a reserved column | {twp1000} |

File-path columns are relative to the set's directory under `stimuli/`.

## events→id rules (verified against all sub-03/04/05 events, 2026-08-17)

- **TB/FIN image trials:** `events.mmmId` → `shared1000.tsv` `mmmId`.
- **TB/FIN word trials:** `(events.word, events.voice)` → `twp1000.tsv` row
  + voice column; `events.itmno` is a redundant join check.
- **NAT trials:** case-insensitive `events.movie_name` →
  `movies.tsv` `movie_name` or `movie_name_variants` (a closed set of every
  spelling observed in events; matching must not assume exact case).

## Known gaps

- `finders-fee` has no annotation file (`movie_annotations/` never had one).
- Three annotation files cover movies never shown in the experiment and are
  deliberately absent from the registry: Fargo, Migration, Paddington.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--check", action="store_true",
                        help="compare regenerated tables to the on-disk TSVs; "
                             "exit 1 on any difference, write nothing")
    parser.add_argument("--validate-events", action="store_true",
                        help="assert every events stimulus reference resolves "
                             "against the regenerated tables")
    args = parser.parse_args()

    tables = {}
    for name, builder in BUILDERS.items():
        print(f"building {name}...")
        tables[name] = render(*builder())
    counts = {name: len(content.splitlines()) - 1 for name, content in tables.items()}
    outputs = dict(tables)
    outputs["README.md"] = README.format(**counts)

    if args.validate_events:
        validate_events(tables)
    if args.check:
        stale = []
        for name, content in outputs.items():
            path = REGISTRY_DIR / (name if name.endswith(".md") else f"{name}.tsv")
            if not path.exists() or path.read_text() != content:
                stale.append(str(path))
        if stale:
            sys.exit(f"ERROR: registry out of date: {stale}. "
                     f"Fix: rerun {Path(__file__).name} without --check.")
        print(f"check OK: {len(outputs)} files match {REGISTRY_DIR}")
        return

    REGISTRY_DIR.mkdir(exist_ok=True)
    for name, content in outputs.items():
        path = REGISTRY_DIR / (name if name.endswith(".md") else f"{name}.tsv")
        path.write_text(content)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
