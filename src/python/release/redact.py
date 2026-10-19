"""Apply a human-reviewed redaction spec, and flag candidate spans.

The tool does not decide what is identifying: a person reads every
transcript and writes the spec. ``flag`` only lists candidates to help that
reading, and its output says so.

Spec format (TSV, UTF-8, plain tab-split with no quoting; lines starting
with ``#`` and blank lines are ignored; an exact duplicate entry is an error)
---------------------------------------------------------------------------
Required header columns, any extra columns (e.g. ``reason``, ``note``) are
ignored::

    file      path of the TSV, relative to the stage root
              (e.g. sub-##/ses-##/beh/sub-##_ses-##_task-<task>_beh.tsv)
    row       which data row: "#N" (1-based, header excluded), or one or more
              "column=value" conditions joined by "&" that must select
              exactly one row (e.g. "onset=12.345", "participant_id=sub-##")
    column    the column holding the cell
    span      the exact text to replace (case-sensitive, matched as whole
              words where it starts/ends with a word character; every
              occurrence in the cell is replaced), or "*" for the whole cell

Each match becomes ``[REDACTED]``. The sidecar beside the TSV (``.tsv`` ->
``.json``) gets a ``Redaction`` key stating the convention; the sidecar is
created if absent.

Every entry must match: an unknown file or column, a selector that does not
pick exactly one row, or a span that is not in the cell is an error, and no
file is written unless every entry applies. Re-applying a spec to a stage
that already holds its redactions therefore fails loudly; re-stage instead.

``flag`` writes the same columns (plus ``reason`` and ``context``), so a
reviewed flag list can be trimmed into a spec.
"""

from __future__ import annotations

import bisect
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .stage import Stage
from .tsvio import Table, is_id_column, text_columns

TOKEN = "[REDACTED]"
WHOLE_CELL = "*"
DEFAULT_NOTE = (
    "Transcripts were read in full by a human reviewer before release. Names of "
    "people, names of places, and self-references that could identify the "
    "participant were removed; each removed span is replaced in the TSV cell by "
    "the token [REDACTED]."
)
SPEC_COLUMNS = ("file", "row", "column", "span")


class RedactionError(ValueError):
    pass


@dataclass(frozen=True)
class SpecEntry:
    line: int
    file: str
    row: str
    column: str
    span: str

    def where(self) -> str:
        return f"spec line {self.line} ({self.file} | {self.row} | {self.column})"


@dataclass
class RedactionResult:
    files: dict[str, int]           # stage-relative TSV -> number of replacements
    sidecars: list[str]

    @property
    def n_replacements(self) -> int:
        return sum(self.files.values())


# --------------------------------------------------------------------------- #
# spec                                                                        #
# --------------------------------------------------------------------------- #
def load_spec(path: str | Path) -> list[SpecEntry]:
    text = Path(path).read_text(encoding="utf-8")
    numbered = [(n, line.rstrip("\r\n")) for n, line in enumerate(text.splitlines(), start=1)
                if line.strip() and not line.lstrip().startswith("#")]
    if not numbered:
        raise RedactionError(f"{path}: redaction spec is empty")
    header = [h.strip() for h in numbered[0][1].split("\t")]
    missing = [c for c in SPEC_COLUMNS if c not in header]
    if missing:
        raise RedactionError(f"{path}: spec header lacks column(s) {missing}; "
                             f"need {list(SPEC_COLUMNS)}")
    idx = {c: header.index(c) for c in SPEC_COLUMNS}
    entries, seen = [], {}
    for n, line in numbered[1:]:
        row = line.split("\t")
        row += [""] * (len(header) - len(row))
        e = SpecEntry(n, row[idx["file"]].strip(), row[idx["row"]].strip(),
                      row[idx["column"]].strip(), row[idx["span"]])
        if not (e.file and e.row and e.column and e.span):
            raise RedactionError(f"{path}: spec line {n} has an empty file/row/"
                                 f"column/span (use '*' for a whole cell)")
        key = (e.file, e.row, e.column, e.span)
        if key in seen:
            raise RedactionError(f"{path}: spec line {n} duplicates line {seen[key]}")
        seen[key] = n
        entries.append(e)
    if not entries:
        raise RedactionError(f"{path}: redaction spec has a header but no entries")
    return entries


def _select_row(table: Table, selector: str, entry: SpecEntry) -> int:
    if selector.startswith("#"):
        try:
            n = int(selector[1:])
        except ValueError:
            raise RedactionError(f"{entry.where()}: bad row selector {selector!r}") from None
        if not 1 <= n <= len(table.rows):
            raise RedactionError(f"{entry.where()}: row {selector} out of range "
                                 f"(table has {len(table.rows)} data rows)")
        return n - 1
    conds = []
    for part in selector.split("&"):
        if "=" not in part:
            raise RedactionError(f"{entry.where()}: bad row selector {selector!r}; "
                                 f"use '#N' or 'column=value'")
        col, val = part.split("=", 1)
        try:
            conds.append((table.col(col.strip()), val.strip()))
        except KeyError as exc:
            raise RedactionError(f"{entry.where()}: {exc.args[0]}") from None
    hits = [i for i, r in enumerate(table.rows) if all(r[j] == v for j, v in conds)]
    if len(hits) != 1:
        raise RedactionError(f"{entry.where()}: row selector matches {len(hits)} "
                             f"rows; it must match exactly one")
    return hits[0]


def _span_pattern(span: str) -> re.Pattern:
    pre = r"(?<!\w)" if re.match(r"\w", span) else ""
    post = r"(?!\w)" if re.search(r"\w$", span) else ""
    return re.compile(pre + re.escape(span) + post)


def sidecar_for(rel: str) -> str:
    if not rel.endswith(".tsv"):
        raise RedactionError(f"{rel}: only plain .tsv files can be redacted")
    return rel[: -len(".tsv")] + ".json"


def plan_redactions(stage: Stage, entries: list[SpecEntry],
                    note: str = DEFAULT_NOTE) -> RedactionResult:
    """Validate and apply every entry in memory; queue the rewrites only if
    all entries match."""
    by_file: dict[str, list[SpecEntry]] = defaultdict(list)
    for e in entries:
        by_file[e.file].append(e)
    tables: dict[str, Table] = {}
    counts: dict[str, int] = {}
    for rel, group in by_file.items():
        if not stage.exists(rel):
            raise RedactionError(f"{group[0].where()}: file not found in the stage")
        table = Table(stage.read_text(rel), name=rel)
        n = 0
        for e in group:
            i = _select_row(table, e.row, e)
            try:
                cell = table.get(i, e.column)
            except KeyError as exc:
                raise RedactionError(f"{e.where()}: {exc.args[0]}") from None
            if is_id_column(e.column):
                raise RedactionError(f"{e.where()}: refusing to redact ID column")
            if e.span == WHOLE_CELL:
                if cell == TOKEN:
                    raise RedactionError(f"{e.where()}: cell is already {TOKEN}; "
                                         f"is this stage already redacted?")
                table.set(i, e.column, TOKEN)
                n += 1
                continue
            new, k = _span_pattern(e.span).subn(TOKEN, cell)
            if k == 0:
                hint = (f" (cell already contains {TOKEN}; is this stage already "
                        f"redacted?)" if TOKEN in cell else "")
                raise RedactionError(f"{e.where()}: span not found in the cell{hint}")
            table.set(i, e.column, new)
            n += k
        tables[rel] = table
        counts[rel] = n
    sidecars = []
    for rel, table in tables.items():
        stage.queue_write(rel, table.to_text())
        side = sidecar_for(rel)
        meta = {}
        if stage.exists(side):
            try:
                meta = json.loads(stage.read_text(side))
            except json.JSONDecodeError as exc:
                raise RedactionError(f"{side}: sidecar is not valid JSON: {exc}") from None
        if meta.get("Redaction") != note:
            meta["Redaction"] = note
            stage.queue_write(side, json.dumps(meta, indent=4, ensure_ascii=False) + "\n")
            sidecars.append(side)
    return RedactionResult(files=counts, sidecars=sidecars)


# --------------------------------------------------------------------------- #
# flag                                                                        #
# --------------------------------------------------------------------------- #
ADVISORY = ("ADVISORY ONLY: candidate spans for the human reader. A flag is not "
            "a decision, and an unflagged transcript is not known to be free of "
            "identifying text; every transcript still needs a full read.")
TRANSCRIPT_COLUMNS = ("word", "text", "transcript", "utterance")
DEFAULT_FLAG_GLOBS = ("sub-*/**/*_beh.tsv",)

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’.@_-]*")
_ALLOWED_CAPS = {"I", "I'm", "I've", "I'd", "I'll", "I’m", "I’ve",
                 "I’d", "I’ll", "OK", "TV"}
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_LONG_NUMBER = re.compile(r"\d{3,}")
# Phrases after which the next word is a likely name or place.
_CUE = re.compile(
    r"\b(?:my name is|my name's|name is|named|called|i live in|i live on|"
    r"i'm from|i am from|i grew up in|grew up in|born in|my hometown|"
    r"my (?:wife|husband|partner|boyfriend|girlfriend|friend|roommate|boss|"
    r"mom|mother|dad|father|brother|sister|son|daughter|cousin|aunt|uncle|"
    r"grandma|grandmother|grandpa|grandfather|dog|cat|teacher|coworker))\b",
    re.IGNORECASE)


@dataclass
class Flag:
    file: str
    row: str
    column: str
    span: str
    reason: str
    context: str


def _flag_stream(rel: str, column: str, cells: list[tuple[int, str]],
                 stream: bool) -> list[Flag]:
    """Flag tokens in a column. ``stream`` treats consecutive cells as one
    running text (word-per-row transcripts); otherwise each cell stands
    alone and its first token starts a sentence."""
    flags: list[Flag] = []
    groups = [cells] if stream else [[c] for c in cells]
    for group in groups:
        toks: list[tuple[int, str]] = []          # (row index, token)
        for i, text in group:
            if text.strip() in ("", "n/a", TOKEN):
                continue
            toks.extend((i, m.group()) for m in _TOKEN_RE.finditer(text))
        if not toks:
            continue
        words = [t for _, t in toks]

        def ctx(k: int) -> str:
            return " ".join(words[max(0, k - 5):k + 6])

        seen = set()

        def add(k, reason):
            key = (k, reason)
            if key not in seen:
                seen.add(key)
                flags.append(Flag(rel, f"#{toks[k][0] + 1}", column,
                                  words[k].rstrip(".,"), reason, ctx(k)))

        for k, w in enumerate(words):
            sentence_start = k == 0 or words[k - 1].endswith((".", "!", "?"))
            bare = w.rstrip(".,!?")
            if _EMAIL.fullmatch(bare):
                add(k, "email")
            elif _LONG_NUMBER.search(bare):
                add(k, "number (3+ digits)")
            elif bare[:1].isupper() and not sentence_start and bare not in _ALLOWED_CAPS:
                add(k, "capitalized mid-sentence")
        # cue phrases: flag the word that follows
        joined, starts, pos = [], [], 0
        for w in words:
            starts.append(pos)
            joined.append(w)
            pos += len(w) + 1
        text = " ".join(joined)
        for m in _CUE.finditer(text):
            k = bisect.bisect_right(starts, m.end()) if m.end() < len(text) else None
            if k is not None and k < len(words):
                add(k, f"follows '{m.group().lower()}'")
            else:
                add(bisect.bisect_right(starts, m.start()) - 1,
                    f"self-reference '{m.group().lower()}'")
    return flags


def flag_candidates(stage: Stage, globs=DEFAULT_FLAG_GLOBS,
                    columns: list[str] | None = None) -> list[Flag]:
    files = sorted({stage.rel(p) for g in globs for p in stage.root.glob(g)
                    if p.is_file()})
    flags: list[Flag] = []
    for rel in files:
        table = Table(stage.read_text(rel), name=rel)
        if columns:
            cols = [c for c in columns if c in table.header]
        else:
            cols = [c for c in TRANSCRIPT_COLUMNS if c in table.header] or text_columns(table)
        for c in cols:
            j = table.col(c)
            cells = [(i, r[j]) for i, r in enumerate(table.rows)]
            ntok = sorted(len(_TOKEN_RE.findall(t)) for _, t in cells)
            stream = bool(ntok) and ntok[len(ntok) // 2] <= 1
            flags.extend(_flag_stream(rel, c, cells, stream))
    # one row per cell+span; reasons merged
    merged: dict[tuple, Flag] = {}
    for f in flags:
        key = (f.file, f.row, f.column, f.span)
        if key in merged:
            if f.reason not in merged[key].reason:
                merged[key].reason += "; " + f.reason
        else:
            merged[key] = f
    return list(merged.values())


def flags_tsv(flags: list[Flag]) -> str:
    lines = [f"# {ADVISORY}", "file\trow\tcolumn\tspan\treason\tcontext"]
    for f in flags:
        lines.append("\t".join([f.file, f.row, f.column, f.span, f.reason, f.context]))
    return "\n".join(lines) + "\n"
