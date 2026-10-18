"""Scan a staged copy for identifiers that should not ship.

Kinds of finding
----------------
``date``          an ISO date or date-time (``YYYY-MM-DD``,
                  ``YYYY-MM-DDThh:mm[:ss]``, ``YYYYMMDDThhmmss``) outside the
                  shifted range [epoch, 1925-12-31]
``path``          an absolute path under ``/gpfs/``, ``/projects/`` or ``/home/``
``email``         an email address
``name``          a user name or person name from a private names file
                  (one per line, ``#`` comments; matched case-insensitively
                  as whole words). The file is supplied at run time and must
                  live outside the stage; it is never committed.
``dataset_link``  a ``dataset_description.json`` (root or derivative) whose
                  ``DatasetLinks`` value or ``SourceDatasets[].URL`` is an
                  absolute filesystem path (``/...`` or ``file:``)
``json_error``    a ``dataset_description.json`` that does not parse

Scanned: files with a text extension (``.tsv .json .txt .md .csv .html
.rst .log .yml .yaml .toml .cfg .bib .tex .bval .bvec``, configurable) and
extension-less ``README``/``CHANGES``/``LICENSE``. The ``.git`` and
``.datalad`` directories are skipped.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .stage import SHIFTED_MAX_YEAR, Stage

TEXT_EXTENSIONS = (".tsv", ".json", ".txt", ".md", ".csv", ".html", ".rst",
                   ".log", ".yml", ".yaml", ".toml", ".cfg", ".bib", ".tex",
                   ".bval", ".bvec")
BARE_TEXT_NAMES = ("README", "CHANGES", "LICENSE")
SKIP_DIRS = {".git", ".datalad"}

_DATE = re.compile(
    r"(?<![\d-])(\d{4})-(\d{2})-(\d{2})"
    r"(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?)?(?![\d])")
_DATE_BASIC = re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})(?!\d)")
_PATH = re.compile(r"(?<![\w.~-])/(?:gpfs|projects|home)/[^\s\"'<>,;)\]}]*")
_EMAIL = re.compile(r"(?<![\w.%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")


@dataclass(frozen=True)
class Finding:
    path: str        # stage-relative
    line: int        # 1-based; 0 if not locatable
    kind: str
    match: str

    def format(self) -> str:
        return f"{self.path}:{self.line}\t{self.kind}\t{self.match}"


def load_names(path: str | Path) -> list[str]:
    names = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            names.append(s)
    if not names:
        raise ValueError(f"names file {path} lists no names; refusing to sweep "
                         f"with an empty list (wrong file?)")
    return names


def _names_regex(names: list[str]) -> re.Pattern | None:
    if not names:
        return None
    alts = []
    for n in sorted(set(names), key=len, reverse=True):
        body = r"\s+".join(re.escape(part) for part in n.split())
        alts.append(r"(?<!\w)" + body + r"(?!\w)")
    return re.compile("|".join(alts), re.IGNORECASE)


def _date_ok(y: str, m: str, d: str, epoch: _dt.date) -> bool | None:
    """True if inside the shifted range, False if outside, None if not a
    valid calendar date (not a date at all; ignored)."""
    try:
        day = _dt.date(int(y), int(m), int(d))
    except ValueError:
        return None
    return epoch <= day <= _dt.date(SHIFTED_MAX_YEAR, 12, 31)


def _is_text_file(name: str, exts) -> bool:
    return name.endswith(tuple(exts)) or name in BARE_TEXT_NAMES


def shipped_files(stage: Stage, exts=TEXT_EXTENSIONS) -> list[Path]:
    files = set()
    for dirpath, dirnames, filenames in os.walk(stage.root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for f in filenames:
            if _is_text_file(f, exts):
                files.add(Path(dirpath) / f)
    for p in stage.pending:                 # files a planned step would create
        if _is_text_file(p.name, exts):
            files.add(p)
    return sorted(files)


def _abs_fs_path(value) -> bool:
    return isinstance(value, str) and (value.startswith("/") or
                                       value.lower().startswith("file:"))


def _line_of(lines: list[str], value: str) -> int:
    needle = json.dumps(value)[1:-1]
    for n, line in enumerate(lines, start=1):
        if needle in line or value in line:
            return n
    return 0


def check_dataset_description(rel: str, text: str) -> list[Finding]:
    lines = text.splitlines()
    try:
        meta = json.loads(text)
    except json.JSONDecodeError as exc:
        return [Finding(rel, exc.lineno, "json_error", str(exc))]
    out = []
    links = meta.get("DatasetLinks") or {}
    if isinstance(links, dict):
        for key, value in links.items():
            if _abs_fs_path(value):
                out.append(Finding(rel, _line_of(lines, value), "dataset_link",
                                   f"DatasetLinks.{key}={value}"))
    sources = meta.get("SourceDatasets") or []
    if isinstance(sources, list):
        for k, src in enumerate(sources):
            url = src.get("URL") if isinstance(src, dict) else None
            if _abs_fs_path(url):
                out.append(Finding(rel, _line_of(lines, url), "dataset_link",
                                   f"SourceDatasets[{k}].URL={url}"))
    return out


def scan_text(rel: str, text: str, epoch: _dt.date,
              names_re: re.Pattern | None) -> list[Finding]:
    out = []
    for n, line in enumerate(text.splitlines(), start=1):
        if "-" in line:
            for m in _DATE.finditer(line):
                if _date_ok(m.group(1), m.group(2), m.group(3), epoch) is False:
                    out.append(Finding(rel, n, "date", m.group()))
        if "T" in line:
            for m in _DATE_BASIC.finditer(line):
                if _date_ok(m.group(1), m.group(2), m.group(3), epoch) is False:
                    out.append(Finding(rel, n, "date", m.group()))
        if "/" in line:
            out.extend(Finding(rel, n, "path", m.group()) for m in _PATH.finditer(line))
        if "@" in line:
            out.extend(Finding(rel, n, "email", m.group()) for m in _EMAIL.finditer(line))
        if names_re is not None:
            out.extend(Finding(rel, n, "name", m.group()) for m in names_re.finditer(line))
    return out


def sweep(stage: Stage, epoch: _dt.date, names: list[str] | None = None,
          exts=TEXT_EXTENSIONS) -> list[Finding]:
    """Every finding in every shipped text file (reads see queued writes)."""
    names_re = _names_regex(names or [])
    findings: list[Finding] = []
    for p in shipped_files(stage, exts):
        rel = stage.rel(p)
        text = stage.read_text(p)
        found = scan_text(rel, text, epoch, names_re)
        if p.name == "dataset_description.json":
            structural = check_dataset_description(rel, text)
            covered = {f.line for f in structural}
            found = [f for f in found if not (f.kind == "path" and f.line in covered)]
            found = structural + found
        findings.extend(found)
    return findings


def summary(findings: list[Finding]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.kind] = counts.get(f.kind, 0) + 1
    return dict(sorted(counts.items()))
