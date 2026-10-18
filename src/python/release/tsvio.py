"""Read and rewrite BIDS TSVs without disturbing rows that did not change.

BIDS TSVs are plain tab-split text: no quoting convention, no tab or newline
inside a cell. They are parsed exactly that way, so a cell's text is the
bytes between two tabs (a cell that a writer happened to wrap in ``"``
keeps its quotes), and rewriting a file re-joins the same pieces: rows that
were not edited come back byte-for-byte, and the line ending style and the
final newline are kept.
"""

from __future__ import annotations

import re

NA = "n/a"
_NUMERIC = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


class Table:
    """A parsed TSV. ``rows`` are lists of strings aligned to ``header``."""

    def __init__(self, text: str, name: str = "<tsv>"):
        self.name = name
        lines = text.splitlines(keepends=True)
        if not lines or not lines[0].strip():
            raise ValueError(f"{name}: empty TSV or blank header")
        self._lines: list[tuple[str, str]] = []      # (content, line ending)
        for line in lines:
            body = line.rstrip("\r\n")
            self._lines.append((body, line[len(body):]))
        self.header = self._lines[0][0].split("\t")
        self._row_line: list[int] = []                # data row -> line index
        self.rows: list[list[str]] = []
        for k, (body, _) in enumerate(self._lines[1:], start=1):
            if body == "":
                continue                              # blank line, passed through
            cells = body.split("\t")
            if len(cells) != len(self.header):
                raise ValueError(
                    f"{name}: line {k + 1} has {len(cells)} cells, header has "
                    f"{len(self.header)}")
            self._row_line.append(k)
            self.rows.append(cells)
        self._dirty: set[int] = set()

    # access ----------------------------------------------------------------
    def col(self, name: str) -> int:
        try:
            return self.header.index(name)
        except ValueError:
            raise KeyError(f"{self.name}: no column {name!r} "
                           f"(columns: {', '.join(self.header)})") from None

    def get(self, row: int, column: str) -> str:
        return self.rows[row][self.col(column)]

    def set(self, row: int, column: str, value: str) -> None:
        if "\t" in value or "\n" in value or "\r" in value:
            raise ValueError(f"{self.name}: refusing to write a tab or newline into a cell")
        j = self.col(column)
        if self.rows[row][j] != value:
            self.rows[row][j] = value
            self._dirty.add(row)

    @property
    def changed(self) -> bool:
        return bool(self._dirty)

    # serialisation -----------------------------------------------------------
    def to_text(self) -> str:
        lines = list(self._lines)
        for i in self._dirty:
            k = self._row_line[i]
            lines[k] = ("\t".join(self.rows[i]), lines[k][1])
        return "".join(body + end for body, end in lines)


def is_numeric_column(table: Table, column: str) -> bool:
    """True if every non-``n/a`` value in the column is a number."""
    j = table.col(column)
    values = [r[j].strip() for r in table.rows if r[j].strip() not in ("", NA)]
    return all(_NUMERIC.match(v) for v in values)


def is_id_column(column: str) -> bool:
    return column in ("participant_id", "session_id") or column.endswith("_id")


def text_columns(table: Table) -> list[str]:
    """Non-ID columns whose values are not all numeric."""
    return [c for c in table.header
            if not is_id_column(c) and not is_numeric_column(table, c)]


def row_key(table: Table, row: int) -> str:
    """Human-readable key: ID column values if present, else ``#N``."""
    parts = [f"{c}={table.rows[row][j]}" for j, c in enumerate(table.header)
             if is_id_column(c)]
    return "&".join(parts) if parts else f"#{row + 1}"
