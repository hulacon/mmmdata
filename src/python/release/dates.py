"""Per-participant date shift of ``sub-##/sub-##_sessions.tsv`` ``acq_time``.

For each participant, the earliest session date maps to the epoch and every
other date keeps its offset in days from it, so intervals between sessions
are preserved and absolute dates are not. Values are ``YYYY-MM-DD``; a
``YYYY-MM-DDThh:mm:ss`` value keeps its time of day. ``n/a`` passes through.

The per-participant offsets are secret: :func:`offsets_tsv` renders them for
a private file outside the stage, and nothing here prints them.

Idempotency: a participant whose dates all have year <= 1925 is already
shifted and is skipped with a notice; a participant with a mix of shifted
and unshifted dates is an error.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field

from .stage import SHIFTED_MAX_YEAR, Stage
from .tsvio import NA, Table

COLUMN = "acq_time"
_VALUE = re.compile(r"^(\d{4}-\d{2}-\d{2})(T\S+)?$")


class DateShiftError(ValueError):
    pass


@dataclass
class ParticipantShift:
    participant: str
    file: str                      # stage-relative path
    status: str                    # "shifted" | "already-shifted" | "no-dates"
    n_dates: int = 0
    offset_days: int | None = field(default=None, repr=False)   # secret


@dataclass
class DateShiftPlan:
    epoch: _dt.date
    participants: list[ParticipantShift]

    @property
    def shifted(self) -> list[ParticipantShift]:
        return [p for p in self.participants if p.status == "shifted"]

    def summary_lines(self) -> list[str]:
        """Public-safe summary: no offsets, no dates."""
        lines = []
        for p in self.participants:
            if p.status == "shifted":
                lines.append(f"  {p.participant}: {p.n_dates} date(s) shifted to the epoch")
            elif p.status == "already-shifted":
                lines.append(f"  {p.participant}: ALREADY SHIFTED (every year <= "
                             f"{SHIFTED_MAX_YEAR}); left as is")
            else:
                lines.append(f"  {p.participant}: no {COLUMN} dates; left as is")
        return lines


def _parse(value: str, where: str) -> tuple[_dt.date, str] | None:
    v = value.strip()
    if v in ("", NA):
        return None
    m = _VALUE.match(v)
    if not m:
        raise DateShiftError(f"{where}: {COLUMN} value is not YYYY-MM-DD[Thh:mm:ss]")
    try:
        return _dt.date.fromisoformat(m.group(1)), m.group(2) or ""
    except ValueError:
        raise DateShiftError(f"{where}: {COLUMN} value is not a valid date") from None


def sessions_files(stage: Stage) -> list[str]:
    files = []
    for p in sorted(stage.root.glob("sub-*/sub-*_sessions.tsv")):
        label = p.name[: -len("_sessions.tsv")]
        if p.parent.name != label:
            raise DateShiftError(f"{stage.rel(p)}: file name does not match its "
                                 f"participant directory")
        files.append(stage.rel(p))
    return files


def plan_date_shift(stage: Stage, epoch: _dt.date) -> DateShiftPlan:
    """Validate every sessions.tsv, compute offsets, queue the rewrites."""
    plans: list[ParticipantShift] = []
    queued: list[tuple[str, Table]] = []
    for rel in sessions_files(stage):
        participant = rel.split("/", 1)[0]
        table = Table(stage.read_text(rel), name=rel)
        if COLUMN not in table.header:
            plans.append(ParticipantShift(participant, rel, "no-dates"))
            continue
        parsed = {}
        for i in range(len(table.rows)):
            got = _parse(table.get(i, COLUMN), f"{rel} data row {i + 1}")
            if got is not None:
                parsed[i] = got
        if not parsed:
            plans.append(ParticipantShift(participant, rel, "no-dates"))
            continue
        old = [d.year <= SHIFTED_MAX_YEAR for d, _ in parsed.values()]
        if all(old):
            plans.append(ParticipantShift(participant, rel, "already-shifted",
                                          n_dates=len(parsed)))
            continue
        if any(old):
            raise DateShiftError(
                f"{rel}: some {COLUMN} values look already shifted (year <= "
                f"{SHIFTED_MAX_YEAR}) and some do not; refusing to guess. "
                f"Re-stage this participant from the working tree.")
        first = min(d for d, _ in parsed.values())
        offset = (first - epoch).days
        for i, (d, tail) in parsed.items():
            new = d - _dt.timedelta(days=offset)
            if new.year > SHIFTED_MAX_YEAR:
                raise DateShiftError(
                    f"{rel}: a shifted date would fall after {SHIFTED_MAX_YEAR} "
                    f"(span too long for epoch {epoch}); choose an earlier epoch")
            table.set(i, COLUMN, new.isoformat() + tail)
        plans.append(ParticipantShift(participant, rel, "shifted",
                                      n_dates=len(parsed), offset_days=offset))
        queued.append((rel, table))
    # queue only after every participant validated
    for rel, table in queued:
        stage.queue_write(rel, table.to_text())
    return DateShiftPlan(epoch=epoch, participants=plans)


def offsets_tsv(plan: DateShiftPlan) -> str:
    """The secret mapping: original date = shifted date + offset_days."""
    lines = ["participant_id\tepoch\toffset_days"]
    for p in plan.shifted:
        lines.append(f"{p.participant}\t{plan.epoch.isoformat()}\t{p.offset_days}")
    return "\n".join(lines) + "\n"
