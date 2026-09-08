"""Events adapters: MMMData-specific relabelling a BIDS Stats Model can name.

BIDS Stats Models describe a design over columns the events file already
has. Some contrasts need a column that exists only in relation to *other*
runs — TBencoding's "first presentation vs later presentation" needs every
run of a session to know which presentation a trial is, and every session
to know which items are the anchors shown in all of them. An adapter is the
named, tested function that derives such a column; the spec declares it as
the first Run-node instruction ``{"Name": "Adapter", "Input": ["<name>"]}``
and the runner applies it before ``Factor``/``Convolve``.

Two-phase by necessity: ``prepare`` sees every run's events, ``apply`` then
relabels one run. Adapters are dataset-specific, which is exactly why they
live here and not in braintwill (Contract C keeps braintwill dataset-agnostic).
"""

from __future__ import annotations

from typing import Protocol, Sequence

import pandas as pd


class EventsAdapter(Protocol):
    name: str

    def prepare(self, events_by_run: Sequence[pd.DataFrame]) -> None: ...

    def apply(self, events: pd.DataFrame) -> pd.DataFrame: ...


class AdapterError(ValueError):
    """The adapter cannot derive its column from the events it was given."""


class TbRepetitionAdapter:
    """TBencoding trials as ``first`` / ``later`` / ``once`` / ``anchor``.

    Facts this encodes (verified on all three subjects' events, glm-strategy
    log 2026-09-08): each trial is an image row and a word row with the same
    onset (one 3 s event here); ``enCon`` 1 items appear once ever
    (``once``); ``enCon`` 2 and 3 items appear three times, always within one
    session, across one to three of its runs; six items recur in every
    session and are excluded from the contrast as ``anchor``. ``first`` is
    the earliest presentation of a non-anchor repeated item within its
    session, ordered by (run, onset); every other presentation is ``later``.

    Anchors are derived, not listed: an item present in every prepared
    session. Preparing fewer than two sessions is refused because the rule
    degenerates to "everything is an anchor".
    """

    name = "tbrepetition"
    required = ("onset", "duration", "modality", "ses_num", "run_idx", "mmmId", "enCon")
    levels = ("first", "later", "once", "anchor")

    def __init__(self, min_sessions: int = 2):
        self.min_sessions = min_sessions
        self.anchors: frozenset = frozenset()
        self._presentation: dict = {}

    def _trials(self, events: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.required if c not in events.columns]
        if missing:
            raise AdapterError(f"{self.name}: events lack columns {missing}; have {list(events.columns)}")
        trials = events[events["modality"].astype(str) == "visual"].copy()
        if trials.empty:
            raise AdapterError(f"{self.name}: no visual (image) rows; are these TBencoding events?")
        return trials

    def prepare(self, events_by_run: Sequence[pd.DataFrame]) -> None:
        trials = pd.concat([self._trials(e) for e in events_by_run], ignore_index=True)
        n_sessions = trials["ses_num"].nunique()
        if n_sessions < self.min_sessions:
            raise AdapterError(
                f"{self.name}: prepared {n_sessions} session(s); anchors are the items present in "
                f"every session, which needs at least {self.min_sessions}. Prepare all of the "
                "subject's TBencoding runs, not a session subset."
            )
        per_item = trials.groupby("mmmId")["ses_num"].nunique()
        self.anchors = frozenset(per_item[per_item == n_sessions].index)
        trials = trials.sort_values(["ses_num", "run_idx", "onset"])
        trials["k"] = trials.groupby(["ses_num", "mmmId"]).cumcount()
        self._presentation = {
            (r.ses_num, r.run_idx, float(r.onset)): int(r.k) for r in trials.itertuples(index=False)
        }

    def apply(self, events: pd.DataFrame) -> pd.DataFrame:
        if not self._presentation:
            raise AdapterError(f"{self.name}: call prepare() with every run's events first")
        trials = self._trials(events).copy()
        labels = []
        for r in trials.itertuples(index=False):
            key = (r.ses_num, r.run_idx, float(r.onset))
            if key not in self._presentation:
                raise AdapterError(f"{self.name}: trial {key} was not among the prepared runs")
            if r.mmmId in self.anchors:
                labels.append("anchor")
            elif int(r.enCon) == 1:
                labels.append("once")
            else:
                labels.append("first" if self._presentation[key] == 0 else "later")
        trials["trial_type"] = labels
        return trials.reset_index(drop=True)


ADAPTERS: dict[str, type] = {TbRepetitionAdapter.name: TbRepetitionAdapter}


def get_adapter(name: str) -> EventsAdapter:
    try:
        return ADAPTERS[name]()
    except KeyError:
        raise KeyError(f"Unknown events adapter {name!r}; available: {sorted(ADAPTERS)}") from None


def adapt_events(adapter_name: str | None, events_by_run: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
    """Apply a model's adapter (if any) to every run's events, in order."""
    if adapter_name is None:
        return list(events_by_run)
    adapter = get_adapter(adapter_name)
    adapter.prepare(events_by_run)
    return [adapter.apply(e) for e in events_by_run]
