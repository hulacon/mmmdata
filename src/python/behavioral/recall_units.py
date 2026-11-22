"""Sentence units of NAT spoken free recall, as text-free pointers into ``beh/``.

A recall unit is a clause-sized span of one recall trial's transcript: the
thing an enc->ret matcher (rMatch, or a human coder) assigns to film events.
The transcript itself stays where it is, in the harmonised word table
``sub-##/ses-##/beh/sub-##_ses-##_task-NATretrieval[_run-##]_beh.tsv`` (one
row per word). A unit is stored only as a row range into that table, so any
redaction of the transcript happens in one place and the unit tables carry no
participant speech.

Segmentation (unchanged from the method card the first rMatch run used):

* filled pauses (``filler == 1``) are dropped first; everything below counts
  lexical words only;
* a sentence ends at a word ending in ``.``, ``!`` or ``?`` (closing quotes or
  brackets allowed after it);
* a sentence longer than ``MAX_WORDS`` is cut at a comma, semicolon or colon
  once the piece has ``PIECE_MIN`` words (and the remainder keeps at least
  ``MIN_UNIT_WORDS``), and any piece still too long is cut into equal parts;
* a unit shorter than ``MIN_UNIT_WORDS`` joins the unit before it (the first
  unit joins the one after it).

Pointer convention: ``first_row`` and ``last_row`` are 0-based data-row
indices (header excluded) into the trial's ``beh.tsv``, both inclusive, and
both always land on lexical words. Filled pauses inside the range are not
part of the unit: :func:`rejoin` drops them, exactly as the cutter did.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

MAX_WORDS = 40
PIECE_MIN = 20
MIN_UNIT_WORDS = 3

_SENTENCE_END = re.compile(r"[.!?][\"”’')]*$")

UNIT_COLUMNS = ["trial_num", "unit_idx", "movie_name", "first_row", "last_row",
                "n_words", "onset", "offset"]


def split_units(words: list[str]) -> list[tuple[int, int]]:
    """Cut a list of lexical words into ``(start, stop)`` half-open spans."""
    sents, s = [], 0
    for i, w in enumerate(words):
        if _SENTENCE_END.search(w):
            sents.append((s, i + 1))
            s = i + 1
    if s < len(words):
        sents.append((s, len(words)))
    out = []
    for a, b in sents:
        pieces = []
        if b - a > MAX_WORDS:  # long run-on: cut at commas once a piece has PIECE_MIN words
            c = a
            for i in range(a, b):
                if (i + 1 - c >= PIECE_MIN and words[i].endswith((",", ";", ":"))
                        and b - (i + 1) >= MIN_UNIT_WORDS):
                    pieces.append((c, i + 1))
                    c = i + 1
            pieces.append((c, b))
        else:
            pieces = [(a, b)]
        for a2, b2 in pieces:  # still too long (no commas): equal cuts
            n = math.ceil((b2 - a2) / MAX_WORDS)
            edges = np.linspace(a2, b2, n + 1).round().astype(int)
            out += list(zip(edges[:-1], edges[1:]))
    merged = []
    for a, b in out:
        if b - a < MIN_UNIT_WORDS and merged:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    if len(merged) > 1 and merged[0][1] - merged[0][0] < MIN_UNIT_WORDS:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)
    return [(int(a), int(b)) for a, b in merged]


def read_beh(path) -> pd.DataFrame:
    """Read a harmonised NATretrieval word table, every cell as written.

    ``keep_default_na=False`` keeps a transcribed word such as "null" or "NA"
    a word, not a missing value.
    """
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def beh_units(beh: pd.DataFrame) -> pd.DataFrame:
    """Units of every recall trial in one word table, as row pointers.

    ``onset`` is the first word's onset and ``offset`` the last word's onset
    plus its duration, both on the run clock; either is NaN where the word
    table has ``n/a`` for that word. A trial with no lexical words has no
    units. Raises if a trial names more than one film.
    """
    rows = []
    trial = beh.trial_num.astype(int)
    for t, g in beh.groupby(trial, sort=True):
        g = g[g.filler.astype(int) == 0]
        if not len(g):
            continue
        movies = g.movie_name.unique()
        if len(movies) != 1:
            raise ValueError(f"trial {t}: more than one movie_name {list(movies)}")
        words = g.word.tolist()
        idx = g.index.to_numpy()
        on = pd.to_numeric(g.onset, errors="coerce").to_numpy()
        du = pd.to_numeric(g.duration, errors="coerce").to_numpy()
        for k, (a, b) in enumerate(split_units(words)):
            rows.append(dict(trial_num=int(t), unit_idx=k, movie_name=movies[0],
                             first_row=int(idx[a]), last_row=int(idx[b - 1]), n_words=b - a,
                             onset=on[a], offset=on[b - 1] + du[b - 1]))
    return pd.DataFrame(rows, columns=UNIT_COLUMNS)


def rejoin(beh: pd.DataFrame, units: pd.DataFrame) -> list[str]:
    """Each unit's text, rebuilt from its row range (filled pauses dropped)."""
    lex = beh.filler.astype(int).to_numpy() == 0
    words = beh.word.to_numpy()
    out = []
    for a, b in zip(units.first_row.astype(int), units.last_row.astype(int)):
        sl = slice(a, b + 1)
        out.append(" ".join(words[sl][lex[sl]]))
    return out


def add_recall_rank(units: pd.DataFrame) -> pd.DataFrame:
    """Order of each trial among one participant's recalls of the same film.

    ``units`` must span every session of each participant (``subject``,
    ``session``, ``trial_num`` and ``stimulus_id`` columns): rank 1 is the
    first time that participant recalled that film, by session then trial.
    """
    key = ["subject", "stimulus_id", "session", "trial_num"]
    order = units[key].drop_duplicates().sort_values(key)
    order["recall_rank"] = order.groupby(["subject", "stimulus_id"]).cumcount() + 1
    return units.merge(order, on=key, how="left", validate="m:1")
