"""Film event timing for the encoding harness, with no braintwill import.

The word onsets and shot cuts that the harness turns into ``word-on`` and ``cuts`` bands
(``harness.film_word_on``, ``harness.film_cuts``), kept here so a script in an env without
braintwill (``control_draws.py``, the stimfeat env) reads them the same way.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

#: Film word tokens: word2psy's tokens of the films' transcript, the rows of every text model's
#: films store table (keyed by ``stimulus_id`` and the corpus-wide ``word_idx``; the five
#: models' token lists are identical). They carry no timing; see ``film_word_onsets``.
FILM_TOKENS = Path("movies") / "transcript_fasttext_words.csv"
#: A film's ASR words with their onsets, per film under ``movies/``.
FILM_ASR_WORDS = "transcribe_transcript_words.csv"
#: The films' shot cuts: space-timescale's cut schema v1, under ``fit_corpora_dir``. A cut bin
#: is a classifier-accepted cut, the schema's high-precision rule, not its high-recall masking
#: rule (the union): as a regressor a false cut is noise in the column (model-readout).
CUT_TABLE = Path("space-timescale") / "cuts" / "cut_schema_v1.parquet"
CUT_CORPUS = "mmmdata"
CUT_RULE = "classifier_accept"
CUT_SPACE = "cuts"

def _chars(word) -> str:
    return re.sub(r"[^a-z0-9]", "", str(word).lower())


def film_word_onsets(stimfeat: Path) -> pd.DataFrame:
    """Every film token's ``onset`` and ``offset`` (keyed ``stimulus_id``, ``word_idx``).

    The tokens are word2psy's re-tokenisation of the ASR transcript (``i'm`` → ``i`` ``'m``)
    and carry no timing, so each takes the timing of the ASR word holding its first character.
    Per film, the tokens' and the ASR words' character streams (lowercase letters and digits)
    must be identical: the alignment is exact or it is an error.
    """
    movies = Path(stimfeat) / FILM_TOKENS.parent
    tokens = pd.read_csv(Path(stimfeat) / FILM_TOKENS, usecols=["stimulus_id", "word_idx", "word"])
    out = []
    for sid, g in tokens.groupby("stimulus_id", sort=True):
        src = movies / sid / FILM_ASR_WORDS
        asr = pd.read_csv(src).sort_values(["chunk_idx", "word_idx"])
        if (np.diff(asr["onset"].to_numpy(float)) < 0).any():
            raise RuntimeError(f"{src}: word onsets go backwards in chunk/word order")
        g = g.sort_values("word_idx")
        t = [_chars(w) for w in g["word"]]
        if not all(t):
            raise RuntimeError(f"{sid}: a transcript token has no letters or digits")
        a = [_chars(w) for w in asr["word"]]
        if "".join(t) != "".join(a):
            raise RuntimeError(f"{sid}: transcript tokens and the ASR words of {src} spell different text")
        owner = np.repeat(np.arange(len(a)), [len(s) for s in a])[np.cumsum([0] + [len(s) for s in t[:-1]])]
        out.append(pd.DataFrame({"stimulus_id": sid, "word_idx": g["word_idx"].to_numpy(),
                                 "onset": asr["onset"].to_numpy(float)[owner],
                                 "offset": asr["offset"].to_numpy(float)[owner]}))
    return pd.concat(out, ignore_index=True)


def load_cuts(fit_corpora: Path) -> pd.DataFrame:
    """The films' accepted shot cuts (``CUT_TABLE``, ``CUT_RULE``): ``stimulus_id``, ``bin``."""
    src = Path(fit_corpora) / CUT_TABLE
    if not src.exists():
        raise FileNotFoundError(f"cut table {src} is missing (space-timescale cut schema v1; "
                                f"fit_corpora_dir in mmmdata config)")
    d = pd.read_parquet(src, columns=["corpus", "id", "bin", CUT_RULE])
    d = d[(d["corpus"] == CUT_CORPUS) & d[CUT_RULE].astype(bool)]
    if d.empty:
        raise RuntimeError(f"{src}: no {CUT_CORPUS!r} rows with {CUT_RULE}")
    return d.rename(columns={"id": "stimulus_id"})[["stimulus_id", "bin"]].reset_index(drop=True)


def event_bins(seconds: np.ndarray, n_bins: int, frame_s: float, name: str = "") -> np.ndarray:
    """A film's ``n_bins`` grid bins holding at least one event at ``seconds`` (bool)."""
    b = np.floor(np.asarray(seconds, float) / frame_s + 1e-9).astype(int)
    if b.size and (b.min() < 0 or b.max() >= n_bins):
        raise RuntimeError(f"{name}: events outside the film's {n_bins} bins")
    on = np.zeros(n_bins, bool)
    on[b] = True
    return on
