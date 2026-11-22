"""Tests for behavioral.recall_units: cutting NAT recall into units, as row pointers.

Every fixture is synthetic; nothing reads the dataset.
"""

import numpy as np
import pandas as pd
import pytest

from behavioral import recall_units as ru


def _words(n, end=""):
    return [f"w{i}" for i in range(n - 1)] + [f"w{n - 1}{end}"]


class TestSplitUnits:
    def test_sentence_ends_cut(self):
        words = ["a", "b", "c.", "d", "e", "f!", "g", "h", "i?"]
        assert ru.split_units(words) == [(0, 3), (3, 6), (6, 9)]

    def test_closing_quote_after_stop_still_ends(self):
        words = ["he", "said", '"go."', "then", "we", "left."]
        assert ru.split_units(words) == [(0, 3), (3, 6)]

    def test_trailing_words_without_stop_form_a_unit(self):
        assert ru.split_units(["a", "b", "c.", "d", "e", "f"]) == [(0, 3), (3, 6)]

    def test_short_unit_joins_previous(self):
        assert ru.split_units(["a", "b", "c.", "d."]) == [(0, 4)]

    def test_short_first_unit_joins_next(self):
        assert ru.split_units(["Okay.", "a", "b", "c."]) == [(0, 4)]

    def test_long_sentence_cut_at_comma_after_piece_min(self):
        words = _words(30) + ["x,"] + _words(20, ".")
        spans = ru.split_units(words)
        assert spans[0] == (0, 31)  # cut at the comma, piece >= PIECE_MIN words
        assert spans[-1][1] == len(words)
        assert all(b - a <= ru.MAX_WORDS for a, b in spans)

    def test_long_sentence_without_commas_cut_equally(self):
        words = _words(100, ".")
        spans = ru.split_units(words)
        assert [b - a for a, b in spans] == [33, 34, 33]

    def test_spans_tile_the_words(self):
        rng = np.random.default_rng(0)
        words = [f"w{i}" + ("." if rng.random() < .1 else "," if rng.random() < .1 else "") for i in range(500)]
        spans = ru.split_units(words)
        assert spans[0][0] == 0 and spans[-1][1] == len(words)
        assert all(b == a2 for (_, b), (a2, _) in zip(spans, spans[1:]))


def _beh(trials):
    """trials: list of (movie, [(word, filler), ...]); onsets 1 s apart, 0.5 s long."""
    rows, t = [], 10.0
    for k, (movie, words) in enumerate(trials, 1):
        for w, f in words:
            rows.append(dict(onset=f"{t:.3f}", duration="0.500", word=w, trial_num=str(k), movie_name=movie,
                             filler=str(f)))
            t += 1.0
    return pd.DataFrame(rows)


class TestBehUnits:
    def test_pointers_skip_fillers_and_rejoin(self):
        beh = _beh([("Film A", [("um", 1), ("The", 0), ("dog", 0), ("ran.", 0), ("uh", 1),
                                ("It", 0), ("uh", 1), ("was", 0), ("fast.", 0)])])
        u = ru.beh_units(beh)
        assert u[["first_row", "last_row", "n_words"]].values.tolist() == [[1, 3, 3], [5, 8, 3]]
        assert ru.rejoin(beh, u) == ["The dog ran.", "It was fast."]
        assert u.onset.tolist() == [11.0, 15.0]
        assert u.offset.tolist() == [13.5, 18.5]

    def test_units_restart_per_trial(self):
        beh = _beh([("Film A", [("a", 0), ("b", 0), ("c.", 0)]), ("Film B", [("d", 0), ("e", 0), ("f.", 0)])])
        u = ru.beh_units(beh)
        assert u[["trial_num", "unit_idx", "movie_name", "first_row"]].values.tolist() == [
            [1, 0, "Film A", 0], [2, 0, "Film B", 3]]

    def test_words_like_na_survive_reading(self, tmp_path):
        beh = _beh([("Film A", [("NA", 0), ("null", 0), ("n/a.", 0)])])
        p = tmp_path / "beh.tsv"
        beh.to_csv(p, sep="\t", index=False)
        back = ru.read_beh(p)
        assert ru.rejoin(back, ru.beh_units(back)) == ["NA null n/a."]

    def test_missing_onset_is_nan(self):
        beh = _beh([("Film A", [("a", 0), ("b", 0), ("c.", 0)])])
        beh.loc[0, "onset"] = "n/a"
        assert np.isnan(ru.beh_units(beh).onset.iloc[0])

    def test_filler_only_trial_has_no_units(self):
        beh = _beh([("Film A", [("um", 1), ("uh", 1)]), ("Film B", [("a", 0), ("b", 0), ("c.", 0)])])
        assert ru.beh_units(beh).trial_num.tolist() == [2]

    def test_two_films_in_one_trial_raise(self):
        beh = _beh([("Film A", [("a", 0), ("b", 0), ("c.", 0)])])
        beh.loc[2, "movie_name"] = "Film B"
        with pytest.raises(ValueError, match="more than one movie_name"):
            ru.beh_units(beh)


def test_recall_rank_orders_by_session_then_trial():
    u = pd.DataFrame(dict(subject=["98"] * 5, session=["21", "20", "20", "20", "20"], trial_num=[1, 5, 2, 2, 3],
                          stimulus_id=["f"] * 4 + ["g"], unit_idx=[0, 0, 0, 1, 0]))
    r = ru.add_recall_rank(u)
    assert dict(zip(zip(r.session, r.trial_num, r.unit_idx), r.recall_rank)) == {
        ("21", 1, 0): 3, ("20", 5, 0): 2, ("20", 2, 0): 1, ("20", 2, 1): 1, ("20", 3, 0): 1}
