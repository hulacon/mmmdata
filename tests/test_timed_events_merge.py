"""timed_events pairs a behavioural CSV with a timing CSV on trial_id.

The scheduled ``onset`` column is computed from row position, so an inner
merge that silently drops an unmatched trial shifts every later onset by one
trial slot (4.5 s). A pair whose trial sets disagree must fail loudly.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTERS = REPO_ROOT / "src" / "python" / "raw2bids_converters"
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))

import timed_events  # noqa: E402


def _math(trials):
    beh = pd.DataFrame({
        "trial": trials,
        "problem": ["1+1"] * len(trials),
        "answer": [2] * len(trials),
        "resp": [2] * len(trials),
        "resp_RT": [1.0] * len(trials),
    })
    timing = pd.DataFrame({
        "trial_id": trials,
        "stim_image_start": [6.0 + 4.5 * (t - 1) for t in trials],
        "stim_image_end": [9.0 + 4.5 * (t - 1) for t in trials],
    })
    return beh, timing


def test_matched_pair_converts():
    beh, timing = _math([1, 2, 3])
    ev = timed_events.convert_cued_recall_math(beh, timing, 1, 4, 1)
    assert list(ev["onset"]) == [6.0, 10.5, 15.0]
    assert list(ev["onset_actual"]) == [6.0, 10.5, 15.0]


def test_trial_missing_from_timing_raises():
    beh, timing = _math([1, 2, 3, 4])
    timing = timing[timing["trial_id"] != 2]
    with pytest.raises(ValueError, match="trial_id"):
        timed_events.convert_cued_recall_math(beh, timing, 1, 4, 1)


def test_trial_missing_from_behavioural_raises():
    beh, timing = _math([1, 2, 3, 4])
    beh = beh[beh["trial"] != 3]
    with pytest.raises(ValueError, match="trial_id"):
        timed_events.convert_cued_recall_math(beh, timing, 1, 4, 1)


def _retrieval(cue_ids, image_timed, word_timed):
    n = len(cue_ids)
    beh = pd.DataFrame({
        "trial": list(range(1, n + 1)),
        "cueId": cue_ids,
        "pairId": [1] * n, "mmmId": [1] * n, "nsdId": [1] * n,
        "itmno": [1] * n, "word": ["w"] * n, "voiceId": [1] * n,
        "voice": ["v"] * n, "sharedId": [1] * n, "enCon": [1] * n,
        "reCon": [1] * n, "resp": [1] * n, "resp_RT": [1.0] * n,
    })
    def col(flags, val):
        return [val if f else np.nan for f in flags]
    timing = pd.DataFrame({
        "trial_id": list(range(1, n + 1)),
        "stim_image_start": col(image_timed, 9.0),
        "stim_image_end": col(image_timed, 12.0),
        "stim_word_start": col(word_timed, 9.0),
        "stim_word_end": col(word_timed, 9.54),
        "stim_fixation_end": [13.5] * n,
    })
    return beh, timing


def test_retrieval_trial_with_no_stimulus_timing_is_not_labelled_word():
    """A cue trial with neither image nor word timing used to fall through
    to trial_type='word'/modality='auditory' whatever its cueId."""
    beh, timing = _retrieval([1, 1], [True, False], [False, False])
    with pytest.raises(ValueError, match="stimulus timing"):
        timed_events.convert_cued_recall_retrieval(beh, timing, 1, 4, 1)


def test_retrieval_word_trial_still_labelled_word():
    beh, timing = _retrieval([2], [False], [True])
    ev = timed_events.convert_cued_recall_retrieval(beh, timing, 1, 4, 1)
    assert list(ev["trial_type"]) == ["word"]
