"""Missing responses, accuracies and RTs must not be scored as real values.

Each function here used to map a NaN onto a valid level (accurate=True,
confidence "maybe", chosen position 2) or mix values from different rows.
"""

import numpy as np
import pandas as pd

from behavioral.accuracy import confidence_accuracy_curve
from behavioral.encoding import subsequent_memory_effect
from behavioral.final_session import long_term_retention_curve
from behavioral.preprocessing import decompose_2afc_resp, filter_rt
from behavioral.rt import rt_by_accuracy


def test_rt_by_accuracy_excludes_missing_accuracy():
    df = pd.DataFrame({"resp_RT": [1.0, 2.0, 9.0],
                       "trial_accuracy": [1, 0, np.nan]})
    out = rt_by_accuracy(df).set_index("accurate")
    assert out.loc[True, "n_trials"] == 1
    assert out.loc[True, "mean_rt"] == 1.0
    assert out["n_trials"].sum() == 2


def test_confidence_curve_excludes_missing_response():
    df = pd.DataFrame({"resp": [1, 2, np.nan, np.nan],
                       "trial_accuracy": [1, 1, 0, 0]})
    out = confidence_accuracy_curve(df).set_index("confidence_level")
    assert out.loc[1, "n_trials"] == 1
    assert out.loc[1, "accuracy"] == 1.0
    assert out["n_trials"].sum() == 2


def test_decompose_2afc_missing_response_is_not_a_choice():
    df = pd.DataFrame({"resp": [np.nan, 3, 5], "correct_resp": [2, 2, 2]})
    out = decompose_2afc_resp(df)
    assert pd.isna(out.loc[0, "chose_position"])
    assert pd.isna(out.loc[0, "confidence"])
    assert not out.loc[0, "is_correct"]
    assert out.loc[1, "chose_position"] == 2 and out.loc[1, "is_correct"]
    # resp outside 1-4 is not a valid choice either
    assert pd.isna(out.loc[2, "chose_position"])
    assert not out.loc[2, "is_correct"]


def test_filter_rt_by_group_uses_numeric_rt():
    rts = ["1.0"] * 20 + ["50.0"]
    df = pd.DataFrame({"resp_RT": rts, "subject": ["01"] * 21})
    grouped = filter_rt(df, by=["subject"])
    pooled = filter_rt(df)
    assert len(grouped) == len(pooled) == 20


def test_retention_initial_values_come_from_one_session():
    tb = pd.DataFrame({
        "subject": ["01", "01"],
        "session": ["04", "10"],
        "pairId": [7, 7],
        "trial_accuracy": [np.nan, 1.0],
    })
    fin = pd.DataFrame({"subject": ["01"], "pairId": [7], "trial_accuracy": [0.0]})
    out = long_term_retention_curve(tb, fin)
    assert len(out) == 1
    assert out.loc[0, "initial_session"] == "04"
    assert pd.isna(out.loc[0, "initial_accuracy"])


def test_subsequent_memory_rounds_half_ratings_consistently():
    """Mean ratings 1.5, 2.5 and 3.5 all round up; half-to-even used to send
    1.5 and 2.5 to the same bin."""
    enc = pd.DataFrame({
        "subject": ["01"] * 6,
        "pairId": [1, 1, 2, 2, 3, 3],
        "resp": [1, 2, 2, 3, 3, 4],
    })
    rec = pd.DataFrame({"subject": ["01"] * 3, "pairId": [1, 2, 3],
                        "trial_accuracy": [1, 1, 1]})
    out = subsequent_memory_effect(enc, rec)
    assert sorted(out["encoding_rating"]) == [2, 3, 4]
