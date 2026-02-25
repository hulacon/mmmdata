"""Shared fixtures for behavioral analysis tests.

All fixtures create synthetic DataFrames that mirror the real column
structures, enabling unit tests without access to actual data files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_tb2afc() -> pd.DataFrame:
    """Minimal TB2AFC DataFrame with known correct answers.

    Creates 24 trials: 2 subjects x 2 sessions x 6 trials each.
    Conditions: 3 enCon levels, 2 reCon levels.
    """
    rng = np.random.RandomState(42)
    rows = []
    trial_id = 0
    for sub in ("03", "04"):
        for ses in ("04", "05"):
            for i in range(6):
                trial_id += 1
                correct_pos = rng.choice([1, 2])
                # Generate resp: 70% correct for testability
                if rng.rand() < 0.7:
                    # Correct: choose the correct position
                    if correct_pos == 1:
                        resp = rng.choice([1, 2])  # sure/maybe image1
                    else:
                        resp = rng.choice([3, 4])  # sure/maybe image2
                else:
                    # Incorrect: choose the wrong position
                    if correct_pos == 1:
                        resp = rng.choice([3, 4])
                    else:
                        resp = rng.choice([1, 2])
                chose = 1 if resp in (1, 2) else 2
                acc = 1.0 if chose == correct_pos else 0.0
                rows.append({
                    "onset": trial_id * 2.0,
                    "duration": rng.uniform(0.5, 3.0),
                    "subject": sub,
                    "session": ses,
                    "run": "01",
                    "trial_type": "recognition",
                    "modality": "visual",
                    "word": f"word{trial_id}",
                    "image1": f"img{trial_id}a.png",
                    "image2": f"img{trial_id}b.png",
                    "correct_resp": correct_pos,
                    "resp": resp,
                    "resp_RT": rng.uniform(0.5, 4.0),
                    "trial_accuracy": acc,
                    "enCon": (i % 3) + 1,
                    "reCon": (i % 2) + 1,
                    "cueId": 1.0,
                    "pairId": trial_id,
                    "recog": float(chose),
                    "trial_id": trial_id,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_encoding() -> pd.DataFrame:
    """Minimal encoding DataFrame with rest trials included.

    Creates 18 trials (3 per subject per session) + 6 rest trials.
    Scanner responses are raw button box values (6, 7, 8).
    """
    rng = np.random.RandomState(42)
    rows = []
    trial_id = 0
    for sub in ("03", "04"):
        for ses in ("04", "05"):
            for i in range(3):
                trial_id += 1
                rows.append({
                    "onset": trial_id * 4.5,
                    "duration": 3.0,
                    "onset_actual": trial_id * 4.5 + 0.01,
                    "duration_actual": 3.0,
                    "subject": sub,
                    "session": ses,
                    "run": "01",
                    "trial_type": "image",
                    "modality": "visual",
                    "word": f"word{trial_id}",
                    "pairId": trial_id,
                    "mmmId": trial_id + 100,
                    "nsdId": trial_id + 200,
                    "itmno": trial_id,
                    "sharedId": 0,
                    "voiceId": 1,
                    "voice": "echo",
                    "enCon": (i % 3) + 1,
                    "reCon": (i % 2) + 1,
                    "resp": rng.choice([6, 7, 8]),  # Raw button box
                    "resp_RT": rng.uniform(0.5, 2.5),
                    "trial_id": trial_id,
                })
            # Add a rest trial
            rows.append({
                "onset": (trial_id + 1) * 4.5,
                "duration": 3.0,
                "onset_actual": np.nan,
                "duration_actual": np.nan,
                "subject": sub,
                "session": ses,
                "run": "01",
                "trial_type": "rest",
                "modality": np.nan,
                "word": np.nan,
                "pairId": np.nan,
                "mmmId": np.nan,
                "nsdId": np.nan,
                "itmno": np.nan,
                "sharedId": np.nan,
                "voiceId": np.nan,
                "voice": np.nan,
                "enCon": np.nan,
                "reCon": np.nan,
                "resp": np.nan,
                "resp_RT": np.nan,
                "trial_id": trial_id + 1,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_retrieval() -> pd.DataFrame:
    """Minimal retrieval DataFrame."""
    rng = np.random.RandomState(42)
    rows = []
    trial_id = 0
    for sub in ("03", "04"):
        for ses in ("04", "05"):
            for i in range(3):
                trial_id += 1
                rows.append({
                    "onset": trial_id * 4.5,
                    "duration": 3.0,
                    "onset_actual": trial_id * 4.5 + 0.01,
                    "duration_actual": 3.0,
                    "subject": sub,
                    "session": ses,
                    "run": "01",
                    "trial_type": "word",
                    "modality": "visual",
                    "word": f"word{trial_id}",
                    "pairId": trial_id,
                    "mmmId": trial_id + 100,
                    "nsdId": trial_id + 200,
                    "itmno": trial_id,
                    "sharedId": 0,
                    "voiceId": 1,
                    "voice": "echo",
                    "enCon": (i % 3) + 1,
                    "reCon": (i % 2) + 1,
                    "resp": rng.choice([6, 7, 8]),
                    "resp_RT": rng.uniform(0.5, 2.5),
                    "cueId": 1.0,
                    "trial_id": trial_id,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_fintimeline() -> pd.DataFrame:
    """Minimal FINtimeline DataFrame."""
    rng = np.random.RandomState(42)
    rows = []
    for sub in ("03", "04"):
        for i in range(10):
            rows.append({
                "onset": i * 3.0,
                "duration": rng.uniform(1.0, 4.0),
                "trial_type": "timeline",
                "word": f"word{i}",
                "timeline_resp": rng.uniform(0, 1),
                "timeline_RT": rng.uniform(1.0, 5.0),
                "enCon": (i % 3) + 1,
                "reCon": (i % 2) + 1,
                "pairId": i + 1,
                "trial_accuracy": rng.choice([0.0, 1.0]),
                "trial_id": i + 1,
                "subject": sub,
                "session": "30",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_fin2afc(sample_tb2afc) -> pd.DataFrame:
    """Minimal FIN2AFC DataFrame (reuses TB2AFC structure)."""
    df = sample_tb2afc.copy()
    df["session"] = "30"
    df["accuracy"] = df["trial_accuracy"]  # FIN2AFC has both columns
    return df
