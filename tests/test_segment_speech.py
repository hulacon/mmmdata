"""segment_speech: a segment's speech share under the diarize frame gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1] / "scripts" / "benchmarks"


@pytest.fixture(scope="module")
def ss():
    spec = importlib.util.spec_from_file_location("segment_speech", HERE / "segment_speech.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_frame_coverage_counts_overlapping_turns_once(ss):
    turns = np.array([[0.1, 0.4], [0.2, 0.45], [0.9, 1.2]])
    cov = ss.frame_coverage(turns, np.array([0.0, 0.5, 1.0]), 0.5)
    assert cov == pytest.approx([0.7, 0.2, 0.4])


def test_build_gates_frames_and_lists_items(ss, tmp_path):
    movies = tmp_path / "movies"
    (movies / "film-a").mkdir(parents=True)
    (movies / "film-b").mkdir(parents=True)
    pd.DataFrame({"stimulus_id": "film-a", "turn_idx": [0], "speaker": "S0", "onset": [0.0], "offset": [1.1]}
                 ).to_csv(movies / "film-a" / "diarize_speakers.csv", index=False)
    pd.DataFrame(columns=["stimulus_id", "turn_idx", "speaker", "onset", "offset"]
                 ).to_csv(movies / "film-b" / "diarize_speakers.csv", index=False)
    segs = pd.DataFrame({"stimulus_id": ["film-a", "film-a", "film-b"], "chunk_idx": [0, 1, 0],
                         "onset": [0.0, 2.0, 0.0], "offset": [2.0, 3.0, 1.0]})
    sp = tmp_path / "segs.csv"
    segs.to_csv(sp, index=False)
    out = tmp_path / "speech.csv"
    assert ss.main(["--segments", str(sp), "-o", str(out), "--movies-dir", str(movies), "--lists", str(tmp_path)]) == 0
    r = pd.read_csv(out)
    # film-a segment 0: frames 0, .5, 1.0, 1.5; coverage 1, 1, .2, 0 -> 2 of 4 gated at >= .25
    assert r["n_speech_frames"].tolist() == [2, 0, 0] and r["speech_share"].tolist() == [0.5, 0.0, 0.0]
    assert (tmp_path / "speech.txt").read_text().split() == ["film-a|0"]
    assert (tmp_path / "nonspeech.txt").read_text().split() == ["film-a|1", "film-b|0"]


def test_build_refuses_a_film_without_a_diarize_file(ss, tmp_path):
    segs = pd.DataFrame({"stimulus_id": ["film-z"], "chunk_idx": [0], "onset": [0.0], "offset": [1.0]})
    with pytest.raises(SystemExit, match="no diarize file"):
        ss.build(segs, tmp_path, window=0.5, min_coverage=0.25)
