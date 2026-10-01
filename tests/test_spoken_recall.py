"""Tests for the scanner time base of raw2bids_converters/spoken_recall.py.

Every fixture is synthetic (subject 98) and built under tmp_path; nothing
reads the real dataset.
"""

import contextlib
import json
import sys
import wave
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTERS = REPO_ROOT / "src" / "python" / "raw2bids_converters"
# The converter modules import each other by bare name (``from common import
# ...``), which resolves only when their own directory is on sys.path -- as it
# is when they run as scripts from that directory.
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))

from raw2bids_converters import spoken_recall as sr  # noqa: E402

SUBJ, SES = 98, 19
SUB, SESL = "sub-98", "ses-19"
# Placeholder recording stamps: they only need to parse and sort.
STAMPS = ["2000-01-01_10h00.00.000", "2000-01-01_10h05.00.000"]


def _trials(routines=((12.0, 100.0), (100.5, 200.0))):
    rows = []
    for i, (start, stop) in enumerate(routines, 1):
        rows.append({"onset": start + 8.0, "duration": stop - start - 8.0,
                     "trial_type": "recall", "trial_num": i,
                     "movie_name": f"Film {i}", "recall1.started": start,
                     "recall1.stopped": stop})
    return pd.DataFrame(rows)


def _words(rows):
    return pd.DataFrame(rows, columns=["word", "start", "end"])


def _write_wav(path, seconds, rate=1000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(wave.open(str(path), "wb")) as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))


@pytest.fixture
def roots(tmp_path, monkeypatch):
    bids, source = tmp_path / "bids", tmp_path / "source"
    monkeypatch.setattr(sr, "BIDS_ROOT", str(bids))
    monkeypatch.setattr(sr, "SOURCE_DIR", str(source))
    return bids, source


def _session(bids, source, runs=(None,), events_run=None, wav_seconds=(87.4, 98.9),
             transcripts=None):
    func = bids / SUB / SESL / "func"
    beh = bids / SUB / SESL / "beh"
    func.mkdir(parents=True)
    beh.mkdir(parents=True)
    for run in runs:
        part = f"_run-{run}" if run else ""
        (func / f"{SUB}_{SESL}_task-NATretrieval{part}_bold.nii.gz").touch()
    epart = f"_run-{events_run}" if events_run else ""
    _trials().to_csv(func / f"{SUB}_{SESL}_task-NATretrieval{epart}_events.tsv",
                     sep="\t", index=False)
    transcripts = transcripts or [
        "word,start,end\nOkay,9.0,9.4\num,10.0,10.2\n",
        "word,start,end\nNone,9.5,9.9\n",
    ]
    for stamp, text, secs in zip(STAMPS, transcripts, wav_seconds):
        (beh / f"recording_mic_{stamp}_word_timestamps.csv").write_text(text)
        if secs is not None:
            _write_wav(source / SUB / SESL / "audio" / f"recording_mic_{stamp}.wav",
                       secs)


class TestScannerEvents:
    def test_onset_is_routine_start_plus_recording_time(self):
        table = sr.scanner_events(
            _trials(), [(_words([["a", 9.0, 9.5]]), 87.4),
                        (_words([["b", 10.0, 10.25]]), 98.9)])
        assert table["onset"].tolist() == [21.0, 110.5]
        assert table["duration"].tolist() == [0.5, 0.25]
        assert table["recording_onset"].tolist() == [9.0, 10.0]
        assert set(table["time_zero"]) == {"routine_start"}

    def test_count_mismatch_refuses(self):
        with pytest.raises(ValueError, match="1 transcripts for 2 recall"):
            sr.scanner_events(_trials(), [(_words([["a", 9.0, 9.5]]), 87.4)])

    def test_typo_start_masks_onset_keeps_vetted_value(self):
        words = _words([["w", 9.0 + i, 9.5 + i] for i in range(30)]
                       + [["typo", 84063.0, 85.1]])
        table = sr.scanner_events(_trials(), [(words, 87.4),
                                              (_words([]), 98.9)])
        typo = table[table["word"] == "typo"].iloc[0]
        assert pd.isna(typo["onset"]) and pd.isna(typo["duration"])
        assert typo["recording_onset"] == 84063.0

    def test_end_before_start_masks_duration_only(self):
        table = sr.scanner_events(
            _trials(), [(_words([["w", 20.4, 2.9]]), 87.4), (_words([]), 98.9)])
        assert table["onset"].iloc[0] == pytest.approx(32.4)
        assert pd.isna(table["duration"].iloc[0])

    def test_mostly_off_window_trial_is_a_pairing_error(self):
        words = _words([["w", 500.0 + i, 500.5 + i] for i in range(5)])
        with pytest.raises(ValueError, match="wrong trial"):
            sr.scanner_events(_trials(), [(words, 87.4), (_words([]), 98.9)])

    def test_zero_is_assumed_when_recording_short_or_missing(self):
        table = sr.scanner_events(
            _trials(), [(_words([["late", 80.0, 80.5]]), 50.0),
                        (_words([["x", 9.0, 9.2]]), None)])
        assert set(table["time_zero"]) == {"routine_start_assumed"}

    def test_withheld_trials_are_dropped(self):
        table = sr.scanner_events(
            _trials(), [(_words([["a", 9.0, 9.5]]), 87.4),
                        (_words([["b", 10.0, 10.2]]), 98.9)], withheld={1})
        assert table["trial_num"].tolist() == [2]

    def test_filler_flag_ignores_punctuation(self):
        table = sr.scanner_events(
            _trials(), [(_words([["Um...", 9.0, 9.5], ["film", 9.6, 9.9]]), 87.4),
                        (_words([]), 98.9)])
        assert table["filler"].tolist() == [1, 0]


class TestReadVettedTranscript:
    def test_cp1252_trailing_columns_blank_and_none(self, tmp_path):
        path = tmp_path / "t.csv"
        path.write_bytes(b"word,start,end,,\nCaf\xe9,1.0,1.5,,\n"
                         b"None,2.0,2.2,,\n,3.0,3.1,,\n")
        words, blank = sr.read_vetted_transcript(path)
        assert words["word"].tolist() == ["Café", "None"]
        assert words["start"].tolist() == [1.0, 2.0]
        assert blank == 1

    def test_byte_order_mark(self, tmp_path):
        path = tmp_path / "t.csv"
        path.write_bytes(b"\xef\xbb\xbfword,start,end\nx,1.0,1.2\n")
        words, _ = sr.read_vetted_transcript(path)
        assert words["word"].tolist() == ["x"]

    def test_missing_column_refuses(self, tmp_path):
        path = tmp_path / "t.csv"
        path.write_text("word,start\nx,1.0\n")
        with pytest.raises(ValueError, match="missing columns"):
            sr.read_vetted_transcript(path)


class TestConvertSession:
    def test_single_run_file_has_no_run_entity(self, roots):
        bids, source = roots
        _session(bids, source)
        tsv = sr.convert_nat_session(SUBJ, SES)
        assert tsv.endswith(f"beh/{SUB}_{SESL}_task-NATretrieval_beh.tsv")
        table = pd.read_csv(tsv, sep="\t", keep_default_na=False)
        assert table["word"].tolist() == ["Okay", "um", "None"]
        side = json.loads(Path(tsv.replace(".tsv", ".json")).read_text())
        assert side["TranscriptStatus"] == "corrected"
        assert "WithheldTrials" not in side

    def test_originals_untouched(self, roots):
        bids, source = roots
        _session(bids, source)
        beh = bids / SUB / SESL / "beh"
        before = {p.name: p.read_bytes() for p in beh.iterdir()}
        sr.convert_nat_session(SUBJ, SES)
        after = {p.name: p.read_bytes() for p in beh.iterdir() if p.name in before}
        assert after == before

    def test_runless_events_with_two_runs_needs_a_map(self, roots):
        bids, source = roots
        _session(bids, source, runs=("01", "02"))
        with pytest.raises(ValueError, match="EVENTS_RUN"):
            sr.convert_nat_session(SUBJ, SES)

    def test_mapped_run_names_the_bold_run(self, roots, monkeypatch):
        bids, source = roots
        _session(bids, source, runs=("01", "02"))
        monkeypatch.setattr(sr, "EVENTS_RUN", {(SUBJ, SES): "02"})
        monkeypatch.setattr(sr, "WITHHELD_TRIALS", {(SUBJ, SES): {1}})
        tsv = sr.convert_nat_session(SUBJ, SES)
        assert tsv.endswith(f"{SUB}_{SESL}_task-NATretrieval_run-02_beh.tsv")
        side = json.loads(Path(tsv.replace(".tsv", ".json")).read_text())
        assert side["WithheldTrials"]["trial_num"] == [1]

    def test_events_run_entity_wins(self, roots):
        bids, source = roots
        _session(bids, source, runs=("01", "02"), events_run="01")
        assert sr.convert_nat_session(SUBJ, SES).endswith("_run-01_beh.tsv")

    def test_dry_run_writes_nothing(self, roots):
        bids, source = roots
        _session(bids, source)
        sr.convert_nat_session(SUBJ, SES, dry_run=True)
        assert not list((bids / SUB / SESL / "beh").glob("*_beh.tsv"))

    def test_nat_sessions_found(self, roots):
        bids, source = roots
        _session(bids, source)
        assert sr.nat_sessions(SUBJ) == [SES]


def _memo_words(rows):
    return pd.DataFrame(rows, columns=["word", "start", "end", "movie"])


class TestSpeechBounds:
    def test_one_event_per_film_first_to_last_word(self):
        table = pd.DataFrame({
            "onset": [10.0, 12.0, 30.0, None], "duration": [0.5, 1.0, 0.2, 0.3],
            "trial_num": [2, 2, 1, 1], "movie_name": ["B", "B", "A", "A"]})
        ev = sr.speech_bounds(table)
        assert ev["trial_num"].tolist() == [2, 1]
        assert ev["onset"].tolist() == [10.0, 30.0]
        assert ev["duration"].tolist() == [3.0, 0.2]


class TestMemoEvents:
    def test_onset_is_memo_time_minus_lead(self):
        table = sr.memo_events(
            _memo_words([["Um...", 30.0, 30.5, "Film 2"], ["b", 40.0, 40.2, "Film 1"]]),
            _trials(), lead_s=20.0, run_seconds=100.0)
        assert table["onset"].tolist() == [10.0, 20.0]
        assert table["trial_num"].tolist() == [2, 1]
        assert table["recording_onset"].tolist() == [30.0, 40.0]
        assert table["filler"].tolist() == [1, 0]
        assert set(table["time_zero"]) == {"memo_aligned"}

    def test_word_outside_run_masks_onset(self):
        words = _memo_words([["w", 30.0 + i, 30.5 + i, "Film 1"] for i in range(30)]
                            + [["late", 500.0, 500.5, "Film 1"]])
        table = sr.memo_events(words, _trials(), lead_s=20.0, run_seconds=100.0)
        late = table[table["word"] == "late"].iloc[0]
        assert pd.isna(late["onset"]) and pd.isna(late["duration"])

    def test_mostly_outside_run_means_wrong_offset(self):
        words = _memo_words([["w", 5.0 + i, 5.5 + i, "Film 1"] for i in range(5)])
        with pytest.raises(ValueError, match="offset is wrong"):
            sr.memo_events(words, _trials(), lead_s=20.0, run_seconds=100.0)

    def test_unknown_film_refuses(self):
        with pytest.raises(ValueError, match="not among"):
            sr.memo_events(_memo_words([["w", 30.0, 30.5, "Film 9"]]), _trials(),
                           lead_s=20.0, run_seconds=100.0)


class TestConvertMemoRun:
    def _memo_session(self, bids, source, monkeypatch):
        _session(bids, source, runs=("01", "02"))
        func = bids / SUB / SESL / "func"
        stem = f"{SUB}_{SESL}_task-NATretrieval_run-01_bold"
        (func / f"{stem}.nii.gz").unlink()
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 40), np.int16), np.eye(4)),
                 str(func / f"{stem}.nii.gz"))
        (func / f"{stem}.json").write_text(json.dumps({"RepetitionTime": 1.5}))
        (bids / SUB / SESL / "beh" / "memo_word_timestamps.csv").write_text(
            "word,start,end,movie\nOkay,25.0,25.4,Film 1\nthen,70.0,70.3,Film 2\n")
        monkeypatch.setattr(sr, "EVENTS_RUN", {(SUBJ, SES): "02"})
        monkeypatch.setattr(sr, "MEMO_RUNS", {(SUBJ, SES): {
            "run": "01", "transcript": "memo_word_timestamps.csv",
            "lead_s": 20.0, "uncertainty_s": 0.2}})

    def test_memo_run_written_beside_events_run(self, roots, monkeypatch):
        bids, source = roots
        self._memo_session(bids, source, monkeypatch)
        sr.convert_nat_session(SUBJ, SES)
        beh = bids / SUB / SESL / "beh"
        tsv = beh / f"{SUB}_{SESL}_task-NATretrieval_run-01_beh.tsv"
        table = pd.read_csv(tsv, sep="\t")
        assert table["onset"].tolist() == [5.0, 50.0]
        side = json.loads(tsv.with_suffix(".json").read_text())
        assert "20.0" in side["TimeBase"] and "SpeechBoundedEvents" in side
        assert (beh / f"{SUB}_{SESL}_task-NATretrieval_run-02_beh.tsv").exists()
        ev = pd.read_csv(bids / SUB / SESL / "func"
                         / f"{SUB}_{SESL}_task-NATretrieval_run-01_events.tsv", sep="\t")
        assert ev["trial_type"].tolist() == ["recall_speech"] * 2
        assert ev["onset"].tolist() == [5.0, 50.0]
        assert ev["duration"].tolist() == [0.4, 0.3]

    def test_rerun_ignores_the_memo_runs_own_events(self, roots, monkeypatch):
        bids, source = roots
        self._memo_session(bids, source, monkeypatch)
        sr.convert_nat_session(SUBJ, SES)
        assert sr.convert_nat_session(SUBJ, SES).endswith("_run-02_beh.tsv")

    def test_memo_run_with_the_events_file_refuses(self, roots, monkeypatch):
        bids, source = roots
        self._memo_session(bids, source, monkeypatch)
        monkeypatch.setattr(sr, "EVENTS_RUN", {(SUBJ, SES): "01"})
        with pytest.raises(ValueError, match="has the events file"):
            sr.convert_memo_run(SUBJ, SES)
