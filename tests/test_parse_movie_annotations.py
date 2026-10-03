"""A segment is a row with times, not a row with a number.

Several annotation masters leave the SEG-C number cell blank on rows that
carry their own times and description. Until 2026-10-03 the parser skipped
those rows, dropping 127 SEG-C segments (nearly the whole SEG-C pass of three
films) and one SEG-B segment; ``--check`` passed them because it compared the
last offset of both levels pooled, and SEG-B still reached the film's end.
"""

import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import parse_movie_annotations as P  # noqa: E402

C = ["SEG-C Number", "SEG-C Start Time (m.ss)", "SEG-C End Time (m.ss)", "SEG-C Description"]
B = ["SEG-B Number", "Start Time (m.ss)", "End Time (m.ss)", "SEG-B Description"]


def _sheet(monkeypatch, rows):
    raw = pd.DataFrame(rows, columns=["Movie Title", *B, *C])
    monkeypatch.setattr(P.pd, "read_excel", lambda path: raw)
    return P.parse_file(Path("Film_annotation_master_XX.xlsx"), "film")


def test_unnumbered_timed_rows_are_segments(monkeypatch):
    nan = float("nan")
    df = _sheet(monkeypatch, [
        ["FILM", 1, 0.0, 0.3, "start", 1, 0.0, 0.07, "c one"],
        [nan, nan, nan, nan, nan, nan, 0.07, 0.1, "c two, number cell blank"],
        [nan, nan, nan, nan, nan, 2, 0.1, 0.3, "c three"],
        [nan, nan, 0.3, 0.45, "b two, number cell blank", nan, nan, nan, nan],
    ])
    c = df[df["level"] == "C"]
    assert c["seg_number"].tolist() == [1, 2, 3]  # sheet order
    assert c["source_number"].tolist()[0] == 1 and pd.isna(c["source_number"].tolist()[1])
    assert c["onset"].tolist() == [0, 7, 10]
    b = df[df["level"] == "B"]
    assert b["onset"].tolist() == [0, 30] and b["seg_number"].tolist() == [1, 2]


def test_corrections_match_the_sheets_number(monkeypatch):
    nan = float("nan")
    df = _sheet(monkeypatch, [
        ["FILM", 1, 0.0, 0.3, "b", nan, 0.0, 0.05, "unnumbered"],
        [nan, nan, nan, nan, nan, 1, 0.05, 0.2, "sheet says C1"],
    ])
    corr = pd.DataFrame([{"stimulus_id": "film", "level": "C", "seg_number": 1, "field": "offset",
                          "old_mss": "0.2", "new_mss": "0.15", "status": "verified", "authority": "x", "note": ""}])
    out, applied, refused = P.apply_corrections(df, corr, "film")
    assert not refused and len(applied) == 1
    fixed = out[(out["level"] == "C") & (out["seg_number"] == 2)]  # position 2, sheet number 1
    assert fixed["offset"].item() == 15


def test_level_problems_reads_each_level_alone():
    df = pd.DataFrame({
        "level": ["B", "B", "C", "C", "C"], "seg_number": [1, 2, 1, 2, 3],
        "onset": [0, 100, 0, 5, 100], "offset": [100, 200, 5, 7, 108],
    })
    probs = P.level_problems(df, 200, 15)
    assert any(p.startswith("SEG-C last offset 108s") for p in probs)  # SEG-B reaching the end does not cover it
    assert any("SEG-C gap of 93s" in p for p in probs)
    assert not any(p.startswith("SEG-B") for p in probs)
    overlap = P.level_problems(pd.DataFrame({"level": ["C", "C"], "seg_number": [1, 2], "onset": [0, 157],
                                             "offset": [217, 236]}), 236, 15)
    assert overlap == ["SEG-C overlap of 60s between segments 1 and 2 (217s -> 157s)"]
