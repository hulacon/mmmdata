"""release.redact: applying a reviewed spec, and advisory flagging."""

import json

import pytest

from release import redact
from release.stage import Stage
from release.tsvio import Table
from tests.release_fixtures import TRANSCRIPT_REL, make_env, write

SIDE_REL = TRANSCRIPT_REL.replace(".tsv", ".json")


def _spec(path, rows):
    lines = ["# reviewed spec", "file\trow\tcolumn\tspan\tnote"]
    lines += ["\t".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def _words(stage):
    t = Table((stage / TRANSCRIPT_REL).read_text())
    return [t.get(i, "word") for i in range(len(t.rows))]


def test_applies_whole_cell_and_span_and_writes_sidecar_note(tmp_path):
    cfg, work, stage, private = make_env(tmp_path)
    spec = _spec(private / "spec.tsv", [
        (TRANSCRIPT_REL, "#4", "word", "*", "place"),
        (TRANSCRIPT_REL, "onset=3.1", "word", "Zorblat", "person"),
        ("phenotype/final_debriefing.tsv", "participant_id=sub-02", "memory_strategies",
         "own memories", "self-reference"),
    ])
    s = Stage(stage, cfg)
    res = redact.plan_redactions(s, redact.load_spec(spec))
    s.commit()
    words = _words(stage)
    assert words[3] == "[REDACTED]"
    assert words[7] == "[REDACTED]."                   # span: punctuation kept
    assert words[0] == "I" and words[8] == "Then"      # others untouched
    assert res.files[TRANSCRIPT_REL] == 2
    side = json.loads((stage / SIDE_REL).read_text())
    assert side["Redaction"] == redact.DEFAULT_NOTE
    assert "[REDACTED]" in side["Redaction"] and "places" in side["Redaction"]
    assert side["word"] == {"Description": "recalled word"}       # kept
    deb = Table((stage / "phenotype/final_debriefing.tsv").read_text())
    assert "her [REDACTED]." in deb.get(1, "memory_strategies")
    # the working tree it was hardlinked from is unchanged
    assert "Quuxville" in (work / TRANSCRIPT_REL).read_text()
    assert "Redaction" not in (work / SIDE_REL).read_text()


def test_sidecar_created_when_absent_and_note_overridable(tmp_path):
    cfg, work, stage, private = make_env(tmp_path, hardlink=False)
    (stage / SIDE_REL).unlink()
    spec = _spec(private / "spec.tsv", [(TRANSCRIPT_REL, "#4", "word", "*", "")])
    s = Stage(stage, cfg)
    redact.plan_redactions(s, redact.load_spec(spec), note="custom note")
    s.commit()
    assert json.loads((stage / SIDE_REL).read_text()) == {"Redaction": "custom note"}


@pytest.mark.parametrize("row,column,span,msg", [
    ("#4", "word", "Springfield", "span not found"),
    ("#99", "word", "*", "out of range"),
    ("onset=9.9", "word", "*", "matches 0 rows"),
    ("#4", "transcript", "*", "no column"),
    ("#4", "word", "Quux", "span not found"),        # whole-word matching
])
def test_unmatched_entry_is_an_error_and_nothing_is_queued(tmp_path, row, column, span, msg):
    cfg, work, stage, private = make_env(tmp_path)
    spec = _spec(private / "spec.tsv", [
        (TRANSCRIPT_REL, "#8", "word", "Zorblat", "good entry first"),
        (TRANSCRIPT_REL, row, column, span, "bad entry"),
    ])
    s = Stage(stage, cfg)
    with pytest.raises(redact.RedactionError, match=msg):
        redact.plan_redactions(s, redact.load_spec(spec))
    assert not s.pending


def test_selector_matching_two_rows_and_missing_file_are_errors(tmp_path):
    cfg, work, stage, private = make_env(tmp_path, hardlink=False)
    write(stage / TRANSCRIPT_REL, "onset\tword\n1\tZorblat\n1\tZorblat\n")
    s = Stage(stage, cfg)
    with pytest.raises(redact.RedactionError, match="matches 2 rows"):
        redact.plan_redactions(s, redact.load_spec(_spec(
            private / "a.tsv", [(TRANSCRIPT_REL, "onset=1", "word", "*", "")])))
    with pytest.raises(redact.RedactionError, match="file not found"):
        redact.plan_redactions(s, redact.load_spec(_spec(
            private / "b.tsv", [("sub-99/nope_beh.tsv", "#1", "word", "*", "")])))


def test_reapplying_a_spec_fails_loudly(tmp_path):
    cfg, work, stage, private = make_env(tmp_path)
    spec = redact.load_spec(_spec(private / "spec.tsv", [
        (TRANSCRIPT_REL, "#4", "word", "Quuxville", "")]))
    s = Stage(stage, cfg)
    redact.plan_redactions(s, spec)
    s.commit()
    with pytest.raises(redact.RedactionError, match="already redacted"):
        redact.plan_redactions(Stage(stage, cfg), spec)


def test_spec_needs_its_columns_and_no_empty_fields(tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("file\trow\tspan\nx.tsv\t#1\t*\n")
    with pytest.raises(redact.RedactionError, match="lacks column"):
        redact.load_spec(bad)
    bad.write_text("file\trow\tcolumn\tspan\nx.tsv\t#1\tword\t\n")
    with pytest.raises(redact.RedactionError, match="empty"):
        redact.load_spec(bad)


def test_id_columns_cannot_be_redacted(tmp_path):
    cfg, work, stage, private = make_env(tmp_path)
    spec = _spec(private / "s.tsv", [("phenotype/final_debriefing.tsv", "#1",
                                      "participant_id", "*", "")])
    with pytest.raises(redact.RedactionError, match="ID column"):
        redact.plan_redactions(Stage(stage, cfg), redact.load_spec(spec))


# ---------------------------------------------------------------- flag ----
def test_flag_lists_candidates_and_says_it_is_advisory(tmp_path):
    cfg, work, stage, private = make_env(tmp_path)
    flags = redact.flag_candidates(Stage(stage, cfg, writable=False))
    by_span = {f.span: f for f in flags}
    assert "Quuxville" in by_span and by_span["Quuxville"].row == "#4"
    assert "Zorblat" in by_span
    assert any("follows 'my friend'" in f.reason for f in flags
               if f.span == "Zorblat")
    assert "I" not in by_span            # allowed capital
    assert "Then" not in by_span         # sentence start after "Zorblat."
    text = redact.flags_tsv(flags)
    assert text.startswith("# ADVISORY ONLY")


def test_flag_list_trimmed_by_a_reviewer_is_a_valid_spec(tmp_path):
    cfg, work, stage, private = make_env(tmp_path)
    flags = redact.flag_candidates(Stage(stage, cfg, writable=False))
    path = private / "flags.tsv"
    path.write_text(redact.flags_tsv(flags))
    entries = redact.load_spec(path)
    s = Stage(stage, cfg)
    redact.plan_redactions(s, entries)
    s.commit()
    assert "Quuxville" not in (stage / TRANSCRIPT_REL).read_text()


def test_flag_per_cell_mode_for_free_text(tmp_path):
    cfg, work, stage, private = make_env(tmp_path, hardlink=False)
    write(stage / "sub-01/ses-01/beh/sub-01_ses-01_task-x_beh.tsv",
          "trial\tresponse\n1\tMy name is Zorblat and I like it\n"
          "2\tWe met near Quuxville once\n")
    flags = redact.flag_candidates(Stage(stage, cfg, writable=False),
                                   globs=("sub-01/ses-01/beh/*_beh.tsv",))
    spans = {f.span for f in flags}
    assert {"Zorblat", "Quuxville"} <= spans
    assert "My" not in spans and "We" not in spans


def test_duplicate_spec_entry_is_an_error(tmp_path):
    spec = _spec(tmp_path / "spec.tsv", [(TRANSCRIPT_REL, "#4", "word", "*", "a"),
                                         (TRANSCRIPT_REL, "#4", "word", "*", "b")])
    with pytest.raises(redact.RedactionError, match="duplicates line"):
        redact.load_spec(spec)
