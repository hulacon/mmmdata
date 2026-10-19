"""release.pronouns: neutralizing gendered pronouns, with a review report."""

import pytest

from release import pronouns
from release.pronouns import neutralize
from release.stage import Stage
from release.tsvio import Table
from tests.release_fixtures import make_env, write

DEB = "phenotype/final_debriefing.tsv"


@pytest.mark.parametrize("before,after", [
    ("He said she was tired.", "They said they were tired."),
    ("She is sure. He has seen it. She does not.", "They are sure. They have seen it. They do not."),
    ("He isn't sure; she doesn't know; he wasn't there; she hasn't.",
     "They aren't sure; they don't know; they weren't there; they haven't."),
    ("He likes films and she watches them.", "They like films and they watch them."),
    ("She tries hard. He goes home. He also enjoys it. She really misses it.",
     "They try hard. They go home. They also enjoy it. They really miss it."),
    ("He told him about himself; she found it for herself.",
     "They told them about themself; they found it for themself."),
    ("Her strategy was hers, and his notes were his.",
     "Their strategy was theirs, and their notes were theirs."),
    ("I gave her a list. I sat with her. I saw her.", "I gave them a list. I sat with them. I saw them."),
    ("He or she said so. He/she knows.", "They said so. They know."),
    ("He'd go again; she'll come back.", "They'd go again; they'll come back."),
    ("She thus thought so.", "They thus thought so."),
    ("SHE IS", "THEY ARE"),
    ("n/a", "n/a"),
    ("The theme of the shell", "The theme of the shell"),   # no partial-word hits
])
def test_neutralize(before, after):
    assert neutralize(before).after == after


def test_curly_apostrophes_are_kept():
    assert neutralize("He’s here and she doesn’t mind.").after == \
        "They’re here and they don’t mind."


@pytest.mark.parametrize("text,needle", [
    ("The film made her nervous.", "could be 'their'"),
    ("She mentioned her running shoes.", "could be 'them'"),
    ("He focuses on it.", "-ses/-zes"),
    ("She's been tired.", "contraction"),
    ("Was she aware?", "inverted"),
    ("It was his.", "theirs"),
    ("The man spoke.", "gendered term"),
])
def test_uncertain_choices_are_noted(text, needle):
    res = neutralize(text)
    assert any(needle in n for n in res.notes), res.notes


def test_plan_rewrites_text_columns_and_reports_every_changed_cell(tmp_path):
    cfg, work, stage, private = make_env(tmp_path)
    s = Stage(stage, cfg)
    res = pronouns.plan_pronouns(s, DEB)
    s.commit()
    t = Table((stage / DEB).read_text())
    assert res.columns == ["memory_strategies", "favorite_least_favorite"]
    assert t.get(0, "memory_strategies") == \
        "They like to repeat words to themself. They are sure it helped."
    assert t.get(0, "favorite_least_favorite") == \
        "Their favorite part was the films; the scans made them sleepy."
    assert t.get(1, "memory_strategies") == \
        "They try to link each image to their own memories."
    assert t.get(1, "favorite_least_favorite") == "\"Long\" sessions were hard for them."
    # ID and numeric columns untouched
    assert [t.get(i, "participant_id") for i in range(2)] == ["sub-01", "sub-02"]
    assert [t.get(i, "session_id") for i in range(2)] == ["ses-30", "ses-30"]
    assert [t.get(i, "overall_experience") for i in range(2)] == ["4", "5"]
    # report: every changed cell, before and after
    lines = [line.split("\t") for line in pronouns.report_tsv(res).splitlines()]
    rows = [dict(zip(lines[0], r)) for r in lines[1:]]
    assert len(rows) == 4
    first = rows[0]
    assert first["key"] == "participant_id=sub-01&session_id=ses-30"
    assert first["before"].startswith("He likes") and first["after"].startswith("They like")
    review = [r for r in rows if r["status"] == "changed-review"]
    assert len(review) == 1 and "made her sleepy" in review[0]["notes"]
    # working tree untouched (hardlinked)
    assert "He likes" in (work / DEB).read_text()


def test_cells_are_plain_tab_split_and_untouched_rows_stay_byte_identical(tmp_path):
    """BIDS TSVs have no quoting convention: quotes are cell text, kept as is."""
    cfg, work, stage, _ = make_env(tmp_path, hardlink=False)
    write(stage / DEB,
          "participant_id\tnote\r\n"
          "sub-01\t\"She said \"\"fine\"\"\"\r\n"
          "sub-02\t\"Quoted\" text, he   was  here\r\n"
          "sub-01\tNothing to change \"here\"\r\n")
    s = Stage(stage, cfg)
    pronouns.plan_pronouns(s, DEB)
    s.commit()
    text = (stage / DEB).read_bytes().decode()
    assert text == ("participant_id\tnote\r\n"
                    "sub-01\t\"They said \"\"fine\"\"\"\r\n"
                    "sub-02\t\"Quoted\" text, they   were  here\r\n"
                    "sub-01\tNothing to change \"here\"\r\n")


def test_second_run_changes_nothing(tmp_path):
    cfg, work, stage, _ = make_env(tmp_path)
    s = Stage(stage, cfg)
    pronouns.plan_pronouns(s, DEB)
    s.commit()
    s2 = Stage(stage, cfg)
    res = pronouns.plan_pronouns(s2, DEB)
    assert res.n_changed == 0 and not s2.pending


def test_column_override_validates(tmp_path):
    cfg, work, stage, _ = make_env(tmp_path)
    s = Stage(stage, cfg)
    with pytest.raises(ValueError, match="ID column"):
        pronouns.plan_pronouns(s, DEB, ["participant_id"])
    with pytest.raises(KeyError):
        pronouns.plan_pronouns(s, DEB, ["no_such_column"])
    res = pronouns.plan_pronouns(s, DEB, ["memory_strategies"])
    assert {c.column for c in res.changes} == {"memory_strategies"}


def test_missing_debriefing_is_loud(tmp_path):
    cfg, work, stage, _ = make_env(tmp_path, hardlink=False)
    (stage / DEB).unlink()
    with pytest.raises(FileNotFoundError, match="final_debriefing"):
        pronouns.plan_pronouns(Stage(stage, cfg), DEB)
