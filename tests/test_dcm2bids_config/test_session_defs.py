"""Tests for session definitions and schedule mapping."""

import pytest

from src.python.dcm2bids_config.session_defs import (
    SESSION_SCHEDULE,
    SESSION_TYPES,
    AnatDef,
    SessionDef,
    TaskDef,
    get_session_def,
)


class TestTaskDef:
    def test_single_run_protocol_name(self):
        t = TaskDef("TBmath", "cued_recall_math", "encoding")
        assert t.protocol_name(1) == "cued_recall_math"

    def test_multi_run_protocol_name(self):
        t = TaskDef("TBencoding", "cued_recall_encoding_run{n}", "encoding", runs=3)
        assert t.protocol_name(1) == "cued_recall_encoding_run1"
        assert t.protocol_name(3) == "cued_recall_encoding_run3"

    def test_sbref_description(self):
        t = TaskDef("TBencoding", "cued_recall_encoding_run{n}", "encoding", runs=3, has_sbref=True)
        assert t.sbref_description(2) == "cued_recall_encoding_run2_SBRef"

    def test_sbref_single_run(self):
        t = TaskDef("NATmath", "free_recall_math", "encoding", has_sbref=True)
        assert t.sbref_description(1) == "free_recall_math_SBRef"


class TestAnatDef:
    def test_anat_custom_entities(self):
        a = AnatDef("T1w", "MPR", "ABCD_T1w_MPR_vNav")
        assert a.custom_entities == "acq-MPR"

    def test_dwi_custom_entities(self):
        a = AnatDef("dwi", "AP", "cmrr_diff_3shell_ap", datatype="dwi")
        assert a.custom_entities == "dir-AP"


class TestSessionDef:
    def test_task_ids_for_fmap_group_single_run(self):
        sd = SessionDef(
            session_type="test",
            tasks=(
                TaskDef("TBmath", "cued_recall_math", "encoding"),
                TaskDef("TBresting", "cued_recall_resting", "retrieval"),
            ),
            fmap_groups=("encoding", "retrieval"),
        )
        assert sd.task_ids_for_fmap_group("encoding") == ["task_TBmath"]
        assert sd.task_ids_for_fmap_group("retrieval") == ["task_TBresting"]

    def test_task_ids_for_fmap_group_multi_run(self):
        sd = SessionDef(
            session_type="test",
            tasks=(
                TaskDef("TBencoding", "cued_recall_encoding_run{n}", "encoding", runs=3),
                TaskDef("TBmath", "cued_recall_math", "encoding"),
            ),
            fmap_groups=("encoding",),
        )
        ids = sd.task_ids_for_fmap_group("encoding")
        assert ids == [
            "task_TBencoding_run-1",
            "task_TBencoding_run-2",
            "task_TBencoding_run-3",
            "task_TBmath",
        ]


class TestSessionSchedule:
    def test_ses01_is_anatomy(self):
        assert SESSION_SCHEDULE["ses-01"] == "anatomy"

    def test_ses04_is_tb_first(self):
        assert SESSION_SCHEDULE["ses-04"] == "tb_first"

    def test_ses05_through_17_are_tb_middle(self):
        for i in range(5, 18):
            assert SESSION_SCHEDULE[f"ses-{i:02d}"] == "tb_middle"

    def test_ses18_is_tb_last(self):
        assert SESSION_SCHEDULE["ses-18"] == "tb_last"

    def test_ses19_through_28_are_naturalistic(self):
        for i in range(19, 29):
            assert SESSION_SCHEDULE[f"ses-{i:02d}"] == "naturalistic"

    def test_ses29_not_in_schedule(self):
        assert "ses-29" not in SESSION_SCHEDULE

    def test_ses30_is_final(self):
        assert SESSION_SCHEDULE["ses-30"] == "final"

    def test_get_session_def_returns_correct_type(self):
        sd = get_session_def("ses-06")
        assert sd.session_type == "tb_middle"

    def test_get_session_def_raises_for_missing(self):
        with pytest.raises(KeyError):
            get_session_def("ses-29")


class TestSessionTypeCompleteness:
    """Verify each session type has expected structure."""

    def test_anatomy_has_anat_and_dwi(self):
        sd = SESSION_TYPES["anatomy"]
        suffixes = [a.suffix for a in sd.anat]
        assert "T1w" in suffixes
        assert "dwi" in suffixes
        assert sd.fmap_strategy == "none"

    def test_tb_first_has_encoding_and_retrieval(self):
        sd = SESSION_TYPES["tb_first"]
        labels = [t.task_label for t in sd.tasks]
        assert "TBencoding" in labels
        assert "TBretrieval" in labels
        assert "TBmath" in labels
        assert "TBresting" in labels

    def test_tb_first_has_sbref(self):
        sd = SESSION_TYPES["tb_first"]
        assert all(t.has_sbref for t in sd.tasks)

    def test_tb_middle_encoding_has_3_runs(self):
        sd = SESSION_TYPES["tb_middle"]
        enc = next(t for t in sd.tasks if t.task_label == "TBencoding")
        assert enc.runs == 3

    def test_tb_middle_retrieval_has_4_runs(self):
        sd = SESSION_TYPES["tb_middle"]
        ret = next(t for t in sd.tasks if t.task_label == "TBretrieval")
        assert ret.runs == 4

    def test_tb_last_has_no_encoding(self):
        sd = SESSION_TYPES["tb_last"]
        labels = [t.task_label for t in sd.tasks]
        assert "TBencoding" not in labels
        assert "TBretrieval" in labels

    def test_naturalistic_tasks(self):
        sd = SESSION_TYPES["naturalistic"]
        labels = [t.task_label for t in sd.tasks]
        assert labels == ["NATencoding", "NATmath", "NATresting", "NATretrieval"]

    def test_naturalistic_uses_series_description(self):
        sd = SESSION_TYPES["naturalistic"]
        assert sd.fmap_strategy == "series_description"

    def test_naturalistic_fm_uses_series_number(self):
        sd = SESSION_TYPES["naturalistic_fm"]
        assert sd.fmap_strategy == "series_number"

    def test_all_functional_sessions_have_fmap_groups(self):
        for name, sd in SESSION_TYPES.items():
            if name in ("anatomy",):
                continue
            if sd.tasks:  # skip empty localizer/final placeholders
                assert len(sd.fmap_groups) > 0, f"{name} has tasks but no fmap_groups"
