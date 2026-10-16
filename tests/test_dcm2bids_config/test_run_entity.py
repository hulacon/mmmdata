"""The generated config names every BOLD/SBRef run itself.

Without a ``run-`` entity in ``custom_entities``, dcm2bids 3.2.0 numbers
same-named files by AcquisitionTime: an aborted series that was repeated takes
a number and shifts every later run, a missing run compresses the rest, and a
single-run task acquired twice gains ``run-01``/``run-02``. Run labels then
drift from the behavioural files they are paired with. So:

- a task label with more than one run in the session carries ``run-NN`` from
  the description's run number, decided across every TaskDef with that label;
- a label with one run keeps no ``run`` entity (unchanged file names);
- a description that matches more than one DICOM series without a
  ``run_series`` pin is an error, because dcm2bids would append its own run
  numbers to the explicit one and collide.
"""

from pathlib import Path

import pytest

from src.python.dcm2bids_config.cli import generate_one
from src.python.dcm2bids_config.config_builder import build_config
from src.python.dcm2bids_config.session_defs import SessionDef, TaskDef


def _func(config, suffix):
    return [d for d in config["descriptions"] if d.get("suffix") == suffix]


def _session(*tasks):
    return SessionDef(session_type="test", tasks=tasks, fmap_strategy="none")


# ---------------------------------------------------------------------------
# custom_entities
# ---------------------------------------------------------------------------

class TestRunEntity:
    def test_multi_run_task_carries_run_entity(self):
        sd = _session(TaskDef("motor", "localizer_motor_run{n}", "first",
                              runs=3, has_sbref=True))
        config = build_config("sub-00", "ses-00", sd)
        for suffix in ("bold", "sbref"):
            got = [d["custom_entities"] for d in _func(config, suffix)]
            assert got == ["task-motor_run-01", "task-motor_run-02",
                           "task-motor_run-03"]

    def test_single_run_task_has_no_run_entity(self):
        sd = _session(TaskDef("resting", "resting", "first", has_sbref=True))
        config = build_config("sub-00", "ses-00", sd)
        assert [d["custom_entities"] for d in _func(config, "bold")] == [
            "task-resting"]
        assert [d["custom_entities"] for d in _func(config, "sbref")] == [
            "task-resting"]

    def test_lone_explicit_run_has_no_run_entity(self):
        """One run of a label in the session: one file, so no entity, as now."""
        sd = _session(TaskDef("floc", "localizer_floc_run{n}", "first",
                              runs=(3,)))
        config = build_config("sub-00", "ses-00", sd)
        assert [d["custom_entities"] for d in _func(config, "bold")] == [
            "task-floc"]

    def test_split_explicit_runs_number_consistently(self):
        """[1] + [2, 3]: one label split across fieldmap groups."""
        sd = _session(
            TaskDef("floc", "localizer_floc_run{n}", "first", runs=(1,),
                    has_sbref=True),
            TaskDef("floc", "localizer_floc_run{n}", "second", runs=(2, 3),
                    has_sbref=True),
        )
        config = build_config("sub-00", "ses-00", sd)
        for suffix in ("bold", "sbref"):
            got = [d["custom_entities"] for d in _func(config, suffix)]
            assert got == ["task-floc_run-01", "task-floc_run-02",
                           "task-floc_run-03"]

    def test_unnumbered_plus_explicit_run_number_consistently(self):
        """``runs=1`` + ``runs=[2]``: an un-numbered protocol then ``_run2``."""
        sd = _session(
            TaskDef("tone", "localizer_tone", "second"),
            TaskDef("tone", "localizer_tone_run{n}", "second", runs=(2,)),
        )
        config = build_config("sub-00", "ses-00", sd)
        bolds = _func(config, "bold")
        assert [d["custom_entities"] for d in bolds] == [
            "task-tone_run-01", "task-tone_run-02"]
        assert [d["criteria"]["ProtocolName"] for d in bolds] == [
            "localizer_tone", "localizer_tone_run2"]

    def test_same_run_defined_twice_refuses(self):
        sd = _session(
            TaskDef("tone", "localizer_tone", "second"),
            TaskDef("tone", "localizer_tone_run{n}", "second", runs=(1,)),
        )
        with pytest.raises(ValueError, match="run 1 of task 'tone'"):
            build_config("sub-00", "ses-00", sd)

    def test_labels_are_decided_independently(self):
        sd = _session(
            TaskDef("motor", "localizer_motor_run{n}", "first", runs=2),
            TaskDef("resting", "resting", "first"),
        )
        config = build_config("sub-00", "ses-00", sd)
        assert [d["custom_entities"] for d in _func(config, "bold")] == [
            "task-motor_run-01", "task-motor_run-02", "task-resting"]

    def test_pinned_run_keeps_its_protocol_run_number(self):
        """Two series share one protocol; the pins decide which is which."""
        sd = _session(TaskDef("FINretrieval", "final_cued_recall_run{n}",
                              "retrieval", runs=2, has_sbref=True))
        config = build_config(
            "sub-00", "ses-00", sd,
            run_protocols={"FINretrieval": {1: "final_cued_recall_run2"}},
            run_series={"FINretrieval": {1: {"bold": 12, "sbref": 11},
                                         2: {"bold": 16, "sbref": 15}}},
        )
        bolds = _func(config, "bold")
        assert [d["custom_entities"] for d in bolds] == [
            "task-FINretrieval_run-01", "task-FINretrieval_run-02"]
        assert [d["criteria"]["SeriesNumber"] for d in bolds] == ["12", "16"]


# ---------------------------------------------------------------------------
# Duplicate series: an error unless pinned
# ---------------------------------------------------------------------------

def _overrides(root: Path, body: str) -> Path:
    ovr = root / "ovr"
    (ovr / "sub-00").mkdir(parents=True)
    (ovr / "sub-00" / "overrides.toml").write_text(body)
    return ovr


MOTOR_TWO_RUNS = """
[ses-01]
fmap_strategy = "none"
fmap_groups = []
[[ses-01.tasks]]
task_label = "motor"
protocol_base = "localizer_motor_run{n}"
fmap_group = "first"
runs = 2
has_sbref = true
"""


def _dicoms(root: Path, series: dict[int, tuple[str, int]]) -> Path:
    """Fake ``<source>/sub-00/ses-01/dicom/Series_NN_<description>`` dirs."""
    src = root / "src"
    dicom = src / "sub-00" / "ses-01" / "dicom"
    for number, (description, n_files) in series.items():
        d = dicom / f"Series_{number:02d}_{description}"
        d.mkdir(parents=True)
        for i in range(n_files):
            (d / f"{i:04d}.dcm").touch()
    return src


CLEAN = {
    10: ("localizer_motor_run1_SBRef", 1),
    11: ("localizer_motor_run1", 30),
    12: ("localizer_motor_run1_PhysioLog", 1),
    14: ("localizer_motor_run2_SBRef", 1),
    15: ("localizer_motor_run2", 30),
}


class TestDuplicateSeries:
    def test_clean_session_is_written(self, tmp_path):
        r = generate_one("sub-00", "ses-01", _dicoms(tmp_path, CLEAN),
                         _overrides(tmp_path, MOTOR_TWO_RUNS))
        assert r["status"] == "written", r["warnings"]

    def test_unpinned_bold_duplicate_is_an_error(self, tmp_path):
        """An aborted run 1 repeated under the same protocol."""
        series = dict(CLEAN)
        series[8] = ("localizer_motor_run1_SBRef", 1)
        series[9] = ("localizer_motor_run1", 5)
        ovr = _overrides(tmp_path, MOTOR_TWO_RUNS)
        r = generate_one("sub-00", "ses-01", _dicoms(tmp_path, series), ovr)
        assert r["status"] == "error"
        msg = " ".join(r["warnings"])
        assert "task-motor_run-01" in msg
        assert "Series 9" in msg and "Series 11" in msg
        assert "run_series" in msg
        assert "output_path" not in r
        assert not (ovr / "sub-00" / "ses-01_conf.json").exists()

    def test_duplicate_is_an_error_in_dry_run_too(self, tmp_path):
        series = dict(CLEAN)
        series[9] = ("localizer_motor_run1", 5)
        r = generate_one("sub-00", "ses-01", _dicoms(tmp_path, series),
                         _overrides(tmp_path, MOTOR_TWO_RUNS), dry_run=True)
        assert r["status"] == "error"

    def test_sbref_duplicate_is_an_error(self, tmp_path):
        """One BOLD, two SBRefs: the SBRef would be renumbered on its own."""
        series = dict(CLEAN)
        series[13] = ("localizer_motor_run2_SBRef", 1)
        r = generate_one("sub-00", "ses-01", _dicoms(tmp_path, series),
                         _overrides(tmp_path, MOTOR_TWO_RUNS))
        assert r["status"] == "error"
        assert "task-motor_run-02 sbref" in " ".join(r["warnings"])

    def test_pinned_duplicate_is_written(self, tmp_path):
        series = dict(CLEAN)
        series[8] = ("localizer_motor_run1_SBRef", 1)
        series[9] = ("localizer_motor_run1", 5)
        body = MOTOR_TWO_RUNS + (
            "[ses-01.run_series.motor]\n1 = {bold = 11, sbref = 10}\n")
        r = generate_one("sub-00", "ses-01", _dicoms(tmp_path, series),
                         _overrides(tmp_path, body))
        assert r["status"] == "written", r["warnings"]
        assert any("SeriesNumber pin" in w for w in r["warnings"])

    def test_bold_pinned_but_sbref_not_is_an_error(self, tmp_path):
        series = dict(CLEAN)
        series[8] = ("localizer_motor_run1_SBRef", 1)
        series[9] = ("localizer_motor_run1", 5)
        body = MOTOR_TWO_RUNS + "[ses-01.run_series.motor]\n1 = {bold = 11}\n"
        r = generate_one("sub-00", "ses-01", _dicoms(tmp_path, series),
                         _overrides(tmp_path, body))
        assert r["status"] == "error"
        assert "task-motor_run-01 sbref" in " ".join(r["warnings"])

    def test_single_run_task_acquired_twice_is_an_error(self, tmp_path):
        body = """
[ses-01]
fmap_strategy = "none"
fmap_groups = []
[[ses-01.tasks]]
task_label = "resting"
protocol_base = "resting"
fmap_group = "first"
"""
        series = {11: ("resting", 30), 13: ("resting", 30)}
        r = generate_one("sub-00", "ses-01", _dicoms(tmp_path, series),
                         _overrides(tmp_path, body))
        assert r["status"] == "error"

    def test_shared_protocol_pinned_per_run_is_written(self, tmp_path):
        """Runs 1 and 2 both acquired under the run-2 protocol, each pinned."""
        body = """
[ses-01]
fmap_strategy = "none"
fmap_groups = []
[[ses-01.tasks]]
task_label = "FINretrieval"
protocol_base = "final_cued_recall_run{n}"
fmap_group = "retrieval"
runs = 2
has_sbref = true
[ses-01.run_protocols.FINretrieval]
1 = "final_cued_recall_run2"
[ses-01.run_series.FINretrieval.1]
bold = 12
sbref = 11
[ses-01.run_series.FINretrieval.2]
bold = 16
sbref = 15
"""
        series = {
            11: ("final_cued_recall_run2_SBRef", 1),
            12: ("final_cued_recall_run2", 30),
            15: ("final_cued_recall_run2_SBRef", 1),
            16: ("final_cued_recall_run2", 30),
        }
        r = generate_one("sub-00", "ses-01", _dicoms(tmp_path, series),
                         _overrides(tmp_path, body))
        assert r["status"] == "written", r["warnings"]
        bolds = _func(r["config"], "bold")
        assert [d["custom_entities"] for d in bolds] == [
            "task-FINretrieval_run-01", "task-FINretrieval_run-02"]

    def test_re_reconstruction_beside_original_is_not_a_duplicate(self, tmp_path):
        """``<protocol>_RR`` matched by run_descriptions; the original stays out."""
        body = MOTOR_TWO_RUNS + (
            '[ses-01.run_descriptions.motor]\n1 = "localizer_motor_run1_RR"\n')
        series = dict(CLEAN)
        series[20] = ("localizer_motor_run1_RR_SBRef", 1)
        series[21] = ("localizer_motor_run1_RR", 30)
        r = generate_one("sub-00", "ses-01", _dicoms(tmp_path, series),
                         _overrides(tmp_path, body))
        assert r["status"] == "written", r["warnings"]

    def test_missing_dicom_dir_says_the_check_was_skipped(self, tmp_path):
        r = generate_one("sub-00", "ses-01", tmp_path / "nowhere",
                         _overrides(tmp_path, MOTOR_TWO_RUNS))
        assert r["status"] == "written"
        assert any("duplicate-series check skipped" in w for w in r["warnings"])
