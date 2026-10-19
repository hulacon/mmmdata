"""release.dates: per-participant shift of sessions.tsv acq_time."""

import datetime as dt
import sys
from pathlib import Path

import pytest

from release import dates
from release.stage import Stage
from release.tsvio import Table
from tests.release_fixtures import EPOCH, make_env, write, write_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import release_export  # noqa: E402


def _acq(stage_root, sub):
    t = Table((stage_root / f"sub-{sub}" / f"sub-{sub}_sessions.tsv").read_text())
    return [t.get(i, "acq_time") for i in range(len(t.rows))]


def _days(values):
    return [dt.date.fromisoformat(v[:10]) for v in values if v != "n/a"]


def test_shift_maps_first_date_to_epoch_and_preserves_intervals(tmp_path):
    cfg, work, stage, _ = make_env(tmp_path)
    before = {s: _acq(stage, s) for s in ("01", "02")}
    s = Stage(stage, cfg)
    plan = dates.plan_date_shift(s, EPOCH)
    s.commit()
    for sub in ("01", "02"):
        old, new = _days(before[sub]), _days(_acq(stage, sub))
        assert min(new) == EPOCH
        assert [(d - min(old)).days for d in old] == [(d - EPOCH).days for d in new]
    # n/a passes through; a time of day survives; row order is untouched
    assert _acq(stage, "02") == ["1900-01-31T10:15:00", "1900-01-01", "n/a"]
    assert {p.participant: p.status for p in plan.participants} == {
        "sub-01": "shifted", "sub-02": "shifted"}


def test_shift_leaves_other_columns_and_scans_tsv_byte_identical(tmp_path):
    cfg, work, stage, _ = make_env(tmp_path)
    scans = (stage / "sub-01/ses-01/sub-01_ses-01_scans.tsv").read_bytes()
    s = Stage(stage, cfg)
    dates.plan_date_shift(s, EPOCH)
    s.commit()
    assert (stage / "sub-01/ses-01/sub-01_ses-01_scans.tsv").read_bytes() == scans
    t = Table((stage / "sub-01/sub-01_sessions.tsv").read_text())
    assert [t.get(i, "session_note") for i in range(3)] == ["n/a", "fine", "n/a"]


def test_offsets_mapping_recovers_originals(tmp_path):
    cfg, work, stage, private = make_env(tmp_path)
    before = _days(_acq(stage, "01"))
    s = Stage(stage, cfg)
    plan = dates.plan_date_shift(s, EPOCH)
    s.commit()
    rows = [line.split("\t") for line in dates.offsets_tsv(plan).splitlines()]
    assert rows[0] == ["participant_id", "epoch", "offset_days"]
    offset = int(dict((r[0], r[2]) for r in rows[1:])["sub-01"])
    assert [d + dt.timedelta(days=offset) for d in _days(_acq(stage, "01"))] == before


def test_cli_writes_mapping_outside_stage_and_never_prints_offsets(tmp_path, capsys):
    cfg, work, stage, private = make_env(tmp_path)
    cdir = write_config(tmp_path, cfg)
    out = private / "offsets.tsv"
    assert release_export.main(["--config-dir", str(cdir), "shift", str(stage),
                                "--offsets-out", str(out)]) == 0
    printed = capsys.readouterr().out
    offsets = [line.split("\t")[2] for line in out.read_text().splitlines()[1:]]
    assert len(offsets) == 2
    for off in offsets:
        assert off not in printed
    assert not list(stage.rglob("*offset*"))


def test_cli_refuses_mapping_inside_stage(tmp_path, capsys):
    cfg, work, stage, _ = make_env(tmp_path)
    cdir = write_config(tmp_path, cfg)
    before = (stage / "sub-01/sub-01_sessions.tsv").read_text()
    rc = release_export.main(["--config-dir", str(cdir), "shift", str(stage),
                              "--offsets-out", str(stage / "offsets.tsv")])
    assert rc == 2
    assert "inside the staged copy" in capsys.readouterr().err
    assert (stage / "sub-01/sub-01_sessions.tsv").read_text() == before
    assert not (stage / "offsets.tsv").exists()


def test_cli_refuses_the_bids_root(tmp_path, capsys):
    cfg, work, stage, private = make_env(tmp_path)
    cdir = write_config(tmp_path, cfg)
    rc = release_export.main(["--config-dir", str(cdir), "shift", str(work),
                              "--offsets-out", str(private / "o.tsv")])
    assert rc == 2
    assert "BIDS root" in capsys.readouterr().err


def test_second_run_is_a_loud_no_op(tmp_path):
    cfg, work, stage, _ = make_env(tmp_path)
    s = Stage(stage, cfg)
    dates.plan_date_shift(s, EPOCH)
    s.commit()
    once = {sub: _acq(stage, sub) for sub in ("01", "02")}
    s2 = Stage(stage, cfg)
    plan = dates.plan_date_shift(s2, EPOCH)
    assert all(p.status == "already-shifted" for p in plan.participants)
    assert not s2.pending and not plan.shifted
    assert any("ALREADY SHIFTED" in line for line in plan.summary_lines())
    assert {sub: _acq(stage, sub) for sub in ("01", "02")} == once


def test_mixed_shifted_and_unshifted_dates_are_refused(tmp_path):
    cfg, work, stage, _ = make_env(tmp_path, hardlink=False)
    write(stage / "sub-01/sub-01_sessions.tsv",
          "session_id\tacq_time\nses-01\t1900-01-01\nses-02\t2031-01-08\n")
    s = Stage(stage, cfg)
    with pytest.raises(dates.DateShiftError, match="already shifted"):
        dates.plan_date_shift(s, EPOCH)
    assert not s.pending           # nothing queued for any participant


def test_malformed_date_is_refused(tmp_path):
    cfg, work, stage, _ = make_env(tmp_path, hardlink=False)
    write(stage / "sub-02/sub-02_sessions.tsv", "session_id\tacq_time\nses-01\t03/04/2031\n")
    with pytest.raises(dates.DateShiftError, match="not YYYY-MM-DD"):
        dates.plan_date_shift(Stage(stage, cfg), EPOCH)


def test_span_that_would_pass_1925_is_refused(tmp_path):
    cfg, work, stage, _ = make_env(tmp_path, hardlink=False)
    write(stage / "sub-02/sub-02_sessions.tsv",
          "session_id\tacq_time\nses-01\t2000-01-01\nses-02\t2030-01-01\n")
    with pytest.raises(dates.DateShiftError, match="after 1925"):
        dates.plan_date_shift(Stage(stage, cfg), EPOCH)


def test_hardlinked_working_tree_file_is_left_unchanged(tmp_path):
    cfg, work, stage, _ = make_env(tmp_path, hardlink=True)
    wt = work / "sub-01/sub-01_sessions.tsv"
    st = stage / "sub-01/sub-01_sessions.tsv"
    assert wt.stat().st_ino == st.stat().st_ino            # truly hardlinked
    original = wt.read_bytes()
    s = Stage(stage, cfg)
    dates.plan_date_shift(s, EPOCH)
    s.commit()
    assert wt.read_bytes() == original
    assert st.read_bytes() != original
    assert wt.stat().st_ino != st.stat().st_ino
    assert wt.stat().st_nlink == 1
