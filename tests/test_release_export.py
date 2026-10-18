"""release.export / scripts/release_export.py: the orchestrated export."""

import sys
from pathlib import Path

import pytest

from release.export import ExportOptions, run_export
from release.stage import StageError
from tests.release_fixtures import (EPOCH, TRANSCRIPT_REL, make_env, snapshot,
                                    write_config)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import release_export  # noqa: E402


def _inputs(private):
    spec = private / "spec.tsv"
    spec.write_text("file\trow\tcolumn\tspan\n"
                    f"{TRANSCRIPT_REL}\t#4\tword\t*\n"
                    f"{TRANSCRIPT_REL}\tonset=3.1\tword\tZorblat\n")
    names = private / "names.txt"
    names.write_text("Zorblat\nQuuxville\n")
    return spec, names


def _argv(cdir, stage, private, spec, names, *extra):
    return ["--config-dir", str(cdir), "export", str(stage),
            "--offsets-out", str(private / "offsets.tsv"),
            "--pronoun-report", str(private / "pronouns.tsv"),
            "--redactions", str(spec), "--names-file", str(names), *extra]


def test_dry_run_writes_nothing_and_sweeps_the_planned_result(tmp_path, capsys):
    cfg, work, stage, private = make_env(tmp_path)
    spec, names = _inputs(private)
    cdir = write_config(tmp_path, cfg)
    before = {p: snapshot(p) for p in (work, stage, private)}
    rc = release_export.main(_argv(cdir, stage, private, spec, names, "--dry-run"))
    out = capsys.readouterr().out
    assert {p: snapshot(p) for p in (work, stage, private)} == before
    assert rc == 0, out                    # the planned result would sweep clean
    assert "[dry-run] would rewrite 5 staged file(s)" in out
    assert "sub-01/sub-01_sessions.tsv" in out
    assert not (private / "offsets.tsv").exists()
    assert not (private / "pronouns.tsv").exists()


def test_dry_run_without_redactions_reports_findings(tmp_path, capsys):
    cfg, work, stage, private = make_env(tmp_path)
    _, names = _inputs(private)
    before = snapshot(stage)
    code, findings = run_export(stage, cfg, ExportOptions(epoch=EPOCH, names_file=names,
                                                          dry_run=True))
    assert code == 1
    assert {f.match for f in findings if f.kind == "name"} == {"Zorblat", "Quuxville"}
    assert not [f for f in findings if f.kind == "date"]      # shift is planned
    assert "NO SPEC GIVEN" in capsys.readouterr().out
    assert snapshot(stage) == before


def test_full_export_transforms_stage_only(tmp_path, capsys):
    cfg, work, stage, private = make_env(tmp_path)
    spec, names = _inputs(private)
    cdir = write_config(tmp_path, cfg)
    work_before = snapshot(work)
    rc = release_export.main(_argv(cdir, stage, private, spec, names))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert {k: v[0] for k, v in snapshot(work).items()} == \
        {k: v[0] for k, v in work_before.items()}
    assert "1900-01-01" in (stage / "sub-01/sub-01_sessions.tsv").read_text()
    assert "They like" in (stage / "phenotype/final_debriefing.tsv").read_text()
    assert "Zorblat" not in (stage / TRANSCRIPT_REL).read_text()
    assert (private / "offsets.tsv").exists() and (private / "pronouns.tsv").exists()
    assert "4/4 sweep: 0 finding(s)" in out


def test_failed_plan_writes_nothing(tmp_path, capsys):
    cfg, work, stage, private = make_env(tmp_path)
    spec, names = _inputs(private)
    spec.write_text("file\trow\tcolumn\tspan\n"
                    f"{TRANSCRIPT_REL}\t#4\tword\tNotThere\n")
    cdir = write_config(tmp_path, cfg)
    before = {p: snapshot(p) for p in (stage, private)}
    rc = release_export.main(_argv(cdir, stage, private, spec, names))
    assert rc == 2
    assert "span not found" in capsys.readouterr().err
    assert {p: snapshot(p) for p in (stage, private)} == before


def test_sweep_findings_block_every_write(tmp_path, capsys):
    """A real export whose planned result still has findings writes nothing:
    neither the private outputs nor the staged files."""
    cfg, work, stage, private = make_env(tmp_path)
    _, names = _inputs(private)                      # names present, no redaction
    before = {p: snapshot(p) for p in (stage, private)}
    code, findings = run_export(stage, cfg, ExportOptions(
        epoch=EPOCH, offsets_out=private / "o.tsv", pronoun_report=private / "p.tsv",
        names_file=names))
    out = capsys.readouterr().out
    assert code == 1 and findings
    assert "nothing written" in out
    assert {p: snapshot(p) for p in (stage, private)} == before


def test_commit_with_findings_writes_and_still_reports(tmp_path, capsys):
    cfg, work, stage, private = make_env(tmp_path)
    _, names = _inputs(private)
    code, findings = run_export(stage, cfg, ExportOptions(
        epoch=EPOCH, offsets_out=private / "o.tsv", pronoun_report=private / "p.tsv",
        names_file=names, commit_with_findings=True))
    assert code == 1 and findings
    assert "1900-01-01" in (stage / "sub-01/sub-01_sessions.tsv").read_text()
    assert (private / "o.tsv").exists()


def test_cli_commit_with_findings_flag(tmp_path, capsys):
    cfg, work, stage, private = make_env(tmp_path)
    _, names = _inputs(private)
    cdir = write_config(tmp_path, cfg)
    argv = ["--config-dir", str(cdir), "export", str(stage),
            "--offsets-out", str(private / "o.tsv"),
            "--pronoun-report", str(private / "p.tsv"), "--names-file", str(names)]
    before = snapshot(stage)
    assert release_export.main(argv) == 1
    assert snapshot(stage) == before
    assert release_export.main(argv + ["--commit-with-findings"]) == 1
    assert snapshot(stage) != before


@pytest.mark.parametrize("report_name", ["o.tsv", "o.tsv/p.tsv"])
def test_colliding_private_outputs_are_refused_up_front(tmp_path, report_name):
    """Same path, or one inside the other: refused before anything is written."""
    cfg, work, stage, private = make_env(tmp_path)
    before = {p: snapshot(p) for p in (stage, private)}
    with pytest.raises(StageError, match="collide"):
        run_export(stage, cfg, ExportOptions(
            epoch=EPOCH, offsets_out=private / "o.tsv",
            pronoun_report=private / report_name))
    assert {p: snapshot(p) for p in (stage, private)} == before


def test_export_refuses_the_bids_root(tmp_path, capsys):
    cfg, work, stage, private = make_env(tmp_path)
    spec, names = _inputs(private)
    cdir = write_config(tmp_path, cfg)
    before = snapshot(work)
    assert release_export.main(_argv(cdir, work, private, spec, names)) == 2
    assert "BIDS root" in capsys.readouterr().err
    assert snapshot(work) == before


def test_rerun_after_export_no_ops_the_shift(tmp_path, capsys):
    cfg, work, stage, private = make_env(tmp_path)
    spec, names = _inputs(private)
    run_export(stage, cfg, ExportOptions(epoch=EPOCH, offsets_out=private / "o.tsv",
                                         pronoun_report=private / "p.tsv",
                                         redaction_spec=spec, names_file=names))
    capsys.readouterr()
    code, findings = run_export(stage, cfg, ExportOptions(
        epoch=EPOCH, offsets_out=private / "o2.tsv", pronoun_report=private / "p2.tsv",
        names_file=names))
    out = capsys.readouterr().out
    assert code == 0 and "ALREADY SHIFTED" in out
    assert not (private / "o2.tsv").exists()          # nothing new to record


def test_cli_subcommands_smoke(tmp_path, capsys):
    cfg, work, stage, private = make_env(tmp_path)
    spec, names = _inputs(private)
    cdir = write_config(tmp_path, cfg)
    pre = ["--config-dir", str(cdir)]
    assert release_export.main(pre + ["flag", str(stage), "--out",
                                      str(private / "flags.tsv")]) == 0
    assert "ADVISORY ONLY" in capsys.readouterr().err
    assert (private / "flags.tsv").read_text().startswith("# ADVISORY ONLY")
    before = snapshot(stage)
    assert release_export.main(pre + ["pronouns", str(stage), "--dry-run"]) == 0
    assert release_export.main(pre + ["redact", str(stage), "--spec", str(spec),
                                      "--dry-run"]) == 0
    assert snapshot(stage) == before
    assert release_export.main(pre + ["pronouns", str(stage), "--report",
                                      str(private / "p.tsv")]) == 0
    assert release_export.main(pre + ["redact", str(stage), "--spec", str(spec)]) == 0
    assert release_export.main(pre + ["shift", str(stage), "--offsets-out",
                                      str(private / "o.tsv")]) == 0
    assert release_export.main(pre + ["sweep", str(stage), "--names-file",
                                      str(names)]) == 0
    # writing subcommands refuse the working tree
    assert release_export.main(pre + ["pronouns", str(work), "--dry-run"]) == 2
