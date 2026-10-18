"""release.sweep: leftover-identifier scan."""

import json
import sys
from pathlib import Path

import pytest

from release import dates, pronouns, redact, sweep
from release.stage import Stage
from tests.release_fixtures import EPOCH, TRANSCRIPT_REL, make_env, write, write_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import release_export  # noqa: E402


def _clean_stage(tmp_path):
    """A stage after shift + pronouns + redaction of the two fictional names."""
    cfg, work, stage, private = make_env(tmp_path)
    s = Stage(stage, cfg)
    dates.plan_date_shift(s, EPOCH)
    pronouns.plan_pronouns(s)
    spec = private / "spec.tsv"
    spec.write_text("file\trow\tcolumn\tspan\n"
                    f"{TRANSCRIPT_REL}\t#4\tword\t*\n"
                    f"{TRANSCRIPT_REL}\t#8\tword\tZorblat\n")
    redact.plan_redactions(s, redact.load_spec(spec))
    s.commit()
    names = private / "names.txt"
    names.write_text("# private list\nZorblat\nquuxville\nsomeuser\n")
    return cfg, stage, private, names


def _kinds(findings):
    return {f.kind for f in findings}


def test_clean_tree_passes(tmp_path):
    cfg, stage, private, names = _clean_stage(tmp_path)
    findings = sweep.sweep(Stage(stage, cfg, writable=False), EPOCH,
                           sweep.load_names(names))
    assert findings == []


def test_unshifted_dates_are_caught_and_shifted_ones_allowed(tmp_path):
    cfg, work, stage, _ = make_env(tmp_path)
    found = sweep.sweep(Stage(stage, cfg, writable=False), EPOCH)
    dates_found = [f for f in found if f.kind == "date"]
    assert {f.path for f in dates_found} == {"sub-01/sub-01_sessions.tsv",
                                             "sub-02/sub-02_sessions.tsv"}
    assert any(f.match == "2032-02-09T10:15:00" for f in dates_found)
    assert all(f.line > 1 for f in dates_found)
    assert sweep.scan_text("x.tsv", "1900-02-03\t1925-12-31", EPOCH, None) == []
    assert _kinds(sweep.scan_text("x.tsv", "1899-12-31", EPOCH, None)) == {"date"}
    assert _kinds(sweep.scan_text("x.json", '"t": "20310304T101500"', EPOCH, None)) == {"date"}
    # not dates: version strings, out-of-range month, a time of day
    assert sweep.scan_text("x", "v1.2.3 2031-13-45 13:45:00 1234-5678", EPOCH, None) == []


def test_absolute_paths_and_emails_are_caught(tmp_path):
    cfg, stage, private, names = _clean_stage(tmp_path)
    write(stage / "code" / "notes.txt",
          "ran from /home/someone/x\nsee /projects/lab/data\ncontact a.person@example.org\n"
          "url https://example.org/home/page is fine\n")
    found = sweep.sweep(Stage(stage, cfg, writable=False), EPOCH)
    got = {(f.path, f.line, f.kind) for f in found}
    assert got == {("code/notes.txt", 1, "path"), ("code/notes.txt", 2, "path"),
                   ("code/notes.txt", 3, "email")}


def test_derivative_dataset_description_links_are_caught(tmp_path):
    cfg, stage, private, names = _clean_stage(tmp_path)
    dd = stage / "derivatives" / "fmriprep" / "dataset_description.json"
    dd.unlink()                       # hardlinked: replace, do not edit in place
    dd.write_text(json.dumps({
        "Name": "fMRIPrep", "DatasetType": "derivative",
        "DatasetLinks": {"raw": "/gpfs/somewhere/raw", "ok": "../.."},
        "SourceDatasets": [{"URL": "file:///scratch/raw"}, {"URL": "https://doi.org/x"}],
    }, indent=4))
    found = sweep.sweep(Stage(stage, cfg, writable=False), EPOCH)
    links = sorted(f.match for f in found if f.kind == "dataset_link")
    assert links == ["DatasetLinks.raw=/gpfs/somewhere/raw",
                     "SourceDatasets[0].URL=file:///scratch/raw"]
    assert all(f.path == "derivatives/fmriprep/dataset_description.json" for f in found)
    assert all(f.line > 0 for f in found)


def test_names_from_a_private_file_are_caught(tmp_path):
    cfg, work, stage, private = make_env(tmp_path)
    names = private / "names.txt"
    names.write_text("Zorblat\nquux ville\n")
    write(stage / "notes.txt", "Spoke with QUUX  VILLE today\n")
    found = [f for f in sweep.sweep(Stage(stage, cfg, writable=False), EPOCH,
                                    sweep.load_names(names)) if f.kind == "name"]
    assert {(f.path, f.match) for f in found} == {
        (TRANSCRIPT_REL, "Zorblat"), ("notes.txt", "QUUX  VILLE")}
    # whole words only
    assert sweep.scan_text("x", "Zorblatian", EPOCH, sweep._names_regex(["Zorblat"])) == []


def test_empty_names_file_is_refused(tmp_path):
    p = tmp_path / "names.txt"
    p.write_text("# nothing\n\n")
    with pytest.raises(ValueError, match="no names"):
        sweep.load_names(p)


def test_cli_exit_codes_and_names_file_inside_stage_refused(tmp_path, capsys):
    cfg, stage, private, names = _clean_stage(tmp_path)
    cdir = write_config(tmp_path, cfg)
    base = ["--config-dir", str(cdir), "sweep", str(stage)]
    assert release_export.main(base + ["--names-file", str(names)]) == 0
    write(stage / "x.txt", "/gpfs/a/b\n")
    assert release_export.main(base + ["--names-file", str(names)]) == 1
    out = capsys.readouterr().out
    assert "x.txt:1\tpath\t/gpfs/a/b" in out
    inside = stage / "names.txt"
    inside.write_text("Zorblat\n")
    assert release_export.main(base + ["--names-file", str(inside)]) == 2
    assert "would ship" in capsys.readouterr().err
