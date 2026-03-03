"""Tests for neuroimaging.qc_dashboard — dashboard generator and decision tracking.

Fixtures (mriqc_dir, fmriprep_dir) are defined in conftest.py
and shared with test_qc.py.
"""

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures specific to dashboard tests
# ---------------------------------------------------------------------------

@pytest.fixture
def decisions_dir(tmp_path):
    """Empty decisions directory."""
    d = tmp_path / "qc_decisions"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Tests — _build_run_key
# ---------------------------------------------------------------------------

class TestBuildRunKey:

    def test_with_all_entities(self):
        from neuroimaging.qc_dashboard import _build_run_key
        key = _build_run_key("03", "04", "TBencoding", "01", "bold")
        assert key == "sub-03_ses-04_task-TBencoding_run-01_bold"

    def test_without_run(self):
        from neuroimaging.qc_dashboard import _build_run_key
        key = _build_run_key("03", "04", "TBencoding", None, "bold")
        assert key == "sub-03_ses-04_task-TBencoding_bold"

    def test_t1w(self):
        from neuroimaging.qc_dashboard import _build_run_key
        key = _build_run_key("01", "01", "encoding", "01", "T1w")
        assert key == "sub-01_ses-01_task-encoding_run-01_T1w"


# ---------------------------------------------------------------------------
# Tests — save_decision
# ---------------------------------------------------------------------------

class TestSaveDecision:

    def test_creates_file(self, decisions_dir):
        from neuroimaging.qc_dashboard import save_decision
        record = save_decision(
            decisions_dir, "01", "01", "encoding", "01",
            "keep", "Looks good", "tester",
        )
        assert record["decision"] == "keep"
        assert record["reason"] == "Looks good"
        assert record["reviewer"] == "tester"
        assert "timestamp" in record

        # File exists
        expected = decisions_dir / "sub-01" / "sub-01_ses-01_task-encoding_run-01_bold_decision.json"
        assert expected.exists()

        data = json.loads(expected.read_text())
        assert data["run_key"] == "sub-01_ses-01_task-encoding_run-01_bold"
        assert len(data["decisions"]) == 1

    def test_appends_to_existing(self, decisions_dir):
        from neuroimaging.qc_dashboard import save_decision
        save_decision(decisions_dir, "01", "01", "encoding", "01",
                      "investigate", "Borderline", "agent")
        save_decision(decisions_dir, "01", "01", "encoding", "01",
                      "keep", "Reviewed, ok", "bhutch")

        path = decisions_dir / "sub-01" / "sub-01_ses-01_task-encoding_run-01_bold_decision.json"
        data = json.loads(path.read_text())
        assert len(data["decisions"]) == 2
        assert data["decisions"][0]["decision"] == "investigate"
        assert data["decisions"][1]["decision"] == "keep"

    def test_validates_decision_enum(self, decisions_dir):
        from neuroimaging.qc_dashboard import save_decision
        with pytest.raises(ValueError, match="Invalid decision"):
            save_decision(decisions_dir, "01", "01", "encoding", "01",
                          "bad_value", "reason", "tester")

    def test_creates_subject_directory(self, decisions_dir):
        from neuroimaging.qc_dashboard import save_decision
        save_decision(decisions_dir, "99", "01", "rest", None,
                      "exclude", "Bad data", "tester")
        assert (decisions_dir / "sub-99").is_dir()

    def test_no_run_entity(self, decisions_dir):
        from neuroimaging.qc_dashboard import save_decision
        save_decision(decisions_dir, "01", "01", "rest", None,
                      "keep", "Fine", "tester")
        path = decisions_dir / "sub-01" / "sub-01_ses-01_task-rest_bold_decision.json"
        assert path.exists()

    def test_t1w_suffix(self, decisions_dir):
        from neuroimaging.qc_dashboard import save_decision
        save_decision(decisions_dir, "01", "01", "encoding", "01",
                      "keep", "ok", "tester", suffix="T1w")
        path = decisions_dir / "sub-01" / "sub-01_ses-01_task-encoding_run-01_T1w_decision.json"
        assert path.exists()


# ---------------------------------------------------------------------------
# Tests — load_decisions
# ---------------------------------------------------------------------------

class TestLoadDecisions:

    def test_load_empty_dir(self, decisions_dir):
        from neuroimaging.qc_dashboard import load_decisions
        result = load_decisions(decisions_dir)
        assert result == {}

    def test_load_nonexistent_dir(self, tmp_path):
        from neuroimaging.qc_dashboard import load_decisions
        result = load_decisions(tmp_path / "nonexistent")
        assert result == {}

    def test_load_single_subject(self, decisions_dir):
        from neuroimaging.qc_dashboard import save_decision, load_decisions
        save_decision(decisions_dir, "01", "01", "encoding", "01",
                      "keep", "ok", "tester")
        save_decision(decisions_dir, "02", "01", "encoding", "01",
                      "exclude", "bad", "tester")

        result = load_decisions(decisions_dir, subject="01")
        assert len(result) == 1
        assert "sub-01_ses-01_task-encoding_run-01_bold" in result

    def test_load_all_subjects(self, decisions_dir):
        from neuroimaging.qc_dashboard import save_decision, load_decisions
        save_decision(decisions_dir, "01", "01", "encoding", "01",
                      "keep", "ok", "tester")
        save_decision(decisions_dir, "02", "01", "encoding", "01",
                      "exclude", "bad", "tester")

        result = load_decisions(decisions_dir)
        assert len(result) == 2

    def test_latest_is_most_recent(self, decisions_dir):
        from neuroimaging.qc_dashboard import save_decision, load_decisions
        save_decision(decisions_dir, "01", "01", "encoding", "01",
                      "investigate", "maybe bad", "agent")
        save_decision(decisions_dir, "01", "01", "encoding", "01",
                      "keep", "actually fine", "bhutch")

        result = load_decisions(decisions_dir)
        key = "sub-01_ses-01_task-encoding_run-01_bold"
        assert result[key]["latest"]["decision"] == "keep"
        assert len(result[key]["history"]) == 2


# ---------------------------------------------------------------------------
# Tests — generate_dashboard
# ---------------------------------------------------------------------------

class TestGenerateDashboard:

    def test_creates_html_file(self, mriqc_dir, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard
        save_path = tmp_path / "dashboard.html"
        result = generate_dashboard(mriqc_dir, save_path=save_path)
        assert Path(result).exists()
        assert result == str(save_path.resolve())

    def test_save_path_required(self, mriqc_dir):
        from neuroimaging.qc_dashboard import generate_dashboard
        with pytest.raises(ValueError, match="save_path is required"):
            generate_dashboard(mriqc_dir)

    def test_html_contains_plotly_inline(self, mriqc_dir, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard
        path = generate_dashboard(mriqc_dir, save_path=tmp_path / "d.html")
        html = Path(path).read_text()
        # Plotly JS is embedded inline (no CDN dependency)
        assert "Plotly" in html

    def test_html_contains_table(self, mriqc_dir, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard
        path = generate_dashboard(mriqc_dir, save_path=tmp_path / "d.html")
        html = Path(path).read_text()
        assert "<table" in html
        assert "<thead>" in html
        assert "<tbody>" in html

    def test_html_contains_run_data(self, mriqc_dir, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard
        path = generate_dashboard(mriqc_dir, save_path=tmp_path / "d.html")
        html = Path(path).read_text()
        # Subject IDs should appear in the table
        assert ">01<" in html  # subject 01
        assert ">02<" in html  # subject 02

    def test_html_contains_mriqc_links(self, mriqc_dir, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard
        path = generate_dashboard(mriqc_dir, save_path=tmp_path / "d.html")
        html = Path(path).read_text()
        # At least some runs should have report links
        assert "View</a>" in html

    def test_outlier_runs_highlighted(self, mriqc_dir, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard
        path = generate_dashboard(mriqc_dir, save_path=tmp_path / "d.html")
        html = Path(path).read_text()
        assert "row-outlier" in html

    def test_decisions_shown(self, mriqc_dir, decisions_dir, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard, save_decision
        save_decision(decisions_dir, "01", "01", "encoding", "01",
                      "keep", "Good run", "tester")

        path = generate_dashboard(
            mriqc_dir, decisions_dir=decisions_dir,
            save_path=tmp_path / "d.html",
        )
        html = Path(path).read_text()
        assert "KEEP" in html

    def test_single_subject(self, mriqc_dir, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard
        path_all = generate_dashboard(mriqc_dir, save_path=tmp_path / "all.html")
        path_one = generate_dashboard(mriqc_dir, subject="01", save_path=tmp_path / "one.html")
        # Single-subject dashboard should be smaller (fewer rows)
        assert len(Path(path_one).read_text()) < len(Path(path_all).read_text())

    def test_t1w_modality(self, mriqc_dir, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard
        path = generate_dashboard(
            mriqc_dir, modality="T1w", save_path=tmp_path / "d.html",
        )
        html = Path(path).read_text()
        assert "T1w" in html
        # No motion section for T1w
        assert "Motion Overview" not in html

    def test_no_fmriprep(self, mriqc_dir, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard
        # Should work fine without fmriprep data
        path = generate_dashboard(mriqc_dir, save_path=tmp_path / "d.html")
        assert Path(path).exists()

    def test_no_decisions_dir(self, mriqc_dir, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard
        path = generate_dashboard(mriqc_dir, save_path=tmp_path / "d.html")
        html = Path(path).read_text()
        # All should show PENDING
        assert "PENDING" in html

    def test_with_fmriprep(self, mriqc_dir, fmriprep_dir, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard
        path = generate_dashboard(
            mriqc_dir, fmriprep_dir=fmriprep_dir,
            save_path=tmp_path / "d.html",
        )
        html = Path(path).read_text()
        assert "Motion Overview" in html
        assert "mean_fd" in html

    # -- Processing status & subject summary --

    def test_with_bids_root_shows_processing_status(self, mriqc_dir, fmriprep_dir, bids_root, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard
        path = generate_dashboard(
            mriqc_dir, fmriprep_dir=fmriprep_dir,
            save_path=tmp_path / "d.html",
            bids_root=bids_root,
        )
        html = Path(path).read_text()
        assert "Processing Status" in html
        assert "BIDS BOLD Runs" in html
        assert "MRIQC Complete" in html

    def test_without_bids_root_no_processing_status(self, mriqc_dir, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard
        path = generate_dashboard(mriqc_dir, save_path=tmp_path / "d.html")
        html = Path(path).read_text()
        assert "Processing Status" not in html

    def test_subject_summary_section_present(self, mriqc_dir, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard
        path = generate_dashboard(mriqc_dir, save_path=tmp_path / "d.html")
        html = Path(path).read_text()
        assert "Subject Summary" in html

    def test_subject_summary_shows_per_subject_rows(self, mriqc_dir, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard
        path = generate_dashboard(mriqc_dir, save_path=tmp_path / "d.html")
        html = Path(path).read_text()
        assert "sub-01" in html
        assert "sub-02" in html

    def test_processing_status_single_subject(self, mriqc_dir, fmriprep_dir, bids_root, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard
        path = generate_dashboard(
            mriqc_dir, fmriprep_dir=fmriprep_dir,
            subject="01", save_path=tmp_path / "d.html",
            bids_root=bids_root,
        )
        html = Path(path).read_text()
        assert "Processing Status" in html

    def test_backward_compat_no_bids_root(self, mriqc_dir, tmp_path):
        from neuroimaging.qc_dashboard import generate_dashboard
        path = generate_dashboard(mriqc_dir, save_path=tmp_path / "d.html")
        assert Path(path).exists()
        html = Path(path).read_text()
        assert "<table" in html


# ---------------------------------------------------------------------------
# Tests — _build_subject_summary
# ---------------------------------------------------------------------------

class TestBuildSubjectSummary:

    def test_basic_structure(self, mriqc_dir):
        from neuroimaging.qc_dashboard import _build_subject_summary
        from neuroimaging import qc

        outlier_result = qc.detect_outliers(mriqc_dir, "bold")
        subject_summary = _build_subject_summary(outlier_result, None, None, "bold")

        assert len(subject_summary) >= 1
        row = subject_summary[0]
        assert "subject" in row
        assert "n_runs" in row
        assert "n_outliers" in row
        assert "n_reviewed" in row
        assert "n_pending" in row

    def test_with_decisions(self, mriqc_dir, decisions_dir):
        from neuroimaging.qc_dashboard import _build_subject_summary, save_decision, load_decisions
        from neuroimaging import qc

        save_decision(decisions_dir, "01", "01", "encoding", "01", "keep", "ok", "tester")
        save_decision(decisions_dir, "01", "01", "encoding", "02", "exclude", "bad", "tester")
        decisions = load_decisions(decisions_dir)

        outlier_result = qc.detect_outliers(mriqc_dir, "bold")
        subject_summary = _build_subject_summary(
            outlier_result, None, decisions, "bold",
        )

        sub01 = next(r for r in subject_summary if r["subject"] == "01")
        assert sub01["n_reviewed"] == 2
        assert sub01["n_exclude"] == 1

    def test_no_motion_for_anat(self, mriqc_dir):
        from neuroimaging.qc_dashboard import _build_subject_summary
        from neuroimaging import qc

        outlier_result = qc.detect_outliers(mriqc_dir, "T1w")
        subject_summary = _build_subject_summary(outlier_result, None, None, "T1w")

        for row in subject_summary:
            assert row["mean_fd"] is None
            assert row["mean_high_motion_pct"] is None
