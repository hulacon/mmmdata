"""Tests for core/catalog.py — Contract A catalog query helpers.

Runs against a miniature catalog built in tmp_path with the same
table/view names and columns the real sweep produces (subset).
"""

import duckdb
import pytest

from core import catalog


@pytest.fixture
def mini_catalog(tmp_path):
    db = tmp_path / "catalog.duckdb"
    conn = duckdb.connect(str(db))
    conn.execute("""
        CREATE TABLE files (
            dataset_relpath VARCHAR, sub VARCHAR, ses VARCHAR,
            datatype VARCHAR, task VARCHAR, run INTEGER,
            suffix VARCHAR, ext VARCHAR, path VARCHAR)
    """)
    conn.execute("""
        INSERT INTO files VALUES
        ('.', '03', '04', 'func', 'TBencoding', 1, 'bold', '.nii.gz',
         'sub-03/ses-04/func/x_run-01_bold.nii.gz'),
        ('.', '03', '04', 'func', 'TBencoding', 2, 'bold', '.nii.gz',
         'sub-03/ses-04/func/x_run-02_bold.nii.gz'),
        ('derivatives/fmriprep', '03', '04', 'func', 'TBencoding', 1,
         'bold', '.nii.gz', 'derivatives/fmriprep/x_run-01_bold.nii.gz')
    """)
    conn.execute("""
        CREATE TABLE datasets (
            relpath VARCHAR, kind VARCHAR, name VARCHAR,
            pipeline VARCHAR, pipeline_version VARCHAR)
    """)
    conn.execute("""
        INSERT INTO datasets VALUES
        ('.', 'canonical', 'Mini', NULL, NULL),
        ('derivatives/fmriprep', 'canonical', 'fMRIPrep',
         'fMRIPrep', '24.1.1')
    """)
    conn.execute("""
        CREATE TABLE sessions (
            sub VARCHAR, ses VARCHAR, session_type VARCHAR,
            session_note VARCHAR)
    """)
    conn.execute("""
        INSERT INTO sessions VALUES
        ('03', '04', 'cued_recall', 'fine')
    """)
    conn.execute("""
        CREATE TABLE expected_units (
            dataset_relpath VARCHAR, sub VARCHAR, ses VARCHAR,
            datatype VARCHAR, task VARCHAR, suffix VARCHAR,
            des VARCHAR, recording VARCHAR,
            runs_min INTEGER, runs_max INTEGER)
    """)
    conn.execute("""
        INSERT INTO expected_units VALUES
        ('.', '03', '04', 'func', 'TBencoding', 'bold', NULL, NULL, 3, 3),
        ('.', '03', '04', 'func', 'TBrecall', 'bold', NULL, NULL, 1, 1)
    """)
    conn.execute("""
        CREATE TABLE expectation_exceptions (
            sub VARCHAR, ses VARCHAR, task VARCHAR, datatype VARCHAR,
            suffix VARCHAR, recording VARCHAR, category VARCHAR,
            disposition VARCHAR, status_eligible BOOLEAN,
            description VARCHAR)
    """)
    conn.execute("""
        CREATE TABLE qc_decisions (
            run_key VARCHAR, sub VARCHAR, ses VARCHAR, task VARCHAR,
            run VARCHAR, suffix VARCHAR, decision VARCHAR,
            reviewer VARCHAR, automated BOOLEAN, signed_off BOOLEAN,
            reason VARCHAR)
    """)
    conn.execute("""
        INSERT INTO qc_decisions VALUES
        ('k1', '03', '04', 'TBencoding', '01', 'bold', 'keep',
         '', TRUE, FALSE, 'auto-stub')
    """)
    conn.execute("""
        CREATE TABLE files_supplemental (
            dataset_relpath VARCHAR, path VARCHAR, category VARCHAR,
            sub VARCHAR, ses VARCHAR)
    """)
    conn.execute("""
        INSERT INTO files_supplemental VALUES
        ('.', 'sub-03/ses-04/beh/x.csv', 'dark', '03', '04')
    """)
    # Simplified stand-in for the real resolution view
    conn.execute("""
        CREATE VIEW resolution AS
        SELECT eu.dataset_relpath, eu.sub, eu.ses, eu.datatype, eu.task,
               eu.suffix, eu.recording, eu.runs_min, eu.runs_max,
               COALESCE(o.n, 0) AS observed_n,
               CASE WHEN COALESCE(o.n, 0) >= eu.runs_min
                    THEN 'present' ELSE 'missing' END AS status,
               NULL AS disposition,
               NULL AS exception_notes
        FROM expected_units eu
        LEFT JOIN (
            SELECT dataset_relpath, sub, ses, task, suffix,
                   COUNT(*) AS n
            FROM files GROUP BY ALL
        ) o USING (dataset_relpath, sub, ses, task, suffix)
    """)
    conn.close()
    return db


class TestBareLabel:
    def test_strips_prefix(self):
        assert catalog.bare_label("sub-03", "sub-") == "03"

    def test_leaves_bare(self):
        assert catalog.bare_label("03", "sub-") == "03"

    def test_none(self):
        assert catalog.bare_label(None, "sub-") is None


class TestRunSelect:
    def test_select(self, mini_catalog):
        cols, rows = catalog.run_select(
            mini_catalog, "SELECT COUNT(*) AS n FROM files")
        assert cols == ["n"]
        assert rows == [{"n": 3}]

    def test_params(self, mini_catalog):
        _, rows = catalog.run_select(
            mini_catalog,
            "SELECT COUNT(*) AS n FROM files WHERE sub = ?", ["03"])
        assert rows[0]["n"] == 3

    def test_cte_allowed(self, mini_catalog):
        _, rows = catalog.run_select(
            mini_catalog,
            "WITH t AS (SELECT 1 AS x) SELECT x FROM t")
        assert rows == [{"x": 1}]

    def test_write_rejected(self, mini_catalog):
        with pytest.raises(ValueError, match="read-only"):
            catalog.run_select(mini_catalog, "DELETE FROM files")

    def test_missing_db_names_fix(self, tmp_path):
        # the engine migrated into duckbrain 2026-08-17; the fix the error
        # names must point there, not at the retired mmmdata scripts
        with pytest.raises(FileNotFoundError, match="duckbrain.catalog"):
            catalog.run_select(tmp_path / "nope.duckdb", "SELECT 1")


class TestSessionSummary:
    def test_shapes_and_prefix_tolerance(self, mini_catalog):
        bare = catalog.session_summary(mini_catalog, "03", "04")
        prefixed = catalog.session_summary(mini_catalog, "sub-03", "ses-04")
        assert bare == prefixed
        assert bare["subject"] == "sub-03"
        assert bare["n_raw_files"] == 2
        assert bare["derivatives"][0]["pipeline"] == "fMRIPrep"
        assert bare["supplemental"] == [{"category": "dark", "n_files": 1}]
        assert bare["qc_decisions"][0]["decision"] == "keep"
        assert bare["session_metadata"]["session_type"] == "cued_recall"
        # TBencoding observed 2 < declared 3, TBrecall 0 < 1
        assert bare["resolution_summary"] == {"missing": 2}
        assert len(bare["resolution_issues"]) == 2


class TestResolutionReport:
    def test_default_issues_only(self, mini_catalog):
        report = catalog.resolution_report(mini_catalog)
        assert report["summary"] == {"missing": 2}
        assert report["n_rows"] == 2
        assert all(r["status"] == "missing" for r in report["rows"])

    def test_status_filter(self, mini_catalog):
        report = catalog.resolution_report(
            mini_catalog, statuses=["present"])
        assert report["n_rows"] == 0

    def test_subject_filter_prefix_tolerant(self, mini_catalog):
        report = catalog.resolution_report(mini_catalog,
                                           subjects=["sub-03"])
        assert report["n_rows"] == 2
        report_none = catalog.resolution_report(mini_catalog,
                                                subjects=["99"])
        assert report_none["n_rows"] == 0
        assert report_none["summary"] == {}


class TestLookupExpectations:
    def test_overview(self, mini_catalog):
        r = catalog.lookup_expectations(mini_catalog)
        assert r["available_tasks"] == ["TBencoding", "TBrecall"]
        assert r["available_session_types"] == ["cued_recall"]
        assert r["subjects"] == ["03"]
        assert r["n_expected_units"] == 2

    def test_task(self, mini_catalog):
        r = catalog.lookup_expectations(mini_catalog, task="TBencoding")
        assert r["expected_units"][0]["runs_min"] == 3
        assert r["expected_units"][0]["sessions"] == ["04"]

    def test_unknown_task_lists_known(self, mini_catalog):
        r = catalog.lookup_expectations(mini_catalog, task="nope")
        assert "TBencoding" in r["error"]

    def test_session_type(self, mini_catalog):
        r = catalog.lookup_expectations(mini_catalog,
                                        session_type="cued_recall")
        assert r["sessions"] == [{"ses": "04", "subjects": ["03"]}]
        assert len(r["expected_units"]) == 2
