"""Tests for neuroimaging.qc — core QC analysis library."""

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures — minimal MRIQC / fMRIPrep directory trees
# ---------------------------------------------------------------------------

SAMPLE_BOLD_IQMS = {
    "bids_meta": {"TaskName": "TBencoding"},
    "provenance": {"version": "24.1.0"},
    "fd_mean": 0.12,
    "fd_num": 5,
    "fd_perc": 2.5,
    "tsnr": 30.0,
    "dvars_std": 1.05,
    "dvars_nstd": 55.0,
    "efc": 0.55,
    "fber": 2500.0,
    "snr": 5.0,
    "gcor": 0.01,
    "aqi": 0.03,
    "aor": 0.001,
    "fwhm_avg": 2.2,
    "gsr_x": 0.001,
    "gsr_y": 0.03,
}

SAMPLE_T1W_IQMS = {
    "bids_meta": {"Modality": "MR"},
    "provenance": {"version": "24.1.0"},
    "cnr": 2.5,
    "cjv": 0.35,
    "efc": 0.45,
    "fber": 3000.0,
    "snr_total": 8.0,
    "snr_gm": 6.0,
    "snr_wm": 10.0,
    "qi_1": 0.001,
    "qi_2": 0.02,
    "fwhm_avg": 1.8,
    "inu_range": 0.15,
    "inu_med": 1.0,
    "tpm_overlap_gm": 0.85,
    "tpm_overlap_wm": 0.88,
    "tpm_overlap_csf": 0.75,
    "wm2max": 0.5,
}


@pytest.fixture
def mriqc_dir(tmp_path):
    """Minimal MRIQC derivative tree with 2 subjects, 2 sessions each."""
    mriqc = tmp_path / "mriqc"
    mriqc.mkdir()

    # Subject 01: 2 sessions, each with 2 BOLD runs
    for ses in ("01", "02"):
        func = mriqc / "sub-01" / f"ses-{ses}" / "func"
        func.mkdir(parents=True)
        for run in ("01", "02"):
            iqms = {**SAMPLE_BOLD_IQMS}
            # Vary fd_mean by run to create some spread
            iqms["fd_mean"] = 0.1 + 0.02 * int(run) + 0.01 * int(ses)
            iqms["tsnr"] = 30.0 - int(run) - int(ses)
            p = func / f"sub-01_ses-{ses}_task-encoding_run-{run}_bold.json"
            p.write_text(json.dumps(iqms))

    # Subject 02: 2 sessions, each with 2 BOLD runs, plus one outlier
    for ses in ("01", "02"):
        func = mriqc / "sub-02" / f"ses-{ses}" / "func"
        func.mkdir(parents=True)
        for run in ("01", "02"):
            iqms = {**SAMPLE_BOLD_IQMS}
            iqms["fd_mean"] = 0.11 + 0.02 * int(run) + 0.01 * int(ses)
            iqms["tsnr"] = 29.0 - int(run) - int(ses)
            # Make sub-02 ses-02 run-02 an outlier
            if ses == "02" and run == "02":
                iqms["fd_mean"] = 0.8  # Very high motion
                iqms["tsnr"] = 10.0  # Very low tSNR
            p = func / f"sub-02_ses-{ses}_task-encoding_run-{run}_bold.json"
            p.write_text(json.dumps(iqms))

    # Subject 01: one T1w per session
    for ses in ("01", "02"):
        anat = mriqc / "sub-01" / f"ses-{ses}" / "anat"
        anat.mkdir(parents=True)
        p = anat / f"sub-01_ses-{ses}_acq-MPR_run-01_T1w.json"
        p.write_text(json.dumps(SAMPLE_T1W_IQMS))

    # HTML reports at root
    for name in [
        "sub-01_ses-01_task-encoding_run-01_bold.html",
        "sub-01_ses-01_acq-MPR_run-01_T1w.html",
        "sub-02_ses-01_task-encoding_run-01_bold.html",
    ]:
        (mriqc / name).write_text("<html></html>")

    # A timeseries JSON that should be excluded
    ts = mriqc / "sub-01" / "ses-01" / "func" / "sub-01_ses-01_task-encoding_run-01_timeseries.json"
    ts.write_text("{}")

    return mriqc


@pytest.fixture
def fmriprep_dir(tmp_path):
    """Minimal fMRIPrep derivative tree with confound TSVs."""
    fmriprep = tmp_path / "fmriprep"
    fmriprep.mkdir()

    header = "framewise_displacement\tdvars\tstd_dvars\trmsd\ttrans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\n"

    for sub in ("01", "02"):
        for ses in ("01", "02"):
            func = fmriprep / f"sub-{sub}" / f"ses-{ses}" / "func"
            func.mkdir(parents=True)
            for run in ("01", "02"):
                tsv = func / f"sub-{sub}_ses-{ses}_task-encoding_run-{run}_desc-confounds_timeseries.tsv"
                lines = [header]
                # First volume: NaN FD
                lines.append("n/a\tn/a\tn/a\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\n")
                # 9 more volumes with small motion
                for i in range(9):
                    fd = 0.1 + 0.02 * i
                    if sub == "02" and ses == "02" and run == "02":
                        fd = 0.6 + 0.1 * i  # High motion outlier
                    dvars = 20.0 + i
                    lines.append(f"{fd}\t{dvars}\t1.0\t{fd*0.8}\t0.01\t0.02\t0.01\t0.001\t0.002\t0.001\n")
                tsv.write_text("".join(lines))

    # HTML reports
    for name in ["sub-01_anat.html", "sub-01_ses-01_func.html"]:
        (fmriprep / name).write_text("<html></html>")

    return fmriprep


@pytest.fixture
def bids_root(tmp_path):
    """Minimal BIDS root with BOLD NIfTI stubs."""
    bids = tmp_path / "bids"
    bids.mkdir()
    for sub in ("01", "02"):
        for ses in ("01", "02"):
            func = bids / f"sub-{sub}" / f"ses-{ses}" / "func"
            func.mkdir(parents=True)
            for run in ("01", "02"):
                (func / f"sub-{sub}_ses-{ses}_task-encoding_run-{run}_bold.nii.gz").write_text("")
            anat = bids / f"sub-{sub}" / f"ses-{ses}" / "anat"
            anat.mkdir(parents=True)
            (anat / f"sub-{sub}_ses-{ses}_T1w.nii.gz").write_text("")
    return bids


# ---------------------------------------------------------------------------
# Tests — parse_bids_entities
# ---------------------------------------------------------------------------

class TestParseBidsEntities:

    def test_bold_with_all_entities(self):
        from neuroimaging.qc import parse_bids_entities
        r = parse_bids_entities("sub-03_ses-04_task-TBencoding_run-01_bold")
        assert r["subject"] == "03"
        assert r["session"] == "04"
        assert r["task"] == "TBencoding"
        assert r["run"] == "01"
        assert r["suffix"] == "bold"

    def test_t1w_with_acq(self):
        from neuroimaging.qc import parse_bids_entities
        r = parse_bids_entities("sub-03_ses-01_acq-MPR_run-01_T1w.json")
        assert r["subject"] == "03"
        assert r["acq"] == "MPR"
        assert r["suffix"] == "T1w"
        assert r["task"] is None

    def test_dwi_with_dir(self):
        from neuroimaging.qc import parse_bids_entities
        r = parse_bids_entities("sub-03_ses-01_dir-AP_dwi")
        assert r["dir"] == "AP"
        assert r["suffix"] == "dwi"

    def test_no_match(self):
        from neuroimaging.qc import parse_bids_entities
        r = parse_bids_entities("random_file.txt")
        assert all(v is None for v in r.values())


# ---------------------------------------------------------------------------
# Tests — collect_mriqc_jsons
# ---------------------------------------------------------------------------

class TestCollectMriqcJsons:

    def test_collects_bold(self, mriqc_dir):
        from neuroimaging.qc import collect_mriqc_jsons
        files = collect_mriqc_jsons(mriqc_dir, "bold")
        assert len(files) == 8  # 2 subjects * 2 sessions * 2 runs

    def test_filter_by_subject(self, mriqc_dir):
        from neuroimaging.qc import collect_mriqc_jsons
        files = collect_mriqc_jsons(mriqc_dir, "bold", subject="01")
        assert len(files) == 4

    def test_filter_by_session(self, mriqc_dir):
        from neuroimaging.qc import collect_mriqc_jsons
        files = collect_mriqc_jsons(mriqc_dir, "bold", subject="01", session="01")
        assert len(files) == 2

    def test_collects_t1w(self, mriqc_dir):
        from neuroimaging.qc import collect_mriqc_jsons
        files = collect_mriqc_jsons(mriqc_dir, "T1w")
        assert len(files) == 2

    def test_excludes_timeseries(self, mriqc_dir):
        from neuroimaging.qc import collect_mriqc_jsons
        files = collect_mriqc_jsons(mriqc_dir, "bold", subject="01", session="01")
        names = [f.name for f in files]
        assert not any("timeseries" in n for n in names)

    def test_invalid_modality(self, mriqc_dir):
        from neuroimaging.qc import collect_mriqc_jsons
        with pytest.raises(ValueError, match="Unknown modality"):
            collect_mriqc_jsons(mriqc_dir, "invalid")


# ---------------------------------------------------------------------------
# Tests — load_iqms
# ---------------------------------------------------------------------------

class TestLoadIqms:

    def test_loads_all(self, mriqc_dir):
        from neuroimaging.qc import load_iqms
        p = mriqc_dir / "sub-01" / "ses-01" / "func" / "sub-01_ses-01_task-encoding_run-01_bold.json"
        result = load_iqms(p)
        assert "entities" in result
        assert "iqms" in result
        assert result["entities"]["subject"] == "01"
        assert "bids_meta" not in result["iqms"]
        assert "provenance" not in result["iqms"]
        assert "fd_mean" in result["iqms"]

    def test_key_metrics_filter(self, mriqc_dir):
        from neuroimaging.qc import load_iqms
        p = mriqc_dir / "sub-01" / "ses-01" / "func" / "sub-01_ses-01_task-encoding_run-01_bold.json"
        result = load_iqms(p, key_metrics=["fd_mean", "tsnr", "nonexistent"])
        assert set(result["iqms"].keys()) == {"fd_mean", "tsnr", "nonexistent"}
        assert result["iqms"]["nonexistent"] is None


# ---------------------------------------------------------------------------
# Tests — list_reports
# ---------------------------------------------------------------------------

class TestListReports:

    def test_mriqc_reports(self, mriqc_dir):
        from neuroimaging.qc import list_reports
        result = list_reports(mriqc_dir=mriqc_dir)
        assert result["mriqc"]["total"] == 3
        assert "01" in result["mriqc"]["by_subject"]

    def test_fmriprep_reports(self, fmriprep_dir):
        from neuroimaging.qc import list_reports
        result = list_reports(fmriprep_dir=fmriprep_dir)
        assert result["fmriprep"]["total"] == 2

    def test_both_pipelines(self, mriqc_dir, fmriprep_dir):
        from neuroimaging.qc import list_reports
        result = list_reports(mriqc_dir=mriqc_dir, fmriprep_dir=fmriprep_dir)
        assert "mriqc" in result
        assert "fmriprep" in result

    def test_missing_dir(self, tmp_path):
        from neuroimaging.qc import list_reports
        result = list_reports(mriqc_dir=tmp_path / "nonexistent")
        assert "error" in result["mriqc"]


# ---------------------------------------------------------------------------
# Tests — get_iqm_table
# ---------------------------------------------------------------------------

class TestGetIqmTable:

    def test_returns_rows(self, mriqc_dir):
        from neuroimaging.qc import get_iqm_table
        table = get_iqm_table(mriqc_dir, "bold")
        assert len(table) == 8
        assert "subject" in table[0]
        assert "fd_mean" in table[0]

    def test_filters_by_subject(self, mriqc_dir):
        from neuroimaging.qc import get_iqm_table
        table = get_iqm_table(mriqc_dir, "bold", subject="01")
        assert len(table) == 4
        assert all(r["subject"] == "01" for r in table)

    def test_custom_metrics(self, mriqc_dir):
        from neuroimaging.qc import get_iqm_table
        table = get_iqm_table(mriqc_dir, "bold", metrics=["fd_mean", "tsnr"])
        # Should only have entity fields + requested metrics
        iqm_keys = {k for k in table[0] if k not in
                    ("subject", "session", "task", "acq", "dir", "run", "suffix")}
        assert iqm_keys == {"fd_mean", "tsnr"}


# ---------------------------------------------------------------------------
# Tests — aggregate_iqms
# ---------------------------------------------------------------------------

class TestAggregateIqms:

    def test_group_by_subject(self, mriqc_dir):
        from neuroimaging.qc import aggregate_iqms
        result = aggregate_iqms(mriqc_dir, "bold", group_by="subject")
        assert "01" in result["groups"]
        assert "02" in result["groups"]
        assert result["groups"]["01"]["n_runs"] == 4

    def test_group_by_session(self, mriqc_dir):
        from neuroimaging.qc import aggregate_iqms
        result = aggregate_iqms(mriqc_dir, "bold", group_by="session", subject="01")
        assert "01" in result["groups"]
        assert "02" in result["groups"]

    def test_group_by_global(self, mriqc_dir):
        from neuroimaging.qc import aggregate_iqms
        result = aggregate_iqms(mriqc_dir, "bold", group_by="global")
        assert "all" in result["groups"]
        assert result["groups"]["all"]["n_runs"] == 8

    def test_stats_structure(self, mriqc_dir):
        from neuroimaging.qc import aggregate_iqms
        result = aggregate_iqms(mriqc_dir, "bold", group_by="global")
        stats = result["groups"]["all"]
        assert "mean" in stats["fd_mean"]
        assert "std" in stats["fd_mean"]
        assert "50%" in stats["fd_mean"]

    def test_empty_result(self, mriqc_dir):
        from neuroimaging.qc import aggregate_iqms
        result = aggregate_iqms(mriqc_dir, "bold", subject="99")
        assert result["groups"] == {}


# ---------------------------------------------------------------------------
# Tests — detect_outliers
# ---------------------------------------------------------------------------

class TestDetectOutliers:

    def test_global_finds_outlier(self, mriqc_dir):
        from neuroimaging.qc import detect_outliers
        result = detect_outliers(mriqc_dir, "bold", scope="global")
        assert result["n_runs_checked"] == 8
        # The extreme sub-02/ses-02/run-02 should be flagged
        outlier_keys = [
            (o["subject"], o["session"], o["run"]) for o in result["outliers"]
        ]
        assert ("02", "02", "02") in outlier_keys

    def test_within_subject_finds_outlier(self, mriqc_dir):
        from neuroimaging.qc import detect_outliers
        result = detect_outliers(mriqc_dir, "bold", scope="within_subject")
        outlier_keys = [
            (o["subject"], o["session"], o["run"]) for o in result["outliers"]
        ]
        assert ("02", "02", "02") in outlier_keys

    def test_outlier_has_flagged_metrics(self, mriqc_dir):
        from neuroimaging.qc import detect_outliers
        result = detect_outliers(mriqc_dir, "bold", scope="global")
        outlier = [o for o in result["outliers"]
                   if o["subject"] == "02" and o["session"] == "02" and o["run"] == "02"][0]
        assert "fd_mean" in outlier["flagged_metrics"]
        assert outlier["flagged_metrics"]["fd_mean"]["direction"] == "high"

    def test_summary_by_subject(self, mriqc_dir):
        from neuroimaging.qc import detect_outliers
        result = detect_outliers(mriqc_dir, "bold", scope="global")
        assert "01" in result["summary_by_subject"]
        assert "02" in result["summary_by_subject"]
        assert result["summary_by_subject"]["02"]["n_outlier_runs"] >= 1

    def test_strict_multiplier(self, mriqc_dir):
        from neuroimaging.qc import detect_outliers
        # With a very high multiplier, fewer outliers
        strict = detect_outliers(mriqc_dir, "bold", scope="global", iqr_multiplier=10.0)
        lenient = detect_outliers(mriqc_dir, "bold", scope="global", iqr_multiplier=1.0)
        assert strict["n_outlier_runs"] <= lenient["n_outlier_runs"]

    def test_empty_result(self, mriqc_dir):
        from neuroimaging.qc import detect_outliers
        result = detect_outliers(mriqc_dir, "bold", subject="99")
        assert result["n_runs_checked"] == 0
        assert result["outliers"] == []


# ---------------------------------------------------------------------------
# Tests — summarize_motion
# ---------------------------------------------------------------------------

class TestSummarizeMotion:

    def test_returns_runs(self, fmriprep_dir):
        from neuroimaging.qc import summarize_motion
        result = summarize_motion(fmriprep_dir)
        assert result["n_runs"] == 8
        assert len(result["runs"]) == 8

    def test_motion_fields(self, fmriprep_dir):
        from neuroimaging.qc import summarize_motion
        result = summarize_motion(fmriprep_dir, subject="01", session="01")
        run = result["runs"][0]
        assert "mean_fd" in run
        assert "median_fd" in run
        assert "max_fd" in run
        assert "n_high_motion" in run
        assert "pct_high_motion" in run

    def test_fd_threshold(self, fmriprep_dir):
        from neuroimaging.qc import summarize_motion
        # sub-02 ses-02 run-02 has high FD values (0.6-1.5)
        result = summarize_motion(fmriprep_dir, subject="02", session="02")
        high_run = [r for r in result["runs"] if r["run"] == "02"][0]
        assert high_run["n_high_motion"] > 0

    def test_summary_by_subject(self, fmriprep_dir):
        from neuroimaging.qc import summarize_motion
        result = summarize_motion(fmriprep_dir)
        assert "01" in result["summary_by_subject"]
        assert result["summary_by_subject"]["01"]["n_runs"] == 4


# ---------------------------------------------------------------------------
# Tests — processing_status
# ---------------------------------------------------------------------------

class TestProcessingStatus:

    def test_totals(self, bids_root, mriqc_dir, fmriprep_dir):
        from neuroimaging.qc import processing_status
        result = processing_status(bids_root, mriqc_dir, fmriprep_dir)
        assert result["totals"]["n_bids_bold"] == 8
        assert result["totals"]["n_mriqc_bold"] == 8
        assert result["totals"]["n_fmriprep_bold"] == 8
        assert result["totals"]["pct_mriqc_complete"] == 100.0

    def test_filter_subject(self, bids_root, mriqc_dir, fmriprep_dir):
        from neuroimaging.qc import processing_status
        result = processing_status(bids_root, mriqc_dir, fmriprep_dir, subject="01")
        assert "01" in result["subjects"]
        assert "02" not in result["subjects"]

    def test_mriqc_only(self, bids_root, mriqc_dir, fmriprep_dir):
        from neuroimaging.qc import processing_status
        result = processing_status(bids_root, mriqc_dir, fmriprep_dir, pipeline="mriqc")
        assert "n_mriqc_bold" in result["totals"]
        assert "n_fmriprep_bold" not in result["totals"]


# ---------------------------------------------------------------------------
# Tests — run_details
# ---------------------------------------------------------------------------

class TestRunDetails:

    def test_bold_with_both(self, mriqc_dir, fmriprep_dir):
        from neuroimaging.qc import run_details
        result = run_details(mriqc_dir, fmriprep_dir, "01", "01", "encoding", run="01")
        assert result["mriqc_iqms"] is not None
        assert "fd_mean" in result["mriqc_iqms"]
        assert result["fmriprep_motion"] is not None
        assert "mean_fd" in result["fmriprep_motion"]

    def test_missing_mriqc(self, mriqc_dir, fmriprep_dir):
        from neuroimaging.qc import run_details
        result = run_details(mriqc_dir, fmriprep_dir, "01", "01", "nonexistent", run="01")
        assert result["mriqc_iqms"] is None

    def test_t1w_no_motion(self, mriqc_dir, fmriprep_dir):
        from neuroimaging.qc import run_details
        result = run_details(mriqc_dir, fmriprep_dir, "01", "01", "encoding",
                             run="01", suffix="T1w")
        # T1w doesn't have fMRIPrep motion
        assert "fmriprep_motion" not in result
