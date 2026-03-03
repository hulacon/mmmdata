"""Pytest configuration and shared fixtures for tests."""

import json

import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Sample IQM data for MRIQC test fixtures
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


# ---------------------------------------------------------------------------
# Shared QC test fixtures
# ---------------------------------------------------------------------------

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


@pytest.fixture
def sample_bids_dir(tmp_path):
    """
    Create a minimal BIDS directory structure for testing.

    Returns:
        Path: Path to the temporary BIDS directory
    """
    bids_dir = tmp_path / "bids_dataset"
    bids_dir.mkdir()

    # Create dataset_description.json
    dataset_desc = bids_dir / "dataset_description.json"
    dataset_desc.write_text('''{
    "Name": "Test Dataset",
    "BIDSVersion": "1.6.0"
}''')

    # Create participants.tsv
    participants_tsv = bids_dir / "participants.tsv"
    participants_tsv.write_text('''participant_id\tage\tsex
sub-01\t25\tM
sub-02\t30\tF
''')

    return bids_dir


@pytest.fixture
def sample_config_dir(tmp_path):
    """
    Create a sample configuration directory with base.toml.

    Returns:
        Path: Path to the temporary config directory
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create base.toml
    base_toml = config_dir / "base.toml"
    base_toml.write_text('''[paths]
bids_project_dir = "/test/bids"
code_root = "/test/code"
singularity_dir = "/test/singularity"
venv_dir = "/test/venv"
output_dir = "/test/output"

[slurm]
partition = "compute"
time = "12:00:00"
memory = "16G"
cpus = "4"
email = "test@example.com"
''')

    return config_dir
