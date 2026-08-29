"""DICOM resolution must follow the sibling source tree, not the BIDS root.

Raw data moved out of ``<bids_root>/sourcedata`` to the sibling
``mmmsourcedata`` so the BIDS dataset can be shared without exposing PII.
Three call sites resolved DICOMs under the BIDS root and silently found
nothing after that migration; these pin the corrected resolution.
"""

from pathlib import Path

import pytest

from src.python.dcm2bids_config.cli import _resolve_dicom_dir
from src.python.core.config import load_config


def test_resolve_dicom_dir_uses_source_root():
    source_root = Path("/some/mmmsourcedata")
    got = _resolve_dicom_dir(source_root, "sub-06", "ses-06")
    assert got == source_root / "sub-06" / "ses-06" / "dicom"


def test_resolve_dicom_dir_does_not_nest_under_bids_root():
    """The old form inserted a literal 'sourcedata' component. It must not."""
    got = _resolve_dicom_dir(Path("/some/mmmsourcedata"), "sub-06", "ses-06")
    assert "sourcedata" not in got.relative_to("/some").parts[1:]


@pytest.mark.requires_dataset
def test_configured_source_dir_exists_and_holds_dicoms():
    """The configured source_dir must be a real tree with per-subject DICOMs.

    Guards the migration itself: a config still pointing at the retired
    in-tree sourcedata/ fails here rather than at conversion time.

    Reads the real tree, so it is dataset-tier: off-cluster it skips
    rather than reporting the absent mount as a config error.
    """
    cfg = load_config()
    source_root = Path(cfg["paths"]["source_dir"])
    assert source_root.is_dir(), f"source_dir does not exist: {source_root}"
    subjects = sorted(source_root.glob("sub-*"))
    assert subjects, f"no sub-* under {source_root}"
    assert any(
        (s / ses.name / "dicom").is_dir()
        for s in subjects
        for ses in sorted(s.glob("ses-*"))
    ), f"no <subject>/<session>/dicom under {source_root}"


# ---------------------------------------------------------------------------
# Criteria must match what the scanner actually emits, for both cohorts.
# dcm2bids matches with fnmatch: an unwildcarded string is an exact match.
# ---------------------------------------------------------------------------

from src.python.dcm2bids_config.config_builder import (  # noqa: E402
    _build_bold_description,
    _build_fmap_description_seriesnumber,
)
from src.python.dcm2bids_config.session_defs import TaskDef  # noqa: E402


def test_bold_criteria_do_not_require_multiband_field():
    """sub-06/07's DICOM export omits MultibandAccelerationFactor entirely.

    Requiring it matched zero BOLD series while the job still exited 0.
    """
    task = TaskDef("TBencoding", "cued_recall_encoding_run{n}", "encoding", runs=3)
    desc = _build_bold_description(task, 1, "sub-06", "ses-06")
    assert "MultibandAccelerationFactor" not in desc["criteria"]


def test_bold_criteria_exclude_the_sbref():
    """SeriesDescription is what separates a BOLD from its SBRef."""
    task = TaskDef("TBencoding", "cued_recall_encoding_run{n}", "encoding", runs=3)
    desc = _build_bold_description(task, 1, "sub-06", "ses-06")
    sd = desc["criteria"]["SeriesDescription"]
    assert sd == "cued_recall_encoding_run1"
    # fnmatch, unwildcarded -> exact; the SBRef's description must not match it
    import fnmatch
    assert not fnmatch.fnmatch("cued_recall_encoding_run1_SBRef", sd)


def test_fmap_description_matches_every_observed_cohort_spelling():
    """The fieldmap suffix is not stable; the direction anchor plus SeriesNumber is."""
    import fnmatch
    ap = _build_fmap_description_seriesnumber(
        "AP", 5, "encoding", [], "sub-06", "ses-06"
    )
    pattern = ap["criteria"]["SeriesDescription"]
    for observed in ("se_epi_ap",              # sub-03/04/05
                     "se_epi_ap_encoding",     # sub-06 ses-06
                     "se_epi_ap encoding",     # sub-06 ses-02 (space!)
                     "se_epi_ap retrieval"):
        assert fnmatch.fnmatch(observed, pattern), observed
    # ...but never the opposite phase-encode direction
    for wrong in ("se_epi_pa", "se_epi_pa_encoding"):
        assert not fnmatch.fnmatch(wrong, pattern), wrong
    assert ap["criteria"]["SeriesNumber"] == "5"
