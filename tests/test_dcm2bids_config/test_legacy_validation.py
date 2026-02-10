"""Validate new config generation against legacy dcm2bids config files.

These tests compare the output of ``build_config()`` against the existing
hand-crafted configs at ``dcm2bids_configfiles/``.  They serve two purposes:

1. Verify the new system produces equivalent DICOM→BIDS mappings
2. Document every known bug in the legacy configs

Legacy config directory:
    /gpfs/projects/hulacon/shared/mmmdata/code/dcm2bids_configfiles/

To run just these tests::

    pytest tests/test_dcm2bids_config/test_legacy_validation.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.python.dcm2bids_config.config_builder import build_config
from src.python.dcm2bids_config.session_defs import (
    SESSION_SCHEDULE,
    SESSION_TYPES,
    get_session_def,
)

LEGACY_DIR = Path("/gpfs/projects/hulacon/shared/mmmdata/code/dcm2bids_configfiles")


def _skip_if_no_legacy():
    if not LEGACY_DIR.is_dir():
        pytest.skip(f"Legacy config dir not found: {LEGACY_DIR}")


# ---------------------------------------------------------------------------
# Known corrections for legacy config bugs
# ---------------------------------------------------------------------------

# Maps (subject, session) → dict of known issues in legacy config.
# Used to adjust comparisons so tests pass despite legacy bugs.
KNOWN_LEGACY_BUGS: dict[tuple[str, str], dict] = {
    # All _confedit configs use task-encoding instead of task-TBencoding, etc.
    # and are missing TaskName in sidecar_changes.
    ("sub-03", "ses-10"): {
        "task_prefix_missing": True,
        "missing_taskname": True,
        "extra_fmap_groups": True,  # 3 groups instead of 2 (re-entry)
        "note": "Hand-edited; CR prefix, 3 fmap pairs from scanner re-entry",
    },
    ("sub-04", "ses-05"): {
        "task_prefix_missing": True,
        "missing_taskname": True,
        "note": "confdeletemath variant; CR prefix, missing TaskName",
    },
    ("sub-03", "ses-02"): {
        "task_prefix_missing": True,
        "missing_taskname": True,
        "note": "Localizer session; custom task set",
    },
    ("sub-03", "ses-30"): {
        "task_prefix_missing": True,
        "missing_taskname": True,
        "note": "Final session; custom task set, T1w + misc tasks",
    },
}

# Task label corrections: legacy label → correct label
TASK_LABEL_CORRECTIONS = {
    "encoding": "TBencoding",
    "math": "TBmath",
    "resting": "TBresting",
    "retrieval": "TBretrieval",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_mappings(config: dict) -> list[dict]:
    """Extract normalized DICOM→BIDS mappings from a config.

    Returns a list of dicts with keys:
        protocol_or_series: the DICOM matching key
        datatype, suffix, task_entity
    """
    mappings = []
    for desc in config.get("descriptions", []):
        criteria = desc.get("criteria", {})
        mapping = {
            "datatype": desc.get("datatype"),
            "suffix": desc.get("suffix"),
            "custom_entities": desc.get("custom_entities", ""),
        }
        # Use ProtocolName or SeriesDescription as the match key
        if "ProtocolName" in criteria:
            mapping["match_key"] = ("ProtocolName", criteria["ProtocolName"])
        elif "SeriesDescription" in criteria:
            sn = criteria.get("SeriesNumber")
            mapping["match_key"] = (
                "SeriesDescription",
                criteria["SeriesDescription"],
                sn,
            )
        mappings.append(mapping)
    return mappings


def _load_legacy_config(subject: str, session: str) -> dict | None:
    """Load a legacy config file, trying multiple naming conventions."""
    sub_dir = LEGACY_DIR / subject
    if not sub_dir.is_dir():
        return None

    # Try each naming convention in order of preference
    for suffix in ["_conf.json", "_confedit.json", "_confdeletemath.json", "_conf.editjson"]:
        path = sub_dir / f"{session}{suffix}"
        if path.exists():
            return json.loads(path.read_text())
    return None


# ---------------------------------------------------------------------------
# Tests: structural comparisons for template-generated configs
# ---------------------------------------------------------------------------

class TestLegacyTemplateConfigs:
    """Compare against configs that were generated from templates (not hand-edited).

    These are the _conf.json files (no 'edit' in the name) and should match
    closely with the new system, modulo series numbers.
    """

    @pytest.fixture(autouse=True)
    def check_legacy_dir(self):
        _skip_if_no_legacy()

    def test_ses04_task_structure(self):
        """ses-04 (tb_first) generated config should match task structure."""
        legacy = _load_legacy_config("sub-03", "ses-04")
        if legacy is None:
            pytest.skip("Legacy config not found")

        generated = build_config(
            "sub-03", "ses-04", SESSION_TYPES["tb_first"],
            {"encoding": {"ap": 10, "pa": 11}, "retrieval": {"ap": 28, "pa": 30}},
        )

        # Compare BOLD counts
        legacy_bolds = [d for d in legacy["descriptions"] if d.get("suffix") == "bold"]
        gen_bolds = [d for d in generated["descriptions"] if d.get("suffix") == "bold"]
        assert len(gen_bolds) == len(legacy_bolds)

        # Compare task labels (should match since ses-04 was template-generated)
        legacy_tasks = sorted(d.get("custom_entities", "") for d in legacy_bolds)
        gen_tasks = sorted(d.get("custom_entities", "") for d in gen_bolds)
        assert gen_tasks == legacy_tasks

    def test_ses04_sbref_count(self):
        """ses-04 should have matching SBRef count."""
        legacy = _load_legacy_config("sub-03", "ses-04")
        if legacy is None:
            pytest.skip("Legacy config not found")

        generated = build_config(
            "sub-03", "ses-04", SESSION_TYPES["tb_first"],
            {"encoding": {"ap": 10, "pa": 11}, "retrieval": {"ap": 28, "pa": 30}},
        )

        legacy_sbrefs = [d for d in legacy["descriptions"] if d.get("suffix") == "sbref"]
        gen_sbrefs = [d for d in generated["descriptions"] if d.get("suffix") == "sbref"]
        assert len(gen_sbrefs) == len(legacy_sbrefs)

    def test_ses06_tb_middle_task_count(self):
        """ses-06 (tb_middle) should have same number of functional descriptions."""
        legacy = _load_legacy_config("sub-03", "ses-06")
        if legacy is None:
            pytest.skip("Legacy config not found")

        generated = build_config(
            "sub-03", "ses-06", SESSION_TYPES["tb_middle"],
            {"encoding": {"ap": 5, "pa": 6}, "retrieval": {"ap": 20, "pa": 21}},
        )

        legacy_func = [d for d in legacy["descriptions"] if d["datatype"] == "func"]
        gen_func = [d for d in generated["descriptions"] if d["datatype"] == "func"]
        assert len(gen_func) == len(legacy_func)

    def test_ses20_naturalistic_task_structure(self):
        """ses-20 (naturalistic_fm) task labels should match legacy."""
        legacy = _load_legacy_config("sub-03", "ses-20")
        if legacy is None:
            pytest.skip("Legacy config not found")

        # ses-20 for sub-03 uses series_number (FM variant) based on legacy config
        generated = build_config(
            "sub-03", "ses-20", SESSION_TYPES["naturalistic_fm"],
            {"encoding": {"ap": 13, "pa": 15}, "retrieval": {"ap": 29, "pa": 31}},
        )

        legacy_bolds = [d for d in legacy["descriptions"] if d.get("suffix") == "bold"]
        gen_bolds = [d for d in generated["descriptions"] if d.get("suffix") == "bold"]
        assert len(gen_bolds) == len(legacy_bolds)

        legacy_tasks = sorted(d.get("custom_entities", "") for d in legacy_bolds)
        gen_tasks = sorted(d.get("custom_entities", "") for d in gen_bolds)
        assert gen_tasks == legacy_tasks

    def test_ses04_fmap_series_numbers(self):
        """Fieldmap series numbers should match the legacy config."""
        legacy = _load_legacy_config("sub-03", "ses-04")
        if legacy is None:
            pytest.skip("Legacy config not found")

        # Extract series numbers from legacy
        legacy_fmaps = [d for d in legacy["descriptions"] if d["datatype"] == "fmap"]
        legacy_series = sorted(d["criteria"].get("SeriesNumber") for d in legacy_fmaps)

        generated = build_config(
            "sub-03", "ses-04", SESSION_TYPES["tb_first"],
            {"encoding": {"ap": 10, "pa": 11}, "retrieval": {"ap": 28, "pa": 30}},
        )
        gen_fmaps = [d for d in generated["descriptions"] if d["datatype"] == "fmap"]
        gen_series = sorted(d["criteria"].get("SeriesNumber") for d in gen_fmaps)

        assert gen_series == legacy_series

    def test_generated_always_has_taskname(self):
        """Verify our generated configs always include TaskName (fixing P0 bug)."""
        for session_type in ("tb_first", "tb_middle", "tb_last"):
            config = build_config(
                "sub-03", "ses-06", SESSION_TYPES[session_type],
                FMAP_INFO_STANDARD if session_type != "tb_last" else FMAP_INFO_STANDARD,
            )
            for desc in config["descriptions"]:
                if desc["datatype"] == "func":
                    assert "TaskName" in desc["sidecar_changes"], (
                        f"Missing TaskName in {session_type}/{desc['id']}"
                    )


FMAP_INFO_STANDARD = {
    "encoding": {"ap": 5, "pa": 6},
    "retrieval": {"ap": 20, "pa": 21},
}


class TestLegacyBugDocumentation:
    """Tests that explicitly document known bugs in legacy configs.

    These tests READ legacy configs and verify the bugs exist, ensuring
    our KNOWN_LEGACY_BUGS dict is accurate. If a legacy config is fixed,
    the corresponding test should fail — prompting removal of the bug entry.
    """

    @pytest.fixture(autouse=True)
    def check_legacy_dir(self):
        _skip_if_no_legacy()

    def test_confedit_missing_taskname(self):
        """Verify that _confedit files actually lack TaskName."""
        for (subject, session), bugs in KNOWN_LEGACY_BUGS.items():
            if not bugs.get("missing_taskname"):
                continue
            legacy = _load_legacy_config(subject, session)
            if legacy is None:
                continue
            for desc in legacy["descriptions"]:
                if desc["datatype"] == "func":
                    assert "TaskName" not in desc.get("sidecar_changes", {}), (
                        f"{subject}/{session} has TaskName — remove from "
                        f"KNOWN_LEGACY_BUGS if fixed"
                    )

    def test_confedit_missing_task_prefix(self):
        """Verify that _confedit files use bare task labels (no TB/NAT prefix)."""
        for (subject, session), bugs in KNOWN_LEGACY_BUGS.items():
            if not bugs.get("task_prefix_missing"):
                continue
            legacy = _load_legacy_config(subject, session)
            if legacy is None:
                continue
            for desc in legacy["descriptions"]:
                entity = desc.get("custom_entities", "")
                if entity.startswith("task-"):
                    task_val = entity.split("-", 1)[1]
                    # Should NOT have TB/NAT prefix (that's the bug)
                    assert not task_val.startswith("TB"), (
                        f"{subject}/{session} has TB prefix — remove from "
                        f"KNOWN_LEGACY_BUGS if fixed"
                    )
                    assert not task_val.startswith("NAT"), (
                        f"{subject}/{session} has NAT prefix — remove from "
                        f"KNOWN_LEGACY_BUGS if fixed"
                    )
